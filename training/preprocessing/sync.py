"""Phone <-> vehicle synchronisation on UTC time (never on row index).

Phase 2 proved the "Synchronised" S/V files are misaligned by row index (up to 314 s), and
that joining on UTC time is correct wherever the data can verify it. This module builds,
per session, ONE frame on the phone's timeline:

1. Phone rows: drop rows without a timestamp and exact duplicate rows (both counted),
   order by time.
2. Clock: ``t_utc = t_utc_phone_raw + c``. ``c`` is the per-session phone-clock
   correction measured in Phase 2 (gyro vs CAN yaw rate); it is applied ONLY where that
   measurement was verified, else ``c = 0`` and ``alignment_verified`` is False.
3. Phone GNSS fix timing. The phone logs a new fix only every ~9 s in 64 of the 67
   moving sessions (1 s in 3) and REPEATS it on every 10 Hz row in between
   (sample-and-hold). Each row therefore carries the epoch of the fix it holds:
   ``ph_gnss_epoch_utc = (time the fix first appeared) - delay`` and
   ``ph_gnss_fix_age_s = t_utc - ph_gnss_epoch_utc``. ``delay`` (reporting delay of a
   NEW fix) is measured per session on new-fix rows only (phone speed vs VBOX speed at
   candidate epochs) -- measured at ~0.1 s -- else the pooled median; the source is
   recorded. The fix values are kept exactly as received (causal).
   Phase 3 v1 of this module correlated the HELD column against truth and reported a
   "4.1 s latency": that was the average age of a held 9 s fix, not a delay. Fixed in
   Phase 4 (PROJECT_STATUS.md 4.x).
4. Ground truth: VBOX/CAN columns joined by nearest UTC time (<= 60 ms) as ``gt_*``.
   ``gt_wheel_speed_mps`` uses the Phase 3 empirical calibration (rear, non-driven axle).
5. Segments: a new ``segment_id`` wherever the real ``dt`` exceeds 1 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.preprocessing.iovnbd_audit import BREAK_S

JOIN_TOLERANCE_S = 0.06
FIX_DELAY_GRID_S = (-2.0, 5.0, 0.1)  # candidate reporting delays of a new fix
FIX_DELAY_MIN_FIXES = 20
FIX_DELAY_MAX_MEDIAN_ERR_MPS = 1.0  # accept the estimate only if the fix matches truth this well
FIX_DELAY_MIN_SPEED_MPS = 3.0
STATIONARY_VBOX_MPS = 0.3
STATIONARY_CAN_MPS = 0.05
SCHEMA_VERSION = "iovnbd-synced-v1"

_PHONE_GNSS = {
    "gps_lat_deg": "ph_gnss_lat_deg", "gps_lon_deg": "ph_gnss_lon_deg",
    "gps_alt_m": "ph_gnss_alt_m", "gps_speed_mps": "ph_gnss_speed_mps",
    "gps_accuracy_m": "ph_gnss_accuracy_m", "gps_bearing_deg": "ph_gnss_bearing_deg",
    "gps_sats_used": "ph_gnss_sats_used", "gps_sats_visible": "ph_gnss_sats_visible",
}
_WHEELS_REAR = ["wheel_speed_rl_raw", "wheel_speed_rr_raw"]
_WHEELS_ALL = ["wheel_speed_fl_raw", "wheel_speed_fr_raw", *_WHEELS_REAR]

SYNCED_UNITS = {
    "session_id": "id", "segment_id": "id", "t_utc": "UTC (clock-corrected)",
    "t_utc_phone_raw": "UTC (phone clock as logged)", "dt_s": "s (real, NaN at segment start)",
    "alignment_verified": "bool", "gt_valid": "bool",
    "acc_x_mps2": "m/s^2", "acc_y_mps2": "m/s^2", "acc_z_mps2": "m/s^2",
    "gyro_x_radps": "rad/s (raw column order; col y = vertical)", "gyro_y_radps": "rad/s",
    "gyro_z_radps": "rad/s", "mag_x_uT": "uT", "mag_y_uT": "uT", "mag_z_uT": "uT",
    "grav_x_mps2": "m/s^2 (normalised to g0)", "grav_y_mps2": "m/s^2", "grav_z_mps2": "m/s^2",
    "orient_azimuth_deg": "deg", "orient_pitch_deg": "deg", "orient_roll_deg": "deg",
    "ph_gnss_lat_deg": "deg WGS84", "ph_gnss_lon_deg": "deg WGS84", "ph_gnss_alt_m": "m",
    "ph_gnss_speed_mps": "m/s", "ph_gnss_accuracy_m": "m", "ph_gnss_bearing_deg": "deg",
    "ph_gnss_sats_used": "count", "ph_gnss_sats_visible": "count",
    "ph_gnss_new_fix": "bool (row where the logged fix changed)",
    "ph_gnss_epoch_utc": "UTC instant the HELD fix describes (first appearance - delay)",
    "ph_gnss_fix_age_s": "s since the held fix's epoch (0..~9 s: sample-and-hold)",
    "gt_lat_deg": "deg WGS84 (VBOX)", "gt_lon_deg": "deg WGS84 (VBOX)",
    "gt_height_msl_m": "m above MSL (VBOX; file label 'km' is wrong)",
    "gt_speed_mps": "m/s (VBOX Doppler)", "gt_heading_deg": "deg (VBOX)",
    "gt_vvel_mps": "m/s (VBOX; file label 'km/hr' is wrong)", "gt_sats": "count (VBOX)",
    "gt_wheel_speed_mps": "m/s (rear-axle mean, empirically calibrated)",
    "gt_can_speed_mps": "m/s (CAN indicated)", "gt_yaw_rate_radps": "rad/s (CAN)",
    "gt_steering_deg": "deg (CAN)", "gt_long_accel_mps2": "m/s^2 (CAN)",
    "gt_lat_accel_mps2": "m/s^2 (CAN)", "gt_brake_on": "0/1 (CAN)",
    "gt_stationary": "bool (VBOX, CAN and all wheels read zero)",
}


@dataclass
class SyncInfo:
    session_id: str
    rows_in: int = 0
    dropped_no_time: int = 0
    dropped_exact_duplicates: int = 0
    reordered_rows: int = 0
    duplicate_timestamps_kept: int = 0
    clock_correction_s: float = 0.0
    alignment_status: str = ""
    alignment_verified: bool = False
    gnss_fix_delay_s: float | None = None
    gnss_fix_delay_source: str = ""
    gnss_fix_delay_measured_s: float | None = None
    gnss_fix_delay_measured_err_mps: float | None = None
    gnss_fix_interval_median_s: float | None = None
    gnss_n_fixes: int = 0
    rows_out: int = 0
    gt_matched: int = 0
    n_segments: int = 0


def prepare_phone(p: pd.DataFrame, info: SyncInfo) -> pd.DataFrame:
    info.rows_in = len(p)
    has_t = p["t_utc"].notna()
    info.dropped_no_time = int((~has_t).sum())
    p = p[has_t]
    dup = p.drop(columns=["row_idx"]).duplicated()
    info.dropped_exact_duplicates = int(dup.sum())
    p = p[~dup]
    order = np.argsort(p["t_utc"].to_numpy(), kind="stable")
    info.reordered_rows = int((order != np.arange(order.size)).sum())
    p = p.iloc[order].reset_index(drop=True)
    info.duplicate_timestamps_kept = int(p["t_utc"].duplicated().sum())
    return p


def _join_vehicle(t: pd.Series, v: pd.DataFrame) -> pd.DataFrame:
    """Nearest vehicle row within JOIN_TOLERANCE_S for each phone time in ``t``."""
    # one resolution on both sides: a fractional-second clock correction promotes ms -> us
    left = pd.DataFrame({"t_utc": pd.Series(t).astype("datetime64[us, UTC]").to_numpy(),
                         "_i": np.arange(len(t))})
    left["t_utc"] = left["t_utc"].astype("datetime64[us, UTC]")
    right = v.sort_values("t_utc").rename(columns={"row_idx": "row_idx_veh"})
    right = right.assign(t_utc=right["t_utc"].astype("datetime64[us, UTC]"))
    out = pd.merge_asof(left, right, on="t_utc", direction="nearest",
                        tolerance=pd.Timedelta(seconds=JOIN_TOLERANCE_S))
    return out.sort_values("_i").reset_index(drop=True)


def fix_changes(p: pd.DataFrame) -> np.ndarray:
    """Rows where the logged phone fix CHANGED (a new fix appeared)."""
    g = p[list(_PHONE_GNSS)[:6]]
    if not len(g):
        return np.array([], dtype=bool)
    return np.r_[True, (g.iloc[1:].to_numpy() != g.iloc[:-1].to_numpy()).any(axis=1)]


def measure_fix_delay(p: pd.DataFrame, t_aligned: pd.Series, v: pd.DataFrame
                      ) -> tuple[float | None, float | None, float | None, int]:
    """Reporting delay of a NEW phone fix vs VBOX truth, from new-fix rows only.

    Grid search over FIX_DELAY_GRID_S for the delay L minimising the median
    |phone speed at first appearance - VBOX speed at (appearance - L)|.
    Returns (delay or None if unreliable, median error, median fix interval, n fixes).
    """
    new = fix_changes(p) & p["gps_speed_mps"].notna().to_numpy()
    tf = t_aligned[new]
    if new.sum() < 2:
        return None, None, None, int(new.sum())
    t0 = t_aligned.iloc[0]
    tf_s = (tf - t0).dt.total_seconds().to_numpy()
    interval = float(np.median(np.diff(tf_s)))
    vt = v[["t_utc", "vbox_speed_mps"]].dropna().sort_values("t_utc")
    tv = (vt["t_utc"].astype("datetime64[us, UTC]") - t0).dt.total_seconds().to_numpy()
    vs = vt["vbox_speed_mps"].to_numpy()
    ph = p["gps_speed_mps"].to_numpy()[new]
    inside = (tf_s > tv[0] + 6) & (tf_s < tv[-1])
    moving = inside & (np.interp(tf_s, tv, vs) > FIX_DELAY_MIN_SPEED_MPS)
    if moving.sum() < FIX_DELAY_MIN_FIXES:
        return None, None, interval, int(new.sum())
    grid = np.round(np.arange(*FIX_DELAY_GRID_S), 3)
    err = [float(np.median(np.abs(ph[moving] - np.interp(tf_s[moving] - L, tv, vs)))) for L in grid]
    k = int(np.argmin(err))
    if err[k] > FIX_DELAY_MAX_MEDIAN_ERR_MPS:
        return None, err[k], interval, int(new.sum())
    return float(grid[k]), err[k], interval, int(new.sum())


def sync_session(sid: str, p_raw: pd.DataFrame, v: pd.DataFrame, align: dict,
                 wheel_k_raw_per_kmh: float, default_fix_delay_s: float | None
                 ) -> tuple[pd.DataFrame, SyncInfo]:
    info = SyncInfo(session_id=sid)
    p = prepare_phone(p_raw, info)

    info.alignment_status = align.get("alignment_status") or "unknown"
    info.alignment_verified = info.alignment_status == "time_join_verified"
    info.clock_correction_s = float(align["phone_clock_correction_s"]) if info.alignment_verified else 0.0
    t_raw = p["t_utc"]
    t = t_raw + pd.Timedelta(seconds=info.clock_correction_s)

    (info.gnss_fix_delay_measured_s, info.gnss_fix_delay_measured_err_mps,
     info.gnss_fix_interval_median_s, info.gnss_n_fixes) = measure_fix_delay(p, t, v)
    if info.gnss_fix_delay_measured_s is not None:
        # A reporting delay cannot be negative: a negative estimate is residual phone-vs-VBOX
        # CLOCK offset (all seen in clock-unverified sessions), not GNSS timing. Clamp so every
        # epoch stays causal (epoch <= time received); the raw value stays in the record.
        d = info.gnss_fix_delay_measured_s
        info.gnss_fix_delay_s = max(d, 0.0)
        info.gnss_fix_delay_source = "measured" if d >= 0 else "measured_clamped_at_0"
    elif default_fix_delay_s is not None:
        info.gnss_fix_delay_s = max(default_fix_delay_s, 0.0)
        info.gnss_fix_delay_source = "pooled_median"
    else:
        info.gnss_fix_delay_source = "unavailable"

    out = pd.DataFrame({"session_id": sid, "t_utc": t, "t_utc_phone_raw": t_raw})
    for c in ("acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "gyro_x_radps", "gyro_y_radps",
              "gyro_z_radps", "mag_x_uT", "mag_y_uT", "mag_z_uT", "grav_x_mps2", "grav_y_mps2",
              "grav_z_mps2", "orient_azimuth_deg", "orient_pitch_deg", "orient_roll_deg"):
        out[c] = p[c].to_numpy()
    for src, dst in _PHONE_GNSS.items():
        out[dst] = p[src].to_numpy()
    new = fix_changes(p)
    out["ph_gnss_new_fix"] = new
    # each row holds the most recent fix: its epoch = that fix's first appearance - delay
    appeared = t.where(pd.Series(new, index=t.index)).ffill()
    if info.gnss_fix_delay_s is not None:
        out["ph_gnss_epoch_utc"] = appeared - pd.Timedelta(seconds=info.gnss_fix_delay_s)
        out["ph_gnss_fix_age_s"] = (t - out["ph_gnss_epoch_utc"]).dt.total_seconds()
    else:
        out["ph_gnss_epoch_utc"] = pd.NaT
        out["ph_gnss_fix_age_s"] = np.nan

    j = _join_vehicle(t, v)
    matched = j["row_idx_veh"].notna().to_numpy()
    info.gt_matched = int(matched.sum())
    out["gt_valid"] = matched
    out["gt_lat_deg"] = j["vbox_lat_deg"].to_numpy()
    out["gt_lon_deg"] = j["vbox_lon_deg"].to_numpy()
    out["gt_height_msl_m"] = j["vbox_height_raw"].to_numpy()
    out["gt_speed_mps"] = j["vbox_speed_mps"].to_numpy()
    out["gt_heading_deg"] = j["vbox_heading_deg"].to_numpy()
    out["gt_vvel_mps"] = j["vbox_vvel_raw"].to_numpy()
    out["gt_sats"] = j["vbox_sats"].to_numpy()
    out["gt_wheel_speed_mps"] = (j[_WHEELS_REAR].mean(axis=1) / (3.6 * wheel_k_raw_per_kmh)).to_numpy()
    out["gt_can_speed_mps"] = j["can_speed_mps"].to_numpy()
    out["gt_yaw_rate_radps"] = j["yaw_rate_radps"].to_numpy()
    out["gt_steering_deg"] = j["steering_angle_deg"].to_numpy()
    out["gt_long_accel_mps2"] = j["can_long_accel_mps2"].to_numpy()
    out["gt_lat_accel_mps2"] = j["can_lat_accel_mps2"].to_numpy()
    out["gt_brake_on"] = j["brake_on"].to_numpy()
    out["gt_stationary"] = (matched
                            & (j["vbox_speed_mps"].to_numpy() <= STATIONARY_VBOX_MPS)
                            & (np.abs(j["can_speed_mps"].to_numpy()) <= STATIONARY_CAN_MPS)
                            & (j[_WHEELS_ALL].abs().max(axis=1).to_numpy() == 0))

    dt = out["t_utc"].diff().dt.total_seconds().to_numpy(copy=True)
    brk = np.r_[True, dt[1:] > BREAK_S] if len(dt) else np.array([], dtype=bool)
    out["segment_id"] = np.cumsum(brk).astype(np.int32) - 1
    dt[brk] = np.nan
    out["dt_s"] = dt
    out["alignment_verified"] = info.alignment_verified
    info.rows_out = len(out)
    info.n_segments = int(brk.sum())
    return out, info
