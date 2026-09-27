"""Phone <-> vehicle synchronisation on UTC time (never on row index).

Phase 2 proved the "Synchronised" S/V files are misaligned by row index (up to 314 s), and
that joining on UTC time is correct wherever the data can verify it. This module builds,
per session, ONE frame on the phone's timeline:

1. Phone rows: drop rows without a timestamp and exact duplicate rows (both counted),
   order by time.
2. Clock: ``t_utc = t_utc_phone_raw + c``. ``c`` is the per-session phone-clock
   correction measured in Phase 2 (gyro vs CAN yaw rate); it is applied ONLY where that
   measurement was verified, else ``c = 0`` and ``alignment_verified`` is False.
3. Phone GNSS latency: the phone's GNSS fields lag the true motion (Phase 2 median 4.1 s).
   The fix is kept exactly as received -- shifting it earlier would feed future data to
   a real-time filter -- and each row carries ``ph_gnss_epoch_utc = t_utc - latency``, the
   instant the fix actually describes (delayed-measurement form). ``latency`` is measured
   per session (phone GNSS speed vs VBOX speed) or, if that estimate is unreliable, the
   pooled median of the reliable sessions; the source is recorded.
4. Ground truth: VBOX/CAN columns joined by nearest UTC time (<= 60 ms) as ``gt_*``.
   ``gt_wheel_speed_mps`` uses the Phase 3 empirical calibration (rear, non-driven axle).
5. Segments: a new ``segment_id`` wherever the real ``dt`` exceeds 1 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.preprocessing.iovnbd_audit import BREAK_S, fft_xcorr_lag

JOIN_TOLERANCE_S = 0.06
GNSS_LATENCY_SEARCH_S = 10.0
GNSS_LATENCY_MIN_CORR = 0.9
GNSS_LATENCY_RANGE_S = (0.0, 10.0)
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
    "ph_gnss_epoch_utc": "UTC instant the logged fix describes (t_utc - latency)",
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
    gnss_latency_s: float | None = None
    gnss_latency_source: str = ""
    gnss_latency_measured_s: float | None = None
    gnss_latency_measured_corr: float | None = None
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


def measure_gnss_latency(p: pd.DataFrame, t_aligned: pd.Series, v: pd.DataFrame
                         ) -> tuple[float | None, float | None]:
    """Latency [s] of phone GNSS speed behind VBOX speed on the aligned phone timeline.

    Returns (latency, corr); latency is None when the estimate is not trustworthy.
    """
    j = _join_vehicle(t_aligned, v[["t_utc", "vbox_speed_mps"]])
    ph = p["gps_speed_mps"].to_numpy()
    vb = j["vbox_speed_mps"].to_numpy()
    ok = np.isfinite(ph) & np.isfinite(vb)
    if ok.sum() < 600 or np.nanmax(vb) < 2.0:
        return None, None
    lag, c = fft_xcorr_lag(np.where(ok, ph, np.nan), np.where(ok, vb, np.nan),
                           int(GNSS_LATENCY_SEARCH_S * 10))
    lat = lag / 10.0
    if c is None or c < GNSS_LATENCY_MIN_CORR or not (
            GNSS_LATENCY_RANGE_S[0] <= lat <= GNSS_LATENCY_RANGE_S[1]):
        return None, c
    return lat, c


def sync_session(sid: str, p_raw: pd.DataFrame, v: pd.DataFrame, align: dict,
                 wheel_k_raw_per_kmh: float, default_gnss_latency_s: float | None
                 ) -> tuple[pd.DataFrame, SyncInfo]:
    info = SyncInfo(session_id=sid)
    p = prepare_phone(p_raw, info)

    info.alignment_status = align.get("alignment_status") or "unknown"
    info.alignment_verified = info.alignment_status == "time_join_verified"
    info.clock_correction_s = float(align["phone_clock_correction_s"]) if info.alignment_verified else 0.0
    t_raw = p["t_utc"]
    t = t_raw + pd.Timedelta(seconds=info.clock_correction_s)

    info.gnss_latency_measured_s, info.gnss_latency_measured_corr = measure_gnss_latency(p, t, v)
    if info.gnss_latency_measured_s is not None:
        info.gnss_latency_s, info.gnss_latency_source = info.gnss_latency_measured_s, "measured"
    elif default_gnss_latency_s is not None:
        info.gnss_latency_s, info.gnss_latency_source = default_gnss_latency_s, "pooled_median"
    else:
        info.gnss_latency_source = "unavailable"

    out = pd.DataFrame({"session_id": sid, "t_utc": t, "t_utc_phone_raw": t_raw})
    for c in ("acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "gyro_x_radps", "gyro_y_radps",
              "gyro_z_radps", "mag_x_uT", "mag_y_uT", "mag_z_uT", "grav_x_mps2", "grav_y_mps2",
              "grav_z_mps2", "orient_azimuth_deg", "orient_pitch_deg", "orient_roll_deg"):
        out[c] = p[c].to_numpy()
    for src, dst in _PHONE_GNSS.items():
        out[dst] = p[src].to_numpy()
    g = p[list(_PHONE_GNSS)[:6]]
    out["ph_gnss_new_fix"] = np.r_[True, (g.iloc[1:].to_numpy() != g.iloc[:-1].to_numpy()).any(axis=1)] \
        if len(g) else np.array([], dtype=bool)
    lat_s = info.gnss_latency_s
    out["ph_gnss_epoch_utc"] = t - pd.Timedelta(seconds=lat_s) if lat_s is not None else pd.NaT

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
