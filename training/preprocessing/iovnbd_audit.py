"""Per-session integrity and physical-plausibility audit for IO-VNBD.

Every function takes the canonical SI frames produced by ``iovnbd.to_si_phone`` /
``to_si_vehicle`` and returns plain JSON-serialisable dicts. Nothing here modifies data;
it measures it. Thresholds are named constants so the report can state them.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from training.preprocessing.iovnbd import STANDARD_GRAVITY_MPS2

# WGS84 normal gravity (docs/navigation_math.md sections 2 and 5.1). Phase 3 will generate
# these from the single constants source; duplicated here ONLY for the Phase 2 audit, and
# asserted equal to the doc values in tests/unit/test_iovnbd_audit.py.
WGS84_GAMMA_E = 9.7803253359
WGS84_K = 1.93185265241e-3
WGS84_E2 = 6.69437999014e-3
FREE_AIR_PER_M = 3.086e-6

GAP_S = 0.15  # dt above this is a gap at a nominal 10 Hz
BREAK_S = 1.0  # dt above this (or < 0) breaks a continuous trajectory segment
SMOOTH_SAMPLES = 10  # 1 s centred moving average before correlating gyro vs CAN yaw rate
ALIGN_MAX_LAG_S = 400.0  # wide search: clock errors up to ~314 s were observed
ALIGN_MIN_CORR = 0.5  # below this the lag estimate is not trusted
ALIGN_MAX_CORRECTION_S = 2.0  # |phone clock correction| accepted as "time join verified"
MOVING_MPS = 3.0  # vehicle speed above which yaw-rate / axis correlations are computed
STATIONARY_MPS = 0.05
DT_BINS = [-np.inf, -1e-9, 1e-9, 0.05, 0.095, 0.105, 0.15, 1.0, np.inf]
DT_BIN_LABELS = ["<0", "0", "(0,50ms]", "(50,95ms]", "(95,105ms]", "(105,150ms]",
                 "(150ms,1s]", ">1s"]


def normal_gravity(lat_deg: float, h_m: float) -> float:
    """Somigliana normal gravity with free-air correction [m/s^2]."""
    s2 = math.sin(math.radians(lat_deg)) ** 2
    return WGS84_GAMMA_E * (1 + WGS84_K * s2) / math.sqrt(1 - WGS84_E2 * s2) - FREE_AIR_PER_M * h_m


def _f(x) -> float | None:
    """JSON-safe float."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def dt_stats(t_s: np.ndarray) -> dict:
    """Timing audit on a timestamp vector [s]: real dt, never a nominal rate."""
    dt = np.diff(t_s)
    dt = dt[np.isfinite(dt)]
    counts, _ = np.histogram(dt, bins=DT_BINS)
    breaks = (dt < -1e-9) | (dt > BREAK_S)  # dt == 0 is a duplicate, counted separately
    seg_edges = np.flatnonzero(breaks)
    # continuous segments: durations between breaks
    starts = np.concatenate([[0], seg_edges + 1])
    ends = np.concatenate([seg_edges, [len(t_s) - 1]])
    seg_dur = np.array([t_s[e] - t_s[s] for s, e in zip(starts, ends) if e > s])
    return {
        "n_intervals": int(dt.size),
        "dt_median_s": _f(np.median(dt)) if dt.size else None,
        "dt_p01_s": _f(np.percentile(dt, 1)) if dt.size else None,
        "dt_p99_s": _f(np.percentile(dt, 99)) if dt.size else None,
        "dt_min_s": _f(dt.min()) if dt.size else None,
        "dt_max_s": _f(dt.max()) if dt.size else None,
        "effective_rate_hz": _f(dt.size / (t_s[-1] - t_s[0])) if t_s[-1] > t_s[0] else None,
        "n_backwards": int((dt < -1e-9).sum()),
        "n_duplicate_t": int((np.abs(dt) <= 1e-9).sum()),
        "n_gaps_gt_150ms": int((dt > GAP_S).sum()),
        "n_gaps_gt_1s": int((dt > BREAK_S).sum()),
        "dt_histogram": dict(zip(DT_BIN_LABELS, map(int, counts))),
        "n_continuous_segments": int(seg_dur.size),
        "longest_segment_s": _f(seg_dur.max()) if seg_dur.size else 0.0,
    }


def nan_counts(df: pd.DataFrame) -> dict[str, int]:
    return {c: int(v) for c, v in df.isna().sum().items()}


def audit_phone(p: pd.DataFrame) -> dict:
    t = p["t_rel_s"].to_numpy()
    out: dict = {"n_rows": len(p), "duration_s": _f(t[-1] - t[0])}
    out["timing"] = dt_stats(t)
    # the DATE column and TIME SINCE START are two clocks; they must agree
    loc = (p["t_local"] - p["t_local"].iloc[0]).dt.total_seconds().to_numpy()
    out["date_vs_trel_max_abs_s"] = _f(np.nanmax(np.abs(loc - (t - t[0]))))
    out["t_start_utc"] = str(p["t_utc"].iloc[0])
    out["t_end_utc"] = str(p["t_utc"].iloc[-1])
    out["nan"] = nan_counts(p)
    out["n_exact_duplicate_rows"] = int(p.drop(columns=["row_idx"]).duplicated().sum())

    lat, lon = p["gps_lat_deg"].to_numpy(), p["gps_lon_deg"].to_numpy()
    zero_fix = (lat == 0) & (lon == 0)
    changed = np.concatenate([[True], (np.diff(lat) != 0) | (np.diff(lon) != 0)])
    upd_t = t[changed & ~zero_fix]
    used = p["gps_sats_used"].astype("float64").to_numpy()
    vis = p["gps_sats_visible"].astype("float64").to_numpy()
    both = np.isfinite(used) & np.isfinite(vis)
    out["gnss"] = {
        "n_zero_fix_rows": int(zero_fix.sum()),
        "n_position_updates": int(upd_t.size),
        "update_interval_median_s": _f(np.median(np.diff(upd_t))) if upd_t.size > 1 else None,
        "accuracy_m_p50": _f(p["gps_accuracy_m"].median()),
        "accuracy_m_p95": _f(p["gps_accuracy_m"].quantile(0.95)),
        "accuracy_m_max": _f(p["gps_accuracy_m"].max()),
        "sats_used_p05": _f(np.nanpercentile(used, 5)) if both.any() else None,
        "sats_used_p50": _f(np.nanmedian(used)) if both.any() else None,
        "n_sats_used_gt_visible": int((used[both] > vis[both]).sum()),
        "n_sats_excel_recovered": int(p["gps_sats_excel_recovered"].sum()),
        "speed_mps_max": _f(p["gps_speed_mps"].max()),
        "lat_mean_deg": _f(np.nanmean(np.where(zero_fix, np.nan, lat))),
        "alt_mean_m": _f(np.nanmean(np.where(zero_fix, np.nan, p["gps_alt_m"]))),
    }

    g = np.sqrt(p["grav_x_mps2"] ** 2 + p["grav_y_mps2"] ** 2 + p["grav_z_mps2"] ** 2).to_numpy()
    a = np.sqrt(p["acc_x_mps2"] ** 2 + p["acc_y_mps2"] ** 2 + p["acc_z_mps2"] ** 2).to_numpy()
    w = np.sqrt(p["gyro_x_radps"] ** 2 + p["gyro_y_radps"] ** 2 + p["gyro_z_radps"] ** 2).to_numpy()
    gamma = normal_gravity(out["gnss"]["lat_mean_deg"] or 52.4, out["gnss"]["alt_mean_m"] or 0.0)
    out["gravity_channel"] = {
        "norm_mean_mps2": _f(np.nanmean(g)),
        "norm_std_mps2": _f(np.nanstd(g)),
        "norm_p01_mps2": _f(np.nanpercentile(g, 1)),
        "norm_p99_mps2": _f(np.nanpercentile(g, 99)),
        "local_normal_gravity_mps2": _f(gamma),
        "mean_minus_g0": _f(np.nanmean(g) - STANDARD_GRAVITY_MPS2),
        "mean_minus_local": _f(np.nanmean(g) - gamma),
        "dominant_axis": ["x", "y", "z"][int(np.argmax(np.abs(
            p[["grav_x_mps2", "grav_y_mps2", "grav_z_mps2"]].mean().to_numpy())))],
    }
    out["kinematics"] = {
        "acc_norm_max_mps2": _f(np.nanmax(a)),
        "acc_norm_p99_mps2": _f(np.nanpercentile(a, 99)),
        "gyro_norm_max_radps": _f(np.nanmax(w)),
        "gyro_norm_p99_radps": _f(np.nanpercentile(w, 99)),
    }
    return out


def audit_vehicle(v: pd.DataFrame) -> dict:
    t = v["t_utc_tod_s"].to_numpy().copy()
    wrap = np.flatnonzero(np.diff(t) < -43200)  # midnight rollover
    for i in wrap:
        t[i + 1:] += 86400.0
    out: dict = {"n_rows": len(v), "duration_s": _f(t[-1] - t[0])}
    out["timing"] = dt_stats(t)
    out["n_midnight_rollovers"] = int(wrap.size)
    out["sample_period_values"] = {
        str(k): int(n) for k, n in v["sample_period_s"].round(4).value_counts().head(5).items()
    }
    out["nan"] = nan_counts(v)
    spd = v["vbox_speed_mps"].to_numpy()
    dt = np.diff(t, prepend=t[0])
    good = (dt > 0) & (dt <= BREAK_S)
    out["distance_vbox_km"] = _f(np.nansum(np.where(good, spd * dt, 0.0)) / 1000.0)
    out["vbox_speed_mps_max"] = _f(np.nanmax(spd))
    out["vbox_sats_p05"] = _f(v["vbox_sats"].astype("float64").quantile(0.05))
    out["channels"] = {
        c: {"min": _f(v[c].min()), "max": _f(v[c].max()), "p50": _f(v[c].median())}
        for c in ("vbox_height_raw", "vbox_vvel_raw", "steering_angle_deg", "yaw_rate_radps",
                  "can_long_accel_mps2", "can_lat_accel_mps2", "brake_pressure_pa",
                  "accel_pedal_raw", "gear", "engine_rpm")
    }
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return None
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Least-squares y = k x (through origin)."""
    m = np.isfinite(x) & np.isfinite(y)
    den = float(np.sum(x[m] ** 2))
    return float(np.sum(x[m] * y[m]) / den) if m.sum() >= 50 and den > 0 else None


def vehicle_utc(p: pd.DataFrame, v: pd.DataFrame) -> pd.Series:
    """Absolute UTC timestamps for the vehicle log.

    VBOX records only UTC time-of-day. The date comes from the phone's first UTC
    timestamp minus the row-paired clock offset, and midnight rollovers are unwrapped.
    Row counts and clock offsets never matter here: this uses VBOX time only.
    """
    tod = v["t_utc_tod_s"].to_numpy().copy()
    for i in np.flatnonzero(np.diff(tod) < -43200):
        tod[i + 1:] += 86400.0
    p_tod0 = (p["t_utc"].iloc[0] - p["t_utc"].iloc[0].normalize()).total_seconds()
    day0 = p["t_utc"].iloc[0].normalize()
    # if the phone started just after midnight but VBOX just before, step the date back
    if p_tod0 - tod[0] < -43200:
        day0 -= pd.Timedelta(days=1)
    elif p_tod0 - tod[0] > 43200:
        day0 += pd.Timedelta(days=1)
    return (day0 + pd.to_timedelta(np.round(tod * 1000), unit="ms")).astype(
        "datetime64[ms, UTC]")


ALIGN_TOLERANCE_S = 0.06  # nearest phone sample must lie within this of the VBOX sample


def time_align(p: pd.DataFrame, v: pd.DataFrame) -> pd.DataFrame:
    """Join phone and vehicle rows on UTC time (nearest, within ALIGN_TOLERANCE_S).

    This is the ONLY correct way to pair S and V rows -- see audit_pair: the row
    pairing in the "Synchronised" files is off by the phone's per-session clock error.
    """
    vt = v.assign(t_utc=vehicle_utc(p, v)).sort_values("t_utc")
    ps = p.sort_values("t_utc", kind="stable")
    ps = ps[ps["t_utc"].notna()]
    return pd.merge_asof(
        vt, ps, on="t_utc", direction="nearest", suffixes=("_veh", "_ph"),
        tolerance=pd.Timedelta(seconds=ALIGN_TOLERANCE_S),
    )


def _xcorr_lag(x: np.ndarray, ref: np.ndarray, max_lag: int) -> tuple[int | None, float | None]:
    """Lag L maximising corr(x[t+L], ref[t]) over |L| <= max_lag; positive => x lags ref."""
    best, best_c = None, None
    for L in range(-max_lag, max_lag + 1):
        c = _corr(x[L:], ref[:ref.size - L]) if L >= 0 else _corr(x[:L], ref[-L:])
        if c is not None and (best_c is None or c > best_c):
            best, best_c = L, c
    return best, best_c


def _smooth(x: np.ndarray, k: int = SMOOTH_SAMPLES) -> np.ndarray:
    return pd.Series(x).rolling(k, center=True, min_periods=1).mean().to_numpy()


def fft_xcorr_lag(x: np.ndarray, ref: np.ndarray, max_lag: int) -> tuple[int, float | None]:
    """Wide-range lag via FFT: L maximising corr(x[t+L], ref[t]). NaN treated as 0."""
    x = np.nan_to_num(x - np.nanmean(x))
    ref = np.nan_to_num(ref - np.nanmean(ref))
    n = x.size
    max_lag = max(0, min(max_lag, n // 2))
    nfft = 1 << (2 * n).bit_length()
    c = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(ref, nfft)), nfft)
    lags = np.r_[0:max_lag + 1, -max_lag:0]
    vals = np.r_[c[:max_lag + 1], c[nfft - max_lag:] if max_lag else np.array([])]
    L = int(lags[int(np.argmax(vals))])
    return L, (_corr(x[L:], ref[:n - L]) if L >= 0 else _corr(x[:L], ref[-L:]))


def _axis_scan(cols: dict[str, np.ndarray], ref: np.ndarray, mask: np.ndarray) -> dict:
    corr = {ax: _corr(cols[ax][mask], ref[mask]) for ax in cols}
    gain = {ax: _slope(ref[mask], cols[ax][mask]) for ax in cols}
    valid = {k: c for k, c in corr.items() if c is not None}
    best = max(valid, key=lambda k: abs(valid[k])) if valid else None
    return {"corr": corr, "gain": gain, "best_axis": best,
            "best_sign": (1 if valid[best] > 0 else -1) if best else None}


def audit_pair(p: pd.DataFrame, v: pd.DataFrame) -> dict:
    """Cross-checks that need both logs of a session.

    Part 1 measures how wrong the dataset's ROW pairing is. Part 2 re-pairs the rows by
    UTC time and does every physical cross-check on the time-aligned data.
    """
    out: dict = {"rows_equal": len(p) == len(v), "n_rows_phone": len(p), "n_rows_vehicle": len(v)}

    # ---- Part 1: row pairing as delivered --------------------------------------------
    n = min(len(p), len(v))
    tod = (p["t_utc"].iloc[:n] - p["t_utc"].iloc[:n].dt.normalize()).dt.total_seconds().to_numpy()
    off = tod - v["t_utc_tod_s"].iloc[:n].to_numpy()
    off = np.where(off > 43200, off - 86400, np.where(off < -43200, off + 86400, off))
    out["clock_offset_s"] = {  # phone DATE (as UTC) minus VBOX UTC, on the SAME row index
        "median": _f(np.nanmedian(off)), "p01": _f(np.nanpercentile(off, 1)),
        "p99": _f(np.nanpercentile(off, 99)),
        "drift_first_to_last": _f(off[-1] - off[0]) if n > 1 else None,
    }
    yaw_row = v["yaw_rate_radps"].iloc[:n].to_numpy()
    mov_row = np.nan_to_num(v["can_speed_mps"].iloc[:n].to_numpy()) > MOVING_MPS
    out["row_paired_gyro_y_vs_yaw_corr"] = _corr(
        _smooth(p["gyro_y_radps"].iloc[:n].to_numpy())[mov_row], _smooth(yaw_row)[mov_row])

    # Alignment evidence: find the lag between phone gyro and CAN yaw rate over a WIDE
    # window on the row-paired series, then convert it into the correction the phone
    # clock would need for a UTC-time join to be exact:  c = -(lag + row_offset).
    # c ~ 0  => phone DATE clock is right and joining on UTC time is correct.
    align: dict = {"method": "gyro_y vs CAN yaw rate, 1 s smoothed, moving rows, FFT xcorr"}
    if mov_row.sum() >= 100:
        gy = np.where(mov_row, _smooth(np.nan_to_num(p["gyro_y_radps"].iloc[:n].to_numpy())), 0.0)
        yw = np.where(mov_row, _smooth(np.nan_to_num(yaw_row)), 0.0)
        lag, c = fft_xcorr_lag(gy, yw, int(ALIGN_MAX_LAG_S * 10))
        row_off = out["clock_offset_s"]["median"]
        align.update(lag_s=lag / 10.0, corr=c,
                     phone_clock_correction_s=_f(-(lag / 10.0 + row_off)))
        if c is None or c < ALIGN_MIN_CORR:
            align["status"] = "unverified_weak_signal"
        elif abs(align["phone_clock_correction_s"]) <= ALIGN_MAX_CORRECTION_S:
            align["status"] = "time_join_verified"
        else:
            align["status"] = "time_join_contradicted"
    else:
        align["status"] = "unverified_vehicle_not_moving"
    out["alignment"] = align

    # ---- Part 2: time-aligned ---------------------------------------------------------
    a = time_align(p, v)
    matched = a["row_idx_ph"].notna().to_numpy()
    out["time_aligned"] = {
        "n_vehicle_rows": len(a), "n_matched": int(matched.sum()),
        "match_fraction": _f(matched.mean()) if len(a) else None,
    }
    if (out["time_aligned"]["match_fraction"] or 0.0) < 0.5 and align["status"] != "time_join_verified":
        # The phone clock does not even overlap the VBOX clock: a UTC join is impossible.
        align["status"] = "unverified_time_join_no_overlap"
    a = a[matched]
    speed = a["can_speed_mps"].to_numpy()
    moving = np.isfinite(speed) & (speed > MOVING_MPS)
    out["n_moving_rows"] = int(moving.sum())

    yaw = _smooth(a["yaw_rate_radps"].to_numpy())
    gyro_raw = {ax: a[f"gyro_{ax}_radps"].to_numpy() for ax in "xyz"}
    gyro = {ax: _smooth(g) for ax, g in gyro_raw.items()}
    scan = _axis_scan(gyro, yaw, moving)
    if scan["best_axis"]:
        g = gyro[scan["best_axis"]] * scan["best_sign"]
        lag, c = _xcorr_lag(np.where(moving, g, 0.0), np.where(moving, yaw, 0.0), 30)
        scan["residual_lag_s"] = None if lag is None else lag / 10.0
        scan["corr_at_residual_lag"] = c
    out["gyro_vs_can_yaw_rate"] = scan

    lin = {ax: (a[f"acc_{ax}_mps2"] - a[f"grav_{ax}_mps2"]).to_numpy() for ax in "xyz"}
    lin_s = {ax: _smooth(x) for ax, x in lin.items()}
    out["linacc_vs_can_long"] = _axis_scan(
        lin_s, _smooth(a["can_long_accel_mps2"].to_numpy()), moving)
    out["linacc_vs_can_lat"] = _axis_scan(
        lin_s, _smooth(a["can_lat_accel_mps2"].to_numpy()), moving)

    # Stationary: every vehicle speed source reads zero
    wcols = ["wheel_speed_fl_raw", "wheel_speed_fr_raw", "wheel_speed_rl_raw", "wheel_speed_rr_raw"]
    wheels = a[wcols].abs().max(axis=1).to_numpy()
    stat = ((np.abs(speed) <= STATIONARY_MPS) & (wheels == 0)
            & (a["vbox_speed_mps"].to_numpy() <= 0.3))
    acc = np.sqrt(sum(a[f"acc_{ax}_mps2"].to_numpy() ** 2 for ax in "xyz"))
    linn = np.sqrt(sum(lin[ax] ** 2 for ax in "xyz"))
    out["stationary"] = {
        "n_rows": int(stat.sum()),
        "acc_norm_mean_mps2": _f(np.nanmean(acc[stat])) if stat.any() else None,
        "acc_norm_std_mps2": _f(np.nanstd(acc[stat])) if stat.any() else None,
        "linacc_norm_mean_mps2": _f(np.nanmean(linn[stat])) if stat.any() else None,
        "gyro_mean_radps": {ax: _f(np.nanmean(gyro_raw[ax][stat])) if stat.any() else None
                            for ax in "xyz"},
    }

    # Unit-label evidence for VBOX/CAN channels whose labels look wrong
    vb = a["vbox_speed_mps"].to_numpy()
    fast = np.isfinite(vb) & (vb > 5.0)
    wmean = a[wcols].mean(axis=1).to_numpy()
    h = a["vbox_height_raw"].to_numpy()
    tv = (a["t_utc"] - a["t_utc"].iloc[0]).dt.total_seconds().to_numpy() if len(a) else np.array([])
    k = 10  # 1 s baseline for dh/dt, to beat height quantisation
    dhdt = (h[k:] - h[:-k]) / (tv[k:] - tv[:-k]) if h.size > k else np.array([])
    vv = a["vbox_vvel_raw"].to_numpy()[k // 2: k // 2 + dhdt.size]
    gnss_lag, gnss_c = _xcorr_lag(a["gps_speed_mps"].to_numpy(), vb, 80)
    out["unit_evidence"] = {
        "wheel_raw_over_vbox_kmh_median": _f(np.nanmedian(wmean[fast] / (vb[fast] * 3.6)))
        if fast.any() else None,
        "wheel_raw_over_can_speed_kmh_median": _f(np.nanmedian(
            wmean[fast] / (speed[fast] * 3.6))) if fast.any() else None,
        "height_raw_minus_phone_alt_median": _f(np.nanmedian(h - a["gps_alt_m"].to_numpy())),
        "dhdt_mps_over_vvel_raw_slope": _slope(vv, dhdt) if dhdt.size else None,
        "phone_gnss_speed_lag_s_vs_vbox": None if gnss_lag is None else gnss_lag / 10.0,
        "phone_gnss_speed_corr_at_lag": gnss_c,
    }
    return out
