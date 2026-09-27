"""Unit tests for Phase 3 preprocessing: column roles, sync, splits, normalisation.

Every frame in this file is a SYNTHETIC FIXTURE built inline with known answers (a known
clock offset, a known GNSS lag, known drive adjacency). None of it is, or is used as, data
(AGENTS.md 2.2).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from training.preprocessing import columns, normalization, splits, sync

T0 = pd.Timestamp("2019-11-06 12:00:00", tz="UTC")


# ---- column roles: the CAN / coordinate rules ---------------------------------------------

def test_inputs_and_ground_truth_are_disjoint():
    assert not set(columns.INPUT_COLUMNS) & set(columns.GT_COLUMNS)
    assert all(c.startswith("gt_") for c in columns.GT_COLUMNS)
    columns.assert_model_inputs(columns.INPUT_COLUMNS)  # the declared inputs pass


@pytest.mark.parametrize("bad", ["gt_wheel_speed_mps", "gt_speed_mps", "gt_yaw_rate_radps",
                                 "wheel_speed_fl_raw", "can_speed_mps", "steering_angle_deg",
                                 "yaw_rate_radps", "vbox_speed_mps", "engine_rpm"])
def test_vehicle_bus_and_ground_truth_can_never_be_inputs(bad):
    with pytest.raises(columns.ColumnRoleError):
        columns.assert_model_inputs(["acc_x_mps2", bad])


def test_undeclared_input_is_rejected():
    with pytest.raises(columns.ColumnRoleError):
        columns.assert_model_inputs(["some_new_feature"])


@pytest.mark.parametrize("bad", ["gt_lat_deg", "gt_lon_deg", "ph_gnss_lat_deg", "delta_lat_deg"])
def test_coordinates_can_never_be_targets(bad):
    with pytest.raises(columns.ColumnRoleError):
        columns.assert_model_targets([bad])
    columns.assert_model_targets(["gt_speed_mps", "gt_wheel_speed_mps"])  # physical: fine


# ---- synthetic phone / vehicle logs ----------------------------------------------------

def synthetic_logs(n=3000, phone_clock_error_s=0.0, gnss_lag_s=3.0, seed=0):
    """SYNTHETIC: a vehicle doing a smooth speed profile, logged by a 'VBOX' (true time)
    and a 'phone' whose DATE clock is off by ``phone_clock_error_s`` and whose GNSS
    speed lags truth by ``gnss_lag_s``. Row indices deliberately do NOT correspond."""
    rng = np.random.default_rng(seed)
    t = T0 + pd.to_timedelta(np.arange(n) * 100, unit="ms")
    speed = 10 + 5 * np.sin(np.arange(n) / 150.0) + np.convolve(
        rng.standard_normal(n), np.ones(50) / 50, mode="same")
    k = round(gnss_lag_s * 10)
    lagged = np.r_[np.full(k, speed[0]), speed[:-k]] if k else speed
    v = pd.DataFrame({
        "row_idx": np.arange(n, dtype=np.int32), "t_utc": t.astype("datetime64[ms, UTC]"),
        "vbox_lat_deg": 52.4, "vbox_lon_deg": -1.5, "vbox_height_raw": 100.0,
        "vbox_speed_mps": speed, "vbox_heading_deg": 90.0, "vbox_vvel_raw": 0.0,
        "vbox_sats": pd.array([12] * n, dtype="Int16"),
        "wheel_speed_fl_raw": speed * 3.6, "wheel_speed_fr_raw": speed * 3.6,
        "wheel_speed_rl_raw": speed * 3.6, "wheel_speed_rr_raw": speed * 3.6,
        "can_speed_mps": speed, "yaw_rate_radps": 0.0, "steering_angle_deg": 0.0,
        "can_long_accel_mps2": 0.0, "can_lat_accel_mps2": 0.0, "brake_on": 0.0,
    })
    shift = 400  # the phone started 40 s later: row i != vehicle row i
    tp = t[shift:] + pd.Timedelta(seconds=phone_clock_error_s)
    m = n - shift
    p = pd.DataFrame({
        "row_idx": np.arange(m, dtype=np.int32), "t_utc": tp.astype("datetime64[ms, UTC]"),
        **{c: rng.standard_normal(m) for c in (
            "acc_x_mps2", "acc_y_mps2", "acc_z_mps2", "gyro_x_radps", "gyro_y_radps",
            "gyro_z_radps", "mag_x_uT", "mag_y_uT", "mag_z_uT", "grav_x_mps2", "grav_y_mps2",
            "grav_z_mps2", "orient_azimuth_deg", "orient_pitch_deg", "orient_roll_deg")},
        "gps_lat_deg": 52.4, "gps_lon_deg": -1.5, "gps_alt_m": 150.0,
        "gps_speed_mps": lagged[shift:], "gps_accuracy_m": 3.0, "gps_bearing_deg": 90.0,
        "gps_sats_used": pd.array([10] * m, dtype="Int16"),
        "gps_sats_visible": pd.array([12] * m, dtype="Int16"),
    })
    return p, v, speed[shift:]


def test_sync_joins_on_time_not_row_index():
    p, v, truth = synthetic_logs()
    out, info = sync.sync_session("SYN", p, v, {"alignment_status": "time_join_verified",
                                               "phone_clock_correction_s": 0.0}, 1.0, None)
    assert info.gt_matched == len(out)
    np.testing.assert_allclose(out["gt_speed_mps"], truth, atol=1e-12)
    # a row-index join would have paired phone row 0 with vehicle row 0 (40 s earlier)
    assert abs(out["gt_speed_mps"].iloc[0] - v["vbox_speed_mps"].iloc[0]) > 1e-3


def test_clock_correction_applied_only_when_verified():
    p, v, truth = synthetic_logs(phone_clock_error_s=-2.0)
    good = {"alignment_status": "time_join_verified", "phone_clock_correction_s": 2.0}
    out, info = sync.sync_session("SYN", p, v, good, 1.0, None)
    assert info.clock_correction_s == 2.0 and info.alignment_verified
    np.testing.assert_allclose(out["gt_speed_mps"], truth, atol=1e-12)
    bad = {"alignment_status": "unverified_weak_signal", "phone_clock_correction_s": 2.0}
    out2, info2 = sync.sync_session("SYN", p, v, bad, 1.0, None)
    assert info2.clock_correction_s == 0.0 and not out2["alignment_verified"].any()


def test_gnss_latency_is_measured_and_epoch_tagged_not_shifted():
    p, v, _ = synthetic_logs(gnss_lag_s=3.0)
    align = {"alignment_status": "time_join_verified", "phone_clock_correction_s": 0.0}
    out, info = sync.sync_session("SYN", p, v, align, 1.0, None)
    assert info.gnss_latency_source == "measured"
    assert info.gnss_latency_s == pytest.approx(3.0, abs=0.1)
    # the fix is kept exactly as received (causal) ...
    np.testing.assert_array_equal(out["ph_gnss_speed_mps"], p["gps_speed_mps"])
    # ... and tagged with the instant it describes
    lag = (out["t_utc"] - out["ph_gnss_epoch_utc"]).dt.total_seconds()
    np.testing.assert_allclose(lag, info.gnss_latency_s)


def test_unreliable_latency_falls_back_to_pooled_value_and_says_so():
    p, v, _ = synthetic_logs()
    p["gps_speed_mps"] = 0.0  # no usable signal
    align = {"alignment_status": "time_join_verified", "phone_clock_correction_s": 0.0}
    _, info = sync.sync_session("SYN", p, v, align, 1.0, 4.2)
    assert info.gnss_latency_source == "pooled_median" and info.gnss_latency_s == 4.2


def test_duplicates_dropped_and_counted_segments_break_on_gaps():
    p, v, _ = synthetic_logs()
    p = pd.concat([p.iloc[:100], p.iloc[99:100], p.iloc[100:]], ignore_index=True)  # 1 dup
    p = p.drop(index=range(1000, 1030))  # 3 s hole -> new segment
    p["row_idx"] = np.arange(len(p), dtype=np.int32)
    align = {"alignment_status": "time_join_verified", "phone_clock_correction_s": 0.0}
    out, info = sync.sync_session("SYN", p, v, align, 1.0, None)
    assert info.dropped_exact_duplicates == 1
    assert info.n_segments == 2 and out["segment_id"].max() == 1
    assert out["dt_s"].isna().sum() == 2  # first row of each segment
    assert out["dt_s"].dropna().max() <= 0.1 + 1e-9


def test_wheel_speed_ground_truth_uses_calibration():
    p, v, truth = synthetic_logs()
    align = {"alignment_status": "time_join_verified", "phone_clock_correction_s": 0.0}
    out, _ = sync.sync_session("SYN", p, v, align, 1.0, None)
    np.testing.assert_allclose(out["gt_wheel_speed_mps"], truth, rtol=1e-12)
    out2, _ = sync.sync_session("SYN", p, v, align, 1.02, None)
    np.testing.assert_allclose(out2["gt_wheel_speed_mps"], truth / 1.02, rtol=1e-12)


# ---- splits ----------------------------------------------------------------------------

def session_table(rows):
    """SYNTHETIC session table: (sid, driver, route_group, start_min, end_min)."""
    return pd.DataFrame([{
        "session_id": s, "driver": d, "route_group": g,
        "start_utc": T0 + pd.Timedelta(minutes=a), "end_utc": T0 + pd.Timedelta(minutes=b),
        "duration_s": (b - a) * 60.0} for s, d, g, a, b in rows])


def test_adjacent_sessions_form_one_drive():
    t = session_table([("a1", "E", "a1", 0, 10), ("a2", "E", "a2", 10.2, 20),  # 12 s gap
                       ("b1", "E", "b1", 200, 230),                            # 3 h later
                       ("c1", "F", "c1", 10.1, 30),                            # other driver
                       ("s3a", "E", "s3", 500, 510), ("s3b", "E", "s3", 900, 910)])
    d = dict(zip(t["session_id"], splits.assign_drives(t), strict=True))
    assert d["a1"] == d["a2"]
    assert d["b1"] != d["a1"]
    assert d["c1"] != d["a1"]  # adjacency never crosses drivers
    assert d["s3a"] == d["s3b"]  # same route group, even hours apart


def synthetic_fleet():
    rows = []
    for drv, n in (("A", 6), ("B", 2), ("C", 3), ("E", 12)):
        for i in range(n):
            start = i * 300  # 5 h apart: separate drives
            rows.append((f"{drv}{i}", drv, f"{drv}{i}", start, start + 60 + 10 * i))
            rows.append((f"{drv}{i}x", drv, f"{drv}{i}x", start + 60.2 + 10 * i, start + 90 + 10 * i))
    return session_table(rows)


def test_make_split_is_leak_free_and_deterministic():
    t = synthetic_fleet()
    m1, m2 = splits.make_split(t), splits.make_split(t)
    assert m1["splits"] == m2["splits"]
    assert splits.find_leaks(m1) == []
    test_drivers = set(m1["test_drivers"])
    for sid, m in m1["sessions"].items():
        assert (m["split"] == "test") == (m["driver"] in test_drivers), sid
    lo, hi = splits.TEST_FRACTION_BOUNDS
    assert lo <= m1["split_fraction"]["test"] <= hi


def test_find_leaks_detects_every_violation_class():
    m = splits.make_split(synthetic_fleet())
    tr, va = m["splits"]["train"], m["splits"]["val"]
    s = next(x for x in tr if not x.endswith("x"))  # its "x" partner is 12 s later
    bad = json.loads(json.dumps(m))
    bad["sessions"][s + "x"]["split"] = "val"
    bad["splits"]["train"].remove(s + "x")
    bad["splits"]["val"].append(s + "x")
    issues = splits.find_leaks(bad)
    assert any("apart" in i for i in issues)  # temporal adjacency across splits
    assert any("drive_id" in i for i in issues)
    dup = json.loads(json.dumps(m))
    dup["splits"]["val"].append(tr[0])
    assert any("more than one split" in i for i in splits.find_leaks(dup))
    leak = json.loads(json.dumps(m))
    tsid = m["splits"]["test"][0]
    victim = next(x for x in va + tr if m["sessions"][x]["driver"] != m["sessions"][tsid]["driver"])
    leak["sessions"][victim]["driver"] = m["sessions"][tsid]["driver"]
    assert any("test driver" in i for i in splits.find_leaks(leak))


# ---- normalisation -----------------------------------------------------------------------

def test_normalisation_uses_train_only_and_refuses_leaks(tmp_path):
    rng = np.random.default_rng(5)
    manifest = {"splits": {"train": ["t1", "t2"], "val": ["v1"], "test": ["x1"]},
                "sessions": {"t1": {"split": "train"}, "t2": {"split": "train"},
                             "v1": {"split": "val"}, "x1": {"split": "test"}}}
    mp = tmp_path / "split.json"
    mp.write_text(json.dumps(manifest))
    data = {}
    for sid, loc in (("t1", 0.0), ("t2", 1.0), ("v1", 50.0), ("x1", -50.0)):
        df = pd.DataFrame({c: rng.normal(loc, 2.0, 500) for c in columns.IMU_COLUMNS})
        df.loc[3, "acc_x_mps2"] = np.nan  # a NaN must be excluded and counted
        df.to_parquet(tmp_path / f"{sid}.parquet")
        data[sid] = df
    st = normalization.compute_imu_stats(mp, tmp_path)
    train = pd.concat([data["t1"], data["t2"]])
    for c in columns.IMU_COLUMNS:
        assert st["mean"][c] == pytest.approx(train[c].mean(), rel=1e-12)
        assert st["std"][c] == pytest.approx(train[c].std(ddof=0), rel=1e-12)
    assert st["n_nonfinite_excluded"]["acc_x_mps2"] == 2
    assert st["sessions_used"] == ["t1", "t2"]
    assert abs(st["mean"]["acc_y_mps2"] - 0.5) < 0.3  # val/test (+-50) would dominate
    with pytest.raises(normalization.SplitLeakError):
        normalization.compute_imu_stats(mp, tmp_path, sessions=["t1", "v1"])
    with pytest.raises(normalization.SplitLeakError):
        normalization.compute_imu_stats(mp, tmp_path, sessions=["x1"])
    with pytest.raises(columns.ColumnRoleError):
        normalization.compute_imu_stats(mp, tmp_path, channels=["gt_speed_mps"])
