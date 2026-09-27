"""Unit tests for the IO-VNBD loader and audit helpers.

All inputs here are SYNTHETIC test fixtures written inline (AGENTS.md 2.2): short header
strings and hand-made series that reproduce each parsing hazard found in the real data.
They are never used as data; they only pin the parsing behaviour.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from training.preprocessing import iovnbd as io
from training.preprocessing import iovnbd_audit as audit

# Exact byte sequences observed in the real headers: degree/mu as UTF-8 pairs, the
# superscript two as a single cp1252 byte. Decoded as cp1252, as the loader does.
DEG = b"\xc2\xb0".decode("cp1252")  # 'Â°' mojibake
MU = b"\xce\xbc".decode("cp1252")  # 'Î¼' mojibake
SQ = b"\xb2".decode("cp1252")  # '²'

SYNTHETIC_S_HEADER_YPR = (
    f"GPS LATITUDE (degrees), GPS LONGITUDE (degrees), GPS ALTITUDE (m), GPS SPEED (Kmh), "
    f"GPS ACCURACY (m), GPS ORIENTATION ({DEG}),GPS SATELLITES IN RANGE, TIME SINCE START (ms), "
    f"DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X (m/s{SQ}) , ACCELEROMETER Y (m/s{SQ}), "
    f"ACCELEROMETER Z (m/s{SQ}), GRAVITY X (m/s{SQ}), GRAVITY Y (m/s{SQ}), GRAVITY Z (m/s{SQ}), "
    f"GYROSCOPE Yaw (rad/s), GYROSCOPE Pitch (rad/s), GYROSCOPE Roll (rad/s), "
    f"MAGNETIC FIELD X ({MU}T), MAGNETIC FIELD Y ({MU}T), MAGNETIC FIELD Z ({MU}T), "
    f"ORIENTATION (Yaw) ({DEG}), ORIENTATION (Pitch) ({DEG}), ORIENTATION (Roll ) ({DEG})\r\n"
)
SYNTHETIC_S_HEADER_XYZ = (
    SYNTHETIC_S_HEADER_YPR.replace("GYROSCOPE Yaw", "GYROSCOPE X")
    .replace("GYROSCOPE Pitch", "GYROSCOPE Y").replace("GYROSCOPE Roll", "GYROSCOPE Z")
    .replace("ORIENTATION (Yaw)", "ORIENTATION (Azimuth)")
)
SYNTHETIC_V_HEADER = (
    "No of GPS Satellites Available, Time Since Start of Day (seconds), Latitude (degrees), "
    "Longitude (degrees), Velocity (km/hr), Heading (degrees), Height (km), Vertical velocity "
    "(km/hr), Sample period (seconds), Steering Angle (degrees), Wheel Speed Front Left "
    "(rad/sec), Wheel Speed Front Right (rad/sec), Wheel Speed Rear Left (rad/sec), Wheel "
    "Speed Rear Right (rad/sec), Yaw Rate (deg/sec), Indicated Vehicle Speed (km/hr), "
    "Indicated Longitudinal Acceleration (g), Indicated Lateral Acceleration (g), Handbrake "
    "(0 or 1), Gear Requested (Number fof gear employed 1-5), Gear (Number fof gear employed "
    "1-5), Engine Speed (rev/min), Coolant Temperature (degrees), Clutch Position (0 or 1), "
    "Brake Pressure (psi), Brake Position (0 or 1), Battery Voltage (volts), Air Temperature "
    "(degrees), Accelerator Pedal Position (0 or 1)"
)


# ---- encoding & headers --------------------------------------------------------------

def test_real_header_bytes_fail_strict_utf8_but_decode_cp1252():
    raw = b"ACCELEROMETER X (m/s\xb2) , GPS ORIENTATION (\xc2\xb0)"
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    assert raw.decode("cp1252")


def test_mojibake_repaired_only_where_it_applies():
    assert io.repair_header_token(f"GPS ORIENTATION ({DEG})") == "GPS ORIENTATION (°)"
    assert io.repair_header_token(f"MAGNETIC FIELD X ({MU}T)") == "MAGNETIC FIELD X (μT)"
    assert io.repair_header_token(f"ACCELEROMETER X (m/s{SQ}) ") == f"ACCELEROMETER X (m/s{SQ}) "


@pytest.mark.parametrize("token,key", [
    (f" ACCELEROMETER X (m/s{SQ}) ", "accelerometer x (m/s2)"),
    (f" ORIENTATION (Roll ) ({DEG})", "orientation (roll) (deg)"),
    (f"GPS ORIENTATION ({DEG})", "gps orientation (deg)"),
    (f" MAGNETIC FIELD Z ({MU}T)", "magnetic field z (ut)"),
    ("   GPS   SPEED (Kmh)  ", "gps speed (kmh)"),
])
def test_header_key_normalisation(token, key):
    assert io.header_key(token) == key
    assert io.header_key(token).isascii()


@pytest.mark.parametrize("header", [SYNTHETIC_S_HEADER_YPR, SYNTHETIC_S_HEADER_XYZ])
def test_both_phone_header_variants_map_to_one_schema(header):
    names, raw = io.map_header(header, "S")
    assert names == [c for c, _ in io.PHONE_SCHEMA]
    assert names[15:18] == ["gyro_x_radps", "gyro_y_radps", "gyro_z_radps"]
    assert len(raw) == 24


def test_missing_paren_date_variant_is_accepted_explicitly():
    h = SYNTHETIC_S_HEADER_XYZ.replace("HH-MI-SS_SSS)", "HH-MI-SS_SSS")
    assert io.map_header(h, "S")[0][8] == "date_raw"


def test_vehicle_header_maps():
    names, _ = io.map_header(SYNTHETIC_V_HEADER, "V")
    assert len(names) == 29 and names[14] == "yaw_rate_degps"


def test_unknown_or_wrong_count_header_raises():
    with pytest.raises(io.SchemaError, match="columns"):
        io.map_header("a,b,c", "S")
    with pytest.raises(io.SchemaError, match="column 15"):
        io.map_header(SYNTHETIC_S_HEADER_YPR.replace("GYROSCOPE Yaw", "GYROSCOPE Q"), "S")


# ---- satellites ----------------------------------------------------------------------

def test_satellite_split_and_excel_repair():
    s = pd.Series(["18 / 19", " 7/9 ", "Dec-14", "10-Dec", "May-22", "", None, "garbage"],
                  dtype="string")
    used, vis, rec, malformed = io.parse_satellites(s)
    assert used.tolist()[:5] == [18, 7, 12, 10, 5]
    assert vis.tolist()[:5] == [19, 9, 14, 12, 22]
    assert rec.tolist() == [False, False, True, True, True, False, False, False]
    assert used[5] is pd.NA and used[6] is pd.NA and used[7] is pd.NA
    assert malformed == 1  # only 'garbage'; blanks are missing, not malformed


# ---- time ----------------------------------------------------------------------------

def test_datetime_colon_before_milliseconds():
    t, bad = io.parse_phone_datetime(pd.Series(["2019-09-07 09:13:29:506", "2019-09-07 09:13:29.506", ""]))
    assert t[0] == pd.Timestamp("2019-09-07 09:13:29.506")
    assert pd.isna(t[1]) and pd.isna(t[2])
    assert bad == 1  # the decimal-point form is NOT the dataset format and is counted


def test_local_to_utc_bst_gmt_and_dst_gap():
    local = pd.Series(pd.to_datetime([
        "2019-09-08 10:07:49.546",  # BST, UTC+1
        "2019-12-01 12:00:00.000",  # GMT, UTC+0
        "2019-03-31 01:30:00.000",  # does not exist (clocks jump 01:00 -> 02:00)
    ])).astype("datetime64[ms]")
    utc, lost = io.local_to_utc(local)
    assert utc[0] == pd.Timestamp("2019-09-08 09:07:49.546", tz="UTC")
    assert utc[1] == pd.Timestamp("2019-12-01 12:00:00", tz="UTC")
    assert pd.isna(utc[2]) and lost == 1


def test_speed_conversion_kmh_to_mps():
    assert pytest.approx(10.0, rel=1e-15) == io.KMH_TO_MPS * 36.0


# ---- dedupe --------------------------------------------------------------------------

def test_content_digest_ignores_float_formatting_but_not_values():
    a = pd.DataFrame({"x": [1.539, -6.0, 52.402145], "s": pd.Series(["18 / 19"] * 3, dtype="string")})
    b = pd.DataFrame({"x": [1.5390000000000001, -6.0, 52.402145000000004],
                      "s": pd.Series(["18 / 19"] * 3, dtype="string")})
    c = a.assign(x=[1.540, -6.0, 52.402145])
    assert io.content_digest(a) == io.content_digest(b)
    assert io.content_digest(a) != io.content_digest(c)
    assert io.max_abs_numeric_diff(a, b) < 1e-14


def test_route_group_keeps_segments_together():
    assert io.route_group("S3a") == io.route_group("S3c") == "S3"
    assert io.route_group("Vw14b") == "Vw14"
    assert io.route_group("Vfa01") == "Vfa01"
    assert io.route_group("M") == "M"


# ---- audit helpers -------------------------------------------------------------------

def test_audit_uses_the_single_constants_source():
    # Phase 3: the Phase 2 duplicate constants are gone; the audit delegates to navcore
    import navcore.geometry.geodesy as geodesy
    assert audit._normal_gravity_rad is geodesy.normal_gravity
    assert not hasattr(audit, "WGS84_GAMMA_E")


def test_normal_gravity_reproduces_doc_value():
    # docs/navigation_math.md section 5.2 (corrected in Phase 2): 9.812381 m/s^2
    assert audit.normal_gravity(52.402565, 144.59) == pytest.approx(9.812381, abs=1e-6)
    assert audit.normal_gravity(0.0, 0.0) == pytest.approx(9.7803253359, abs=1e-10)


def test_fft_xcorr_recovers_a_known_shift_synthetic():
    rng = np.random.default_rng(0)
    ref = np.convolve(rng.standard_normal(5000), np.ones(20) / 20, mode="same")
    shift = 137
    x = np.roll(ref, shift)  # x lags ref by `shift` samples
    lag, c = audit.fft_xcorr_lag(x, ref, 400)
    assert lag == shift and c > 0.99


def test_dt_stats_duplicates_do_not_break_segments_but_restarts_do():
    t = np.array([0.0, 0.1, 0.2, 0.2, 0.3, 0.1, 0.2, 5.0, 5.1])  # dup, restart, 4.8 s gap
    st = audit.dt_stats(t)
    assert st["n_duplicate_t"] == 1
    assert st["n_backwards"] == 1
    assert st["n_gaps_gt_1s"] == 1
    assert st["n_continuous_segments"] == 3
    assert math.isclose(st["dt_median_s"], 0.1)


def test_recorded_duration_survives_clock_resets_duplicates_and_gaps():
    # synthetic: 10 s, then TIME SINCE START resets to 0 and runs 5 s more; one duplicate;
    # one 30 s gap. Recorded time is 15 s -- t[-1] - t[0] would say 5 s (the Phase 2 bug).
    t = np.r_[np.arange(0, 10.01, 0.1), np.arange(0, 5.01, 0.1)]
    t = np.insert(t, 50, t[50])
    t = np.r_[t, t[-1] + 30.0]
    assert audit.recorded_duration_s(t) == pytest.approx(15.0, abs=1e-9)
    assert t[-1] - t[0] != pytest.approx(15.0)


def test_fft_xcorr_is_unbiased_for_broad_correlation_peaks_synthetic():
    # Regression (Phase 3): the Phase 2 version maximised the RAW xcorr sum, which is
    # weighted by the overlap n-|L| and pulled broad peaks (slow signals like speed)
    # toward zero lag. A very smooth synthetic signal with a known 30-sample lag:
    rng = np.random.default_rng(1)
    ref = np.convolve(rng.standard_normal(3000), np.ones(400) / 400, mode="same")
    x = np.r_[np.full(30, ref[0]), ref[:-30]]
    lag, c = audit.fft_xcorr_lag(x, ref, 100)
    assert lag == 30 and c > 0.999


def test_phone_gps_speed_is_kept_as_logged_mps_not_divided_by_3p6():
    # Regression (Phase 3): the header says "Kmh" but Android logs m/s; the loader must
    # NOT convert it. SYNTHETIC one-row raw frame.
    names = [c for c, _ in io.PHONE_SCHEMA]
    row = dict.fromkeys(names, 0.0)
    row.update(gps_speed_raw=5.57, t_rel_ms=2922.0, gps_sats_raw="27 / 28",
               date_raw="2019-09-08 10:07:49:546")
    raw = pd.DataFrame([row]).astype({"gps_sats_raw": "string", "date_raw": "string"})
    out = io.to_si_phone(raw, io.ParseReport(path="synthetic", kind="S"))
    assert out["gps_speed_mps"].iloc[0] == 5.57
