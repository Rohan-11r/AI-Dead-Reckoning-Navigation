"""Mounting alignment and sensor containers. All inputs are SYNTHETIC with known answers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from navcore.alignment import mounting as MT
from navcore.geometry.frames import Frame
from navcore.geometry.rotation import is_rotation_matrix, rot_x, rot_y, rot_z
from navcore.sensors import samples as S

# ---- alignment ----------------------------------------------------------------------------

@pytest.mark.parametrize("tilt", [(0.0, 0.0), (0.05, -0.08), (0.3, 0.2), (-1.2, 0.4)])
def test_tilt_rotation_levels_gravity(tilt):
    R_bl = rot_x(tilt[0]) @ rot_y(tilt[1])  # true mount tilt: level -> body
    up_b = R_bl @ np.array([0, 0, 1.0])
    R_lb = MT.tilt_rotation(up_b)
    np.testing.assert_allclose(R_lb @ up_b, [0, 0, 1], atol=1e-12)
    assert is_rotation_matrix(R_lb)
    np.testing.assert_allclose(MT.tilt_rotation([0, 0, -1.0]) @ [0, 0, -1.0], [0, 0, 1])


def synthetic_drive(mount_yaw, tilt=(0.04, -0.06), n_fix=300, seed=0, noise=0.05):
    """SYNTHETIC drive: 1 Hz GNSS speed/course from a smooth accel/turn profile, 10 Hz IMU
    whose specific force = the same vehicle accel seen through a known mount + gravity."""
    rng = np.random.default_rng(seed)
    t10 = np.arange(0, n_fix, 0.1)
    a_long = 1.2 * np.sin(t10 / 7.0)
    yaw_rate = 0.15 * np.sin(t10 / 11.0 + 1.0)
    v = 12.0 + np.cumsum(a_long) * 0.1
    psi = np.cumsum(yaw_rate) * 0.1
    a_lat = v * yaw_rate
    R_bv = (rot_x(tilt[0]) @ rot_y(tilt[1])) @ rot_z(mount_yaw)  # vehicle -> body (true)
    f_v = np.column_stack([a_long, a_lat, np.full_like(t10, 9.81)])
    accel_b = f_v @ R_bv.T + rng.normal(0, noise, (len(t10), 3))
    idx = np.arange(0, len(t10), 10)
    bearing = math.pi / 2 - psi[idx]  # course, clockwise from North
    return accel_b, t10, t10[idx], v[idx], bearing, R_bv


@pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -120.0, 179.0])
def test_mount_yaw_recovered_from_gnss_motion(yaw_deg):
    accel_b, t_imu, t_fix, v, brg, R_bv = synthetic_drive(math.radians(yaw_deg))
    est = MT.estimate_mounting(accel_b, t_imu, t_fix, v, brg)
    assert math.degrees(est.yaw_rad) == pytest.approx(yaw_deg, abs=1.5)
    assert est.fit.r2 > 0.9 and 0.9 < est.fit.scale < 1.1 and est.acceptable()
    # R_b^v must undo the true vehicle->body rotation
    np.testing.assert_allclose(est.R_vb @ R_bv, np.eye(3), atol=0.03)
    rot = est.rotation()
    assert (rot.src, rot.dst) == (Frame.BODY, Frame.VEHICLE)


def test_poorly_observed_alignment_is_flagged_not_trusted():
    accel_b, t_imu, t_fix, v, brg, _ = synthetic_drive(0.6, noise=0.05)
    rng = np.random.default_rng(9)
    accel_b = accel_b.copy()
    accel_b[:, :2] = rng.normal(0, 1.0, (len(accel_b), 2))  # horizontal accel unrelated to motion
    est = MT.estimate_mounting(accel_b, t_imu, t_fix, v, brg)
    assert not est.acceptable()


def test_gnss_acceleration_signs_left_turn_and_braking():
    t = np.arange(10.0)
    v = 20.0 - 1.0 * t  # braking at 1 m/s^2
    bearing = np.radians(90.0 - 5.0 * t)  # course turning anticlockwise = LEFT turn
    _, a = MT.gnss_vehicle_acceleration(t, v, bearing)
    np.testing.assert_allclose(a[:, 0], -1.0)
    assert np.all(a[:, 1] > 0)  # centripetal to the left is +y in FLU
    np.testing.assert_allclose(a[:, 1], v[1:-1] * math.radians(5.0), rtol=1e-12)


def test_low_speed_epochs_are_dropped():
    t = np.arange(10.0)
    tm, _ = MT.gnss_vehicle_acceleration(t, np.full(10, 1.0), np.zeros(10))
    assert len(tm) == 0


# ---- sensor samples ----------------------------------------------------------------------

def test_imu_sample_validation():
    s = S.ImuSample(t_s=1.0, accel_mps2=[0, 0, 9.8], gyro_radps=[0, 0, 0.1])
    assert s.frame is Frame.BODY
    with pytest.raises(S.SensorSampleError):
        S.ImuSample(t_s=1.0, accel_mps2=[0, 0, np.nan])
    with pytest.raises(S.SensorSampleError):
        S.ImuSample(t_s=1.0, accel_mps2=[0, 0, 500.0])  # beyond full scale
    with pytest.raises(S.SensorSampleError):
        S.ImuSample(t_s=1.0, accel_mps2=[0, 0, 9.8], frame=Frame.ECEF)
    with pytest.raises(S.SensorSampleError):  # an unmeasured axis must be zero, not guessed
        S.ImuSample(t_s=1.0, gyro_radps=[0.1, 0, 0], gyro_valid=(False, True, True))
    with pytest.raises(ValueError):
        s.accel_mps2[0] = 1.0  # immutable


def test_gnss_sample_validation_and_velocity():
    g = S.GnssSample(t_s=10.0, t_received_s=14.1, lat_rad=0.9, lon_rad=-0.02, h_m=100.0,
                     horizontal_accuracy_m=3.0, speed_mps=10.0, bearing_rad=math.radians(90))
    assert g.latency_s == pytest.approx(4.1)
    np.testing.assert_allclose(g.velocity_enu(), [10.0, 0.0], atol=1e-12)  # due East
    with pytest.raises(S.SensorSampleError, match="degrees"):
        S.GnssSample(t_s=0.0, lat_rad=52.4, lon_rad=-1.5, h_m=0.0, horizontal_accuracy_m=3.0)
    with pytest.raises(S.SensorSampleError):
        S.GnssSample(t_s=5.0, t_received_s=4.0, lat_rad=0.9, lon_rad=0.0, h_m=0.0, horizontal_accuracy_m=3.0)
    with pytest.raises(S.SensorSampleError):
        S.GnssSample(t_s=0.0, lat_rad=0.9, lon_rad=0.0, h_m=0.0, horizontal_accuracy_m=0.0)
    with pytest.raises(S.SensorSampleError):
        S.GnssSample(t_s=0.0, lat_rad=0.9, lon_rad=0.0, h_m=0.0, horizontal_accuracy_m=3.0, speed_mps=5.0)


def test_timestamp_guard_counts_everything():
    g = S.TimestampGuard(max_gap_s=1.0)
    out = [g.step(t) for t in (0.0, 0.1, 0.1, 0.05, 0.2, 5.0, 5.1)]
    assert out[0] is None and out[1] == pytest.approx(0.1)
    assert out[2] is None and out[3] is None  # duplicate, backwards
    assert out[4] == pytest.approx(0.1)
    assert out[5] is None and out[6] == pytest.approx(0.1)  # gap restarts integration
    assert (g.n_duplicates, g.n_backwards, g.n_gaps) == (1, 1, 1)


def test_iovnbd_axis_map_encodes_the_evidence():
    m = S.IOVNBD_AXIS_MAP
    s = m.to_body(0.0, [0.1, 0.2, 9.8], [0.5, 0.07, -0.3])
    np.testing.assert_array_equal(s.gyro_radps, [0.0, 0.0, 0.07])  # col 2 -> z only
    assert s.gyro_valid == (False, False, True)
    np.testing.assert_array_equal(s.accel_mps2, [0.1, 0.2, 9.8])
    with pytest.raises(S.SensorSampleError):
        S.ImuAxisMap(accel=np.eye(3), gyro=np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1.0]]),
                     gyro_valid=(True, True, True), evidence="bad: column reused")
