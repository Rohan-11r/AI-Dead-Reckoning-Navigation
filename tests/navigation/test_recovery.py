"""GNSS lock-out recovery. SYNTHETIC scenario with a known answer."""

from __future__ import annotations

import math

import numpy as np

from navcore.filtering.ekf import A_, N_STATES, ErrorStateEKF, NoiseParams
from navcore.geometry.geodesy import meridian_radius
from navcore.ins.mechanization import NavState
from navcore.recovery.gnss_reset import MAX_ATTITUDE_SIGMA_RAD, ConsecutiveRejectionReset
from navcore.sensors.samples import GnssSample

LAT, LON, H = math.radians(52.4), math.radians(-1.5), 100.0
NOISE = NoiseParams(0.05, 0.005, 0.02, 100.0, 1e-3, 100.0)


def test_lockout_is_broken_after_n_consecutive_rejections():
    ekf = ErrorStateEKF(NavState(0.0, LAT, LON, H, np.zeros(3)), np.eye(N_STATES) * 1e-2, NOISE)
    pol = ConsecutiveRejectionReset(n_consecutive=3)
    rm = float(meridian_radius(LAT)) + H
    far = GnssSample(t_s=0.0, lat_rad=LAT + 500.0 / rm, lon_rad=LON, h_m=H, horizontal_accuracy_m=3.0)
    results = []
    for _ in range(3):
        results.append(pol.after_update(ekf, far, ekf.update_gnss(far)))
    assert results == [False, False, True] and pol.n_resets == 1
    north = (ekf.nominal.lat_rad - LAT) * rm
    assert abs(north - 500.0) < 1e-6  # re-initialised on the fix
    assert math.sqrt(ekf.P[0, 0]) == 3.0
    assert ekf.update_gnss(far)  # and now accepts the stream again
    assert ekf.log[-2].kind == "reset"


def test_repeated_resets_cap_attitude_uncertainty():
    ekf = ErrorStateEKF(NavState(0.0, LAT, LON, H, np.zeros(3)), np.eye(N_STATES) * 0.1, NOISE)
    pol = ConsecutiveRejectionReset(n_consecutive=1)
    rm = float(meridian_radius(LAT)) + H
    for k in range(20):
        far = GnssSample(t_s=0.0, lat_rad=LAT + (1000.0 * (k + 1)) / rm, lon_rad=LON, h_m=H,
                         horizontal_accuracy_m=3.0)
        pol.after_update(ekf, far, ekf.update_gnss(far))
    assert np.all(np.sqrt(np.diag(ekf.P[A_, A_])) <= MAX_ATTITUDE_SIGMA_RAD + 1e-12)
    np.linalg.cholesky(ekf.P)
