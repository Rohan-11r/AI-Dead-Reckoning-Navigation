"""Unit tests: NHC, AI measurement helpers, GNSS state machine, ONNX model integrity.

All inputs are SYNTHETIC with known answers (AGENTS.md 2.2).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import torch

from navcore.filtering.ekf import A_, N_STATES, ErrorStateEKF, NoiseParams
from navcore.fusion.ai_models import (
    ImuWindow,
    ModelIntegrityError,
    OnnxSequenceModel,
    correct_specific_force,
    speed_innovation,
)
from navcore.fusion.features import FEATURE_NAMES
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.geometry.quaternion import q_from_euler, q_from_rotvec, q_mul, q_normalize, q_rotate
from navcore.geometry.rotation import rot_x, rot_y, rot_z
from navcore.ins.mechanization import NavState, gravity_enu
from navcore.nhc.constraints import nhc_innovation
from navcore.state.gnss_state import GnssState, GnssStateConfig, GnssStateMachine

LAT, LON, H = math.radians(52.4), math.radians(-1.5), 100.0
NOISE = NoiseParams(0.05, 0.005, 0.02, 100.0, 1e-3, 100.0)


def inject(s: NavState, dx9) -> NavState:
    rm = float(meridian_radius(s.lat_rad)) + s.h_m
    rn = float(prime_vertical_radius(s.lat_rad)) + s.h_m
    return NavState(s.t_s, s.lat_rad + dx9[1] / rm, s.lon_rad + dx9[0] / (rn * math.cos(s.lat_rad)),
                    s.h_m + dx9[2], s.v_enu_mps + dx9[3:6], q_normalize(q_mul(q_from_rotvec(dx9[6:9]), s.q_nb)))


def moving_ekf(yaw=0.6):
    s = NavState(0.0, LAT, LON, H, np.array([12.0, 8.0, 0.2]), q_from_euler(yaw, 0.03, -0.02))
    return ErrorStateEKF(s, np.eye(N_STATES), NOISE)


# ---- NHC -----------------------------------------------------------------------------------

def nhc_residual(ekf, R_vb, rows, dx, scale):
    """|z_true - (h(x_hat) + H dx)| for truth = nominal (+) scale*dx, and |H dx|."""
    from navcore.geometry.quaternion import q_to_dcm
    nu, Hm = nhc_innovation(ekf, R_vb, rows)
    tr = inject(ekf.nominal, scale * dx[:9])
    z_true = (R_vb @ q_to_dcm(tr.q_nb).T @ tr.v_enu_mps)[list(rows)]
    # nu = 0 - h(x_hat): so h(x_hat) = -nu and the linear prediction is -nu + H dx
    return np.linalg.norm(z_true - (-nu + Hm @ (scale * dx))), np.linalg.norm(Hm @ (scale * dx))


@pytest.mark.parametrize("rows", [(1,), (2,), (1, 2)])
def test_nhc_H_is_the_exact_jacobian(rows):
    ekf = moving_ekf()
    R_vb = rot_z(0.4) @ rot_x(0.05) @ rot_y(-0.03)
    rng = np.random.default_rng(0)
    for _ in range(10):
        dx = np.zeros(N_STATES)
        dx[3:6] = rng.normal(0, 0.3, 3)
        dx[A_] = rng.normal(0, 2e-3, 3)
        r1, lin = nhc_residual(ekf, R_vb, rows, dx, 1.0)
        r01, _ = nhc_residual(ekf, R_vb, rows, dx, 0.1)
        assert r1 < 0.05 * lin and r01 / r1 < 0.02  # quadratic remainder: exact Jacobian


def test_vertical_nhc_row_does_not_depend_on_mount_yaw():
    ekf = moving_ekf()
    tilt = rot_x(0.05) @ rot_y(-0.03)
    rows = [nhc_innovation(ekf, rot_z(y) @ tilt, (2,)) for y in (0.0, 1.0, -2.5)]
    for nu, Hm in rows[1:]:
        np.testing.assert_allclose(nu, rows[0][0], atol=1e-12)
        np.testing.assert_allclose(Hm, rows[0][1], atol=1e-12)


def test_nhc_zero_for_a_vehicle_moving_along_its_axis():
    q = q_from_euler(0.7, 0.0, 0.0)  # body x = vehicle x, heading 0.7 rad
    v = q_rotate(q, [15.0, 0.0, 0.0])
    ekf = ErrorStateEKF(NavState(0.0, LAT, LON, H, v, q), np.eye(N_STATES), NOISE)
    nu, _ = nhc_innovation(ekf, np.eye(3), (1, 2))
    np.testing.assert_allclose(nu, 0.0, atol=1e-12)


# ---- AI speed / Model B helpers -------------------------------------------------------------

def test_speed_innovation_and_jacobian():
    ekf = moving_ekf()
    nu, Hm = speed_innovation(ekf, 15.0)
    sh = math.hypot(12.0, 8.0)
    assert nu[0] == pytest.approx(15.0 - sh)
    eps = 1e-6
    for k in (3, 4, 5):
        v = ekf.nominal.v_enu_mps.copy()
        v[k - 3] += eps
        num = (math.hypot(v[0], v[1]) - sh) / eps
        assert Hm[0, k] == pytest.approx(num, abs=1e-6)
    slow = ErrorStateEKF(NavState(0.0, LAT, LON, H, np.array([0.2, 0.1, 0.0])), np.eye(N_STATES), NOISE)
    assert speed_innovation(slow, 0.5) is None  # singular direction: skipped


def test_model_b_correction_changes_only_the_along_track_component():
    q = q_from_euler(0.3, 0.02, -0.01)
    v = np.array([10.0, 5.0, 0.0])
    g_n = gravity_enu(LAT, H)
    f_b = q_rotate(q.copy() * np.array([1, -1, -1, -1]), np.array([0.4, 0.9, 0.0]) - g_n)  # R^T(a - g)
    f_corr, ok = correct_specific_force(f_b, q, v, g_n, a_long_ai=-1.5, var_ai=1e-9, var_imu=1.0)
    assert ok
    u = np.array([v[0], v[1], 0.0]) / math.hypot(v[0], v[1])
    lin_before = q_rotate(q, f_b) + g_n
    lin_after = q_rotate(q, f_corr) + g_n
    assert float(u @ lin_after) == pytest.approx(-1.5, abs=1e-6)  # trusted AI dominates
    perp = lin_before - (u @ lin_before) * u
    np.testing.assert_allclose(lin_after - (u @ lin_after) * u, perp, atol=1e-9)  # untouched
    same, ok2 = correct_specific_force(f_b, q, np.array([0.5, 0, 0]), g_n, -1.5, 1e-9, 1.0)
    assert not ok2 and np.allclose(same, f_b)  # too slow: direction undefined


def test_imu_window_restarts_on_a_gap():
    w = ImuWindow(size=5, max_dt_s=0.15)
    for k in range(5):
        w.push(k * 0.1, [0, 0, 9.8], [0, 0, 9.8], 0.0)
    assert w.ready()
    w.push(5.0, [0, 0, 9.8], [0, 0, 9.8], 0.0)  # 4.6 s gap
    assert not w.ready()


# ---- GNSS state machine ----------------------------------------------------------------------

def test_state_machine_full_cycle():
    sm = GnssStateMachine(GnssStateConfig(degraded_accuracy_m=10, degraded_age_s=12, lost_age_s=25,
                                          good_streak=2, recover_streak=3))
    assert sm.state is GnssState.LOST  # nothing trusted before the first fix
    assert sm.on_fix(0.0, 3.0, True) is GnssState.RECOVERING
    sm.on_fix(1.0, 3.0, True)
    assert sm.on_fix(2.0, 3.0, True) is GnssState.GOOD
    assert sm.on_fix(3.0, 30.0, True) is GnssState.DEGRADED  # inaccurate fix
    sm.on_fix(4.0, 3.0, True)
    assert sm.on_fix(5.0, 3.0, True) is GnssState.GOOD
    assert sm.on_fix(6.0, 3.0, False) is GnssState.DEGRADED  # gate rejection
    sm.on_fix(7.0, 3.0, True)
    sm.on_fix(8.0, 3.0, True)
    assert sm.state is GnssState.GOOD
    assert sm.on_time(21.0) is GnssState.DEGRADED  # 13 s without a fix
    assert sm.on_time(34.0) is GnssState.LOST  # 26 s
    assert sm.on_fix(35.0, 3.0, True) is GnssState.RECOVERING
    assert sm.on_fix(36.0, 3.0, False) is GnssState.DEGRADED  # bad fix during recovery
    assert [t[2] for t in sm.transitions][:3] == ["RECOVERING", "GOOD", "DEGRADED"]
    assert sm.policy.use_ai_speed and sm.policy.use_gnss


def test_policies_use_ai_speed_only_when_gnss_is_not_good():
    from navcore.state.gnss_state import POLICY
    assert not POLICY[GnssState.GOOD].use_ai_speed
    assert POLICY[GnssState.LOST].use_ai_speed and not POLICY[GnssState.LOST].use_gnss
    assert POLICY[GnssState.DEGRADED].gnss_noise_inflation > 1.0


# ---- ONNX runtime integrity ------------------------------------------------------------------

def test_onnx_model_refuses_a_file_that_differs_from_its_card(tmp_path):
    from training.models.registry import build

    model = build("cnn1d", in_channels=13, positive=True, hidden=8, layers=1, kernel=3).eval()
    path = tmp_path / "m.onnx"
    torch.onnx.export(model, (torch.randn(1, 13, 20),), str(path), opset_version=17, dynamo=False,
                      input_names=["window"], output_names=["mean", "logvar"],
                      dynamic_axes={"window": {0: "batch"}, "mean": {0: "batch"}, "logvar": {0: "batch"}})
    card = {"files": {"onnx_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "input": {"features": FEATURE_NAMES, "shape": ["batch", 13, 20],
                      "scaling": {"accel_scale_mps2": 9.80665, "gyro_scale_radps": 0.5}},
            "outputs": {"mean": {"quantity": "forward_speed", "unit": "m/s"}}}
    card_path = tmp_path / "m.model_card.json"
    card_path.write_text(json.dumps(card))
    m = OnnxSequenceModel(path, card_path)
    mean, var = m.predict(np.tile([0, 0, 9.8], (20, 1)), np.tile([0, 0, 9.8], (20, 1)), np.zeros(20))
    x = torch.from_numpy(__import__("navcore.fusion.features", fromlist=["x"]).make_features(
        np.tile([0, 0, 9.8], (20, 1)), np.tile([0, 0, 9.8], (20, 1)), np.zeros((20, 3)), 9.80665, 0.5)[None])
    with torch.no_grad():
        tm, _ = model(x)
    assert mean == pytest.approx(float(tm[0]), abs=1e-5) and var > 0
    path.write_bytes(path.read_bytes() + b"\0")  # tamper
    with pytest.raises(ModelIntegrityError):
        OnnxSequenceModel(path, card_path)


# ---- AI speed update rate (overlapping windows are NOT independent evidence) ----------------

class _ConstSpeed:
    """SYNTHETIC stub model: fixed speed and variance; counts its calls."""

    window = 20

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, acc, grav, wz):
        self.calls += 1
        return 15.0, 4.0


def test_ai_speed_is_applied_once_per_interval_with_inflated_variance():
    from navcore.fusion.navigator import FusedNavigator, FusionConfig
    from navcore.sensors.samples import IOVNBD_AXIS_MAP

    ekf = moving_ekf()
    model = _ConstSpeed()
    nav = FusedNavigator(ekf, IOVNBD_AXIS_MAP, FusionConfig(use_nhc=False, use_ai_accel=False),
                         model_speed=model)
    assert nav.sm.state is GnssState.LOST  # no fix yet -> the policy asks for AI speed
    Rs = []
    real_update = ekf.update

    def spy(kind, t_s, nu, Hm, Rm, gate=True):
        if kind == "ai_speed":
            Rs.append(float(Rm[0, 0]))
        return real_update(kind, t_s, nu, Hm, Rm, gate)

    ekf.update = spy
    g = -gravity_enu(LAT, H)
    for k in range(100):  # 10 s at 10 Hz
        nav.step(k * 0.1, [0.0, 0.0, float(g[2])], [0.0, 0.0, float(g[2])], np.zeros(3))
    assert model.calls == 100 - model.window + 1  # the model itself runs on every sample
    n = nav.counts["ai_speed_accepted"] + nav.counts["ai_speed_rejected"]
    assert 7 <= n <= 9, n  # ~1 Hz once the 2 s window is full, not 10 Hz
    # 2 s window / 1 s interval -> variance x2 (the stub's 4.0 is above the 0.5^2 floor)
    assert Rs and all(r == pytest.approx(8.0) for r in Rs)


def test_default_fusion_config_is_the_measured_one():
    """Phase 7 real-data ablation decided these defaults; changing them needs new evidence."""
    from navcore.fusion.navigator import FusionConfig

    cfg = FusionConfig()
    assert cfg.use_ai_speed and not cfg.use_ai_accel  # Model B off
    assert cfg.use_nhc and cfg.nhc_requires_ai_speed


def test_nhc_is_inert_without_ai_speed_unless_explicitly_allowed():
    from navcore.fusion.navigator import FusedNavigator, FusionConfig
    from navcore.sensors.samples import IOVNBD_AXIS_MAP

    g = float(-gravity_enu(LAT, H)[2])
    counts = {}
    for allow in (False, True):
        cfg = FusionConfig(use_ai_speed=False, nhc_requires_ai_speed=not allow)
        nav = FusedNavigator(moving_ekf(), IOVNBD_AXIS_MAP, cfg)
        for k in range(20):
            nav.step(k * 0.1, [0.0, 0.0, g], [0.0, 0.0, g], np.zeros(3))
        counts[allow] = sum(v for k, v in nav.counts.items() if k.startswith("nhc_"))
    assert counts[False] == 0 and counts[True] > 0
