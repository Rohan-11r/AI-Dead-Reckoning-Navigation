"""GNSS recovery manager + output smoother (Phase 9): position continuity across LOST -> RECOVERING -> GOOD.

SYNTHETIC drives with known answers (simulation/synthetic/drive.py; AGENTS.md 2.2). The
dropout is long and the dead reckoning is deliberately poor (no AI, no NHC), so the filter
itself DOES jump by hundreds of metres when GNSS returns -- which makes the continuity
assertions on the reported output meaningful rather than vacuous.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from navcore.filtering.ekf import N_STATES, ErrorStateEKF, NoiseParams
from navcore.fusion.navigator import FusedNavigator, FusionConfig
from navcore.geometry.geodesy import meridian_radius, prime_vertical_radius
from navcore.geometry.quaternion import q_from_euler
from navcore.ins.mechanization import NavState
from navcore.recovery.manager import OutputSmoother, RecoveryConfig, RecoveryManager
from navcore.sensors.samples import IOVNBD_AXIS_MAP, GnssSample
from simulation.synthetic.drive import LAT, H, synthetic_drive

NOISE = NoiseParams(0.05, 0.005, 0.02, 1e4, 1e-3, 1e4)
P0 = np.diag([9, 9, 25, 0.04, 0.04, 0.04, 1e-4, 1e-4, 1e-3, 4e-4, 4e-4, 4e-4, 1e-6, 1e-6, 1e-6])
RM = float(meridian_radius(LAT)) + H
RN = (float(prime_vertical_radius(LAT)) + H) * math.cos(LAT)
DT = 0.1


def en(a, b) -> np.ndarray:
    """(lat, lon) a minus b, in local metres [E, N]."""
    return np.array([(a[1] - b[1]) * RN, (a[0] - b[0]) * RM])


class OracleSpeed:
    """SYNTHETIC stand-in for Model A: true speed + N(0, 0.5 m/s) (as in the integration test)."""

    window = 20

    def __init__(self, seed: int) -> None:
        self.speed, self.rng = 0.0, np.random.default_rng(seed)

    def predict(self, acc, grav, wz):
        return self.speed + self.rng.normal(0.0, 0.5), 0.25


def drive(recovery: RecoveryConfig | None, outlier_m: float = 0.0, seed: int = 7, ai: bool = False):
    """Run the navigator through a 100 s dropout; per-step records. ``ai=False``: pure IMU
    (big drift, big jumps); ``ai=True``: the shipped configuration (AI speed + NHC)."""
    cfg = FusionConfig(use_ai_speed=ai, use_nhc=ai, recovery=recovery)
    oracle = OracleSpeed(seed + 100) if ai else None
    nav, rows = None, []
    for st in synthetic_drive(seed=seed, dropouts=((100.0, 200.0),)):
        if oracle is not None:
            oracle.speed = st.speed
        if nav is None:
            models = {"model_speed": oracle} if ai else {}
            nav = FusedNavigator(ErrorStateEKF(st.truth, P0, NOISE), IOVNBD_AXIS_MAP, cfg, R_vb=np.eye(3),
                                 lateral_nhc_allowed=ai, **models)
        fixes = st.fixes
        if outlier_m and fixes and abs(st.t - 200.0) < 1e-6:  # first fix after the outage: multipath
            fixes = [replace(fixes[0], lat_rad=fixes[0].lat_rad + outlier_m / RM)]
        nav.step(st.t, st.acc, st.grav, st.gyro_raw, fixes)
        s, tr = nav.ekf.nominal, st.truth
        rows.append({"t": st.t, "out": nav.output()[:2], "filt": (s.lat_rad, s.lon_rad), "v": s.v_enu_mps[:2].copy(),
                     "truth": (tr.lat_rad, tr.lon_rad), "state": nav.sm.state.value})
    return nav, rows


def anomalies(rows, key):
    """Per-step displacement NOT explained by the filter's own velocity: |dp - v dt|."""
    return np.array([np.linalg.norm(en(b[key], a[key]) - b["v"] * DT) for a, b in itertools.pairwise(rows)])


def output_bound(nav) -> float:
    """Largest correction per step the smoother may make: max(rate, |offset| / blend) * dt."""
    return nav.smoother.max_glide_mps * DT


def err(row, key) -> float:
    return float(np.linalg.norm(en(row[key], row["truth"])))


@pytest.fixture(scope="module")
def clean():
    return drive(RecoveryConfig())


@pytest.mark.slow
def test_output_is_continuous_across_lost_recovering_good(clean):
    nav, rows = clean
    states = [r["state"] for r in rows]
    seq = [s for i, s in enumerate(states) if i == 0 or states[i - 1] != s]
    i = seq.index("LOST", 2)  # the dropout's LOST (the first is before the first fix)
    assert seq[i:i + 3] == ["LOST", "RECOVERING", "GOOD"]
    filt, out = anomalies(rows, "filt"), anomalies(rows, "out")
    assert filt.max() > 100.0  # the FILTER does jump at recovery (its estimate should)...
    assert out.max() <= output_bound(nav) + 0.05  # ...the OUTPUT only glides, never teleports
    assert out.max() < 0.1 * filt.max()
    t_rec = next(r["t"] for r in rows if r["t"] > 150 and r["state"] == "RECOVERING")
    after = [r for r in rows if r["t"] >= t_rec + 45.0]
    # converged: what remains is the lag behind the routine 1 Hz GNSS corrections (a few m)
    assert max(float(np.linalg.norm(en(r["out"], r["filt"]))) for r in after) < 5.0
    assert err(rows[-1], "out") < 5.0


@pytest.mark.slow
def test_continuity_holds_without_the_recovery_manager_too():
    """The output guarantee comes from the smoother alone; it holds for Phase 7 behaviour."""
    nav, rows = drive(None)
    assert anomalies(rows, "filt").max() > 100.0
    assert anomalies(rows, "out").max() <= output_bound(nav) + 0.05


@pytest.mark.slow
def test_multipath_fix_at_recovery_is_not_fused():
    """Shipped configuration (AI speed + NHC keep the velocity known): a confident 150 m
    multipath fix as the first fix after the outage is flagged and never fused.
    (Measured, and stated honestly: in this configuration the filter's own NIS gate ALSO
    rejects it -- the manager's check is a second, explicit line of defence here, not the
    only one.)"""
    t_first = 200.0
    nav, rows = drive(RecoveryConfig(), outlier_m=150.0, ai=True)
    before = next(r for r in rows if r["t"] >= t_first - 0.05)
    r1 = next(r for r in rows if r["t"] >= t_first + 0.5)
    assert nav.recovery.counts["recovery_fixes_inconsistent"] >= 1
    assert err(r1, "filt") < err(before, "filt") + 5.0  # with it: the outlier is never fused
    assert err(rows[-1], "filt") < 5.0 and err(rows[-1], "out") < 5.0


@pytest.mark.slow
def test_an_uncertain_filter_cannot_reject_what_it_cannot_distinguish():
    """Pure IMU after 100 s: the filter claims ~170 m/s velocity sigma, so a 150 m outlier IS
    consistent with what it knows. The manager must say so (accept), not pretend (reject)."""
    nav, _ = drive(RecoveryConfig(), outlier_m=150.0, ai=False)
    assert "recovery_fixes_inconsistent" not in nav.recovery.counts
    assert nav.recovery.counts["recoveries_confirmed"] == 1


# ---- RecoveryManager unit behaviour --------------------------------------------------------

def _static_ekf(p_var: float = 1.0) -> ErrorStateEKF:
    s = NavState(0.0, LAT, math.radians(-1.5), H, np.zeros(3), q_from_euler(0.0, 0, 0))
    return ErrorStateEKF(s, np.eye(N_STATES) * p_var, NOISE)


def _fix(t: float, de: float, dn: float, acc: float = 3.0) -> GnssSample:
    return GnssSample(t_s=t, lat_rad=LAT + dn / RM, lon_rad=math.radians(-1.5) + de / RN, h_m=H,
                      horizontal_accuracy_m=acc)


def test_fixes_are_held_until_two_agree_along_the_dr_path():
    ekf = _static_ekf()
    rm = RecoveryManager(RecoveryConfig())
    rm.on_lost()
    assert rm.submit(_fix(0.0, 200, 0), ekf) == ([], False)            # held
    assert rm.submit(_fix(9.0, 350, 0), ekf) == ([], False)            # jumped 150 m while DR stood still
    out, confirmed = rm.submit(_fix(18.0, 351, 1), ekf)                 # agrees with the previous one
    assert confirmed and [f.t_s for f in out] == [9.0, 18.0]
    assert rm.counts["recovery_fixes_inconsistent"] == 1
    assert rm.submit(_fix(27.0, 351, 1), ekf) == ([_fix(27.0, 351, 1)], False)  # normal flow again


def test_confirmed_fix_inflates_the_covariance_instead_of_being_rejected():
    ekf = _static_ekf(p_var=100.0)  # over-confident: 10 m sigma, truth is 200 m away
    rm = RecoveryManager(RecoveryConfig())
    rm.on_lost()
    rm.submit(_fix(0.0, 200, 0), ekf)
    out, confirmed = rm.submit(_fix(0.0, 201, 0), ekf)
    f = rm.grade(out[-1])
    P_before = ekf.P.copy()
    assert not ekf.update_gnss(f)  # as-is, the gate would reject it (-> reject/reset loop)
    ekf.P = P_before
    Pv = ekf.P[3:6, 3:6].copy()
    rm.admit(ekf, f, confirmed)
    np.testing.assert_array_equal(ekf.P[3:6, 3:6], Pv)  # velocity is never inflated
    assert ekf.P[0, 0] <= 100.0 * P_before[0, 0] + 1e-9  # capped at x100
    assert ekf.update_gnss(f) and rm.counts["recovery_cov_inflations"] == 1
    moved = en((ekf.nominal.lat_rad, ekf.nominal.lon_rad), (LAT, math.radians(-1.5)))
    assert moved[0] > 150.0  # pulled most of the way in one (graded) update
    np.linalg.cholesky(ekf.P)


def test_inflation_beyond_the_cap_is_refused_and_leaves_the_covariance_untouched():
    """1 m sigma vs a 200 m disagreement would need ~x20,000: refused, P unchanged (the
    first real-drive benchmark diverged by 1,000+ km when a failed inflation was kept)."""
    ekf = _static_ekf(p_var=1.0)
    rm = RecoveryManager(RecoveryConfig())
    rm.on_lost()
    rm.submit(_fix(0.0, 200, 0), ekf)
    out, confirmed = rm.submit(_fix(0.0, 201, 0), ekf)
    P0 = ekf.P.copy()
    rm.admit(ekf, rm.grade(out[-1]), confirmed)
    np.testing.assert_array_equal(ekf.P, P0)
    assert rm.counts["recovery_inflation_refused"] == 1


def test_start_up_lost_is_not_treated_as_a_recovery():
    nav, _ = drive(RecoveryConfig())
    assert nav.recovery.counts["outages"] == 1  # only the real dropout; not the initial LOST


def test_fail_open_when_nothing_confirms_within_the_timeout():
    ekf = _static_ekf()
    rm = RecoveryManager(RecoveryConfig(confirm_timeout_s=30.0))
    rm.on_lost()
    for k in range(4):  # every fix 200 m from the previous one while DR stands still
        assert rm.submit(_fix(9.0 * k, 200.0 * k, 0), ekf) == ([], False)
    out, confirmed = rm.submit(_fix(36.0, 800.0, 0), ekf)  # 36 s after the first: give up waiting
    assert [f.t_s for f in out] == [36.0] and not confirmed
    assert rm.counts["recovery_timeouts"] == 1 and rm.phase == "GRADUAL"


def test_consistency_tolerance_widens_with_the_filters_heading_uncertainty():
    """DR moved 135 m east; the fixes moved 135 m at 20 deg off: consistent only if the filter
    admits a heading uncertainty of that size."""
    def pair(yaw_sigma_deg):
        ekf = _static_ekf()
        ekf.P[8, 8] = math.radians(yaw_sigma_deg) ** 2
        ekf.P[3, 3] = ekf.P[4, 4] = 0.01
        rm = RecoveryManager(RecoveryConfig())
        rm.on_lost()
        rm.submit(_fix(0.0, 0.0, 0.0, acc=5.0), ekf)
        ekf.nominal = replace(ekf.nominal, t_s=9.0, lon_rad=ekf.nominal.lon_rad + 135.0 / RN)
        ekf._hist.append(ekf.nominal)
        a = math.radians(20.0)
        return rm.submit(_fix(9.0, 135.0 * math.cos(a), 135.0 * math.sin(a), acc=5.0), ekf)[1]

    assert not pair(1.0)  # a filter claiming a 1 deg heading: 46 m disagreement is inconsistent
    assert pair(20.0)     # claiming 20 deg: the same pair is consistent


def test_graded_noise_schedule_then_normal():
    rm = RecoveryManager(RecoveryConfig(r_schedule=(9.0, 4.0, 2.0)))
    rm.phase = "GRADUAL"
    acc = [rm.grade(_fix(k, 0, 0, acc=2.0)).horizontal_accuracy_m for k in range(5)]
    assert acc == pytest.approx([6.0, 4.0, 2.0 * math.sqrt(2), 2.0, 2.0])
    assert rm.phase == "NORMAL"


# ---- OutputSmoother -------------------------------------------------------------------------

def test_smoother_absorbs_a_jump_then_converges_at_a_bounded_rate():
    ekf = _static_ekf()
    sm = OutputSmoother(max_rate_mps=15.0, max_blend_s=10.0)
    before = sm.output(ekf)
    sm.mark(ekf)
    ekf._inject(np.r_[300.0, 0.0, 0.0, np.zeros(N_STATES - 3)])  # a 300 m correction east
    sm.absorb(ekf, 0.0)
    assert en(sm.output(ekf)[:2], before[:2]) == pytest.approx([0.0, 0.0], abs=1e-4)  # no jump (<0.1 mm)
    prev, steps = sm.output(ekf), []
    for _ in range(600):  # 60 s
        sm.mark(ekf)
        sm.absorb(ekf, 0.1)
        o = sm.output(ekf)
        steps.append(float(np.linalg.norm(en(o[:2], prev[:2]))))
        prev = o
    assert max(steps) <= max(15.0, 300.0 / 10.0) * 0.1 + 1e-9  # a glide, never a teleport
    assert float(np.linalg.norm(sm.offset)) < 0.5  # converged well within 60 s
    assert sum(s > 0.5 * 1.5 for s in steps) > 50  # and it WAS rate-limited for a while


def test_smoother_absorbs_a_kilometre_correction_within_bounded_time():
    ekf = _static_ekf()
    sm = OutputSmoother(max_rate_mps=15.0, max_blend_s=10.0)
    sm.mark(ekf)
    ekf._inject(np.r_[0.0, 1500.0, 0.0, np.zeros(N_STATES - 3)])
    sm.absorb(ekf, 0.0)
    t = 0.0
    while float(np.linalg.norm(sm.offset)) > 20.0:
        sm.mark(ekf)
        sm.absorb(ekf, 0.1)
        t += 0.1
    assert t <= 10.0 + 1e-6  # within max_blend_s (a fixed 15 m/s cap alone would need ~100 s)
