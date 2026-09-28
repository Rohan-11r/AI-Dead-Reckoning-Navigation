"""Golden vectors for the Phase 12 parity check of the Kotlin port of the navigation core (AGENTS.md 4).

Each function returns a JSON-able dict: the INPUTS, fully spelled out, and the Python
reference's OUTPUTS. The Kotlin tests (android-app/core/src/test/.../nav/) replay the
inputs through the port and compare. Nothing here is data: every scenario is SYNTHETIC
(simulation/synthetic/drive.py or hand-built), labelled as such (AGENTS.md 2.2).
Map-matcher networks are written as road bundles (navcore.map_matching.bundle) next to
the JSON, so the port reads exactly the bytes the reference built.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from navcore.filtering.ekf import N_STATES, ErrorStateEKF, NoiseParams
from navcore.fusion.engine import EngineConfig, NavigationEngine
from navcore.fusion.features import make_features
from navcore.fusion.navigator import FusedNavigator, FusionConfig
from navcore.geometry.geodesy import (
    LocalTangentPlane,
    ecef_to_geodetic,
    geodetic_to_ecef,
    normal_gravity,
)
from navcore.geometry.quaternion import (
    dcm_to_q,
    q_from_euler,
    q_from_rotvec,
    q_mul,
    q_normalize,
    q_rotate,
    q_slerp,
    q_to_dcm,
)
from navcore.ins.mechanization import NavState, earth_rate_enu, propagate, transport_rate_enu
from navcore.map_matching.bundle import write_bundle
from navcore.map_matching.hmm import HmmMapMatcher
from navcore.map_matching.outage import DeadReckoningMapMatcher, course_and_sigma
from navcore.map_matching.road_network import RoadNetwork
from navcore.nhc.constraints import NhcConfig, apply_nhc
from navcore.recovery.gnss_reset import reset_to_fix
from navcore.sensors.samples import IOVNBD_AXIS_MAP, GnssSample, ImuSample
from simulation.synthetic.drive import synthetic_drive

GEN = "scripts/parity/nav_golden.py"
NOISE = NoiseParams(0.05, 0.005, 0.02, 1e4, 1e-3, 1e4)
NOISE_DICT = {"accel_white_mps2_rtHz": 0.05, "gyro_white_radps_rtHz": 0.005, "accel_bias_sigma_mps2": 0.02,
              "accel_bias_tau_s": 1e4, "gyro_bias_sigma_radps": 1e-3, "gyro_bias_tau_s": 1e4,
              "unmeasured_rate_radps_rtHz": 0.02}
P0_DIAG = [9, 9, 25, 0.04, 0.04, 0.04, 1e-4, 1e-4, 1e-3, 4e-4, 4e-4, 4e-4, 1e-6, 1e-6, 1e-6]


def fl(a) -> list:
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel()]


def state(s: NavState) -> dict:
    return {"t": s.t_s, "lat": s.lat_rad, "lon": s.lon_rad, "h": s.h_m, "v": fl(s.v_enu_mps), "q": fl(s.q_nb)}


def fix_json(f: GnssSample) -> dict:
    return {"t": f.t_s, "t_rx": f.t_received_s, "lat": f.lat_rad, "lon": f.lon_rad, "h": f.h_m,
            "acc": f.horizontal_accuracy_m, "speed": f.speed_mps, "bearing": f.bearing_rad}


# ---- geometry ------------------------------------------------------------------------------

def geometry() -> dict:
    rng = np.random.default_rng(1)
    quat = []
    for k in range(12):
        p, q = q_normalize(rng.normal(size=4)), q_normalize(rng.normal(size=4))
        v = rng.normal(size=3) * 10
        rv = rng.normal(size=3) * (1e-8 if k < 3 else 1.0)  # small-angle branch too
        quat.append({"p": fl(p), "q": fl(q), "v": fl(v), "rv": fl(rv), "t": 0.37,
                     "mul": fl(q_mul(p, q)), "rotate": fl(q_rotate(q, v)), "dcm": fl(q_to_dcm(q)),
                     "dcm_to_q": fl(dcm_to_q(q_to_dcm(q))), "from_rotvec": fl(q_from_rotvec(rv)),
                     "slerp": fl(q_slerp(p, q, 0.37)),
                     "euler": fl(rng.normal(size=3))})
    for c in quat:
        y, pch, r = c["euler"]
        c["from_euler"] = fl(q_from_euler(y, pch, r))
    geo = []
    for lat_d, lon_d, h in [(52.4, -1.5, 100.0), (-33.9, 151.2, 5.0), (0.0, 0.0, 0.0), (89.0, 45.0, 3000.0),
                            (-60.0, -170.0, -50.0)]:
        lat, lon = math.radians(lat_d), math.radians(lon_d)
        e = geodetic_to_ecef(lat, lon, h)
        la, lo, hh = ecef_to_geodetic(e)
        geo.append({"lat": lat, "lon": lon, "h": h, "ecef": fl(e), "back": [float(la), float(lo), float(hh)],
                    "gravity": float(normal_gravity(lat, h)), "earth_rate": fl(earth_rate_enu(lat)),
                    "transport_rate": fl(transport_rate_enu(lat, h, [12.0, -7.0, 0.3]))})
    ltp = LocalTangentPlane(math.radians(52.4), math.radians(-1.5), 0.0)
    pts = []
    for de, dn, du in [(0, 0, 0), (1500.0, -2300.0, 0.0), (-12000.0, 8000.0, -11.0), (37.5, 12.25, 3.0)]:
        la, lo, hh = ltp.enu_to_geodetic(np.array([de, dn, du]))
        pts.append({"enu": [de, dn, du], "geo": [float(la), float(lo), float(hh)],
                    "back": fl(ltp.geodetic_to_enu(float(la), float(lo), float(hh)))})
    return {"generator": GEN, "quaternion": quat, "geodesy": geo,
            "ltp": {"lat0": ltp.lat0_rad, "lon0": ltp.lon0_rad, "h0": ltp.h0_m, "points": pts}}


# ---- INS -------------------------------------------------------------------------------------

def ins() -> dict:
    rng = np.random.default_rng(2)
    s = NavState(0.0, math.radians(52.4), math.radians(-1.5), 100.0, np.array([10.0, 5.0, 0.1]),
                 q_from_euler(0.3, 0.02, -0.01))
    steps = []
    init = state(s)
    for _ in range(300):
        f_b = np.array([0.0, 0.0, 9.81]) + rng.normal(0, 0.5, 3)
        w_b = rng.normal(0, 0.05, 3)
        dt = float(rng.choice([0.1, 0.1, 0.1, 0.09, 0.11, 0.2]))
        s = propagate(s, f_b, w_b, dt)
        steps.append({"f": fl(f_b), "w": fl(w_b), "dt": dt, "out": state(s)})
    return {"generator": GEN, "initial": init, "steps": steps}


# ---- EKF operations ---------------------------------------------------------------------------

def ekf_ops() -> dict:
    rng = np.random.default_rng(3)
    s0 = NavState(0.0, math.radians(52.4), math.radians(-1.5), 100.0, np.array([12.0, 3.0, 0.0]),
                  q_from_euler(0.25, 0.0, 0.0))
    ekf = ErrorStateEKF(s0, np.diag(P0_DIAG), NOISE)
    ops = []

    n_log = [0]

    def rec(op: dict, ok=None) -> None:
        # the NIS only when THIS op appended a finite log record (a skipped NHC appends none;
        # a reset appends NaN)
        new = len(ekf.log) > n_log[0] and math.isfinite(ekf.log[-1].nis)
        op.update(ok=ok, state=state(ekf.nominal), P=fl(ekf.P), b_a=fl(ekf.b_a), b_g=fl(ekf.b_g),
                  nis=(ekf.log[-1].nis if new else None))
        n_log[0] = len(ekf.log)
        ops.append(op)

    t = 0.0
    valid_sets = [(True, True, True), (False, False, True)]
    for k in range(40):
        valid = valid_sets[(k // 10) % 2]
        acc = fl(np.array([0.2, 0.1, 9.81]) + rng.normal(0, 0.1, 3))
        gyr = [0.0 if not v else float(rng.normal(0, 0.02)) for v in valid]
        dt = 0.1
        ekf.predict(ImuSample(t_s=t, accel_mps2=acc, gyro_radps=gyr, gyro_valid=valid), dt)
        t = round(t + dt, 10)
        rec({"op": "predict", "t": t - dt, "acc": acc, "gyro": gyr, "valid": list(valid), "dt": dt})
        if k % 5 == 4:
            s = ekf.nominal
            f = GnssSample(t_s=t, lat_rad=s.lat_rad + rng.normal(0, 3) / 6.37e6, lon_rad=s.lon_rad + 2e-7, h_m=s.h_m + 1,
                           horizontal_accuracy_m=4.0, speed_mps=12.3, bearing_rad=1.3)
            rec({"op": "gnss", "fix": fix_json(f)}, ekf.update_gnss(f))
        if k == 12:
            rec({"op": "zupt", "t": t}, ekf.update_zupt(t))
        if k == 17:
            imu = ImuSample(t_s=t, accel_mps2=[0.1, -0.05, 9.8])
            rec({"op": "level", "t": t, "acc": [0.1, -0.05, 9.8]}, ekf.update_levelling(imu))
        if k in (20, 30):
            R_vb = q_to_dcm(q_from_euler(0.1, 0.02, -0.03))
            r = apply_nhc(ekf, t, R_vb, k == 30, 0.5, NhcConfig())
            rec({"op": "nhc", "t": t, "R_vb": fl(R_vb), "lateral": k == 30, "lat_acc": 0.5}, r["applied"])
        if k == 25:
            v = ekf.nominal.v_enu_mps
            sh = math.hypot(v[0], v[1])
            H = np.zeros((1, N_STATES))
            H[0, 3], H[0, 4] = v[0] / sh, v[1] / sh
            rec({"op": "update", "kind": "ai_speed", "t": t, "nu": [11.0 - sh], "H": fl(H), "R": [0.25]},
                ekf.update("ai_speed", t, np.array([11.0 - sh]), H, np.array([[0.25]])))
        if k == 33:  # a far fix: gate rejection, then the lock-out reset
            s = ekf.nominal
            far = GnssSample(t_s=t, lat_rad=s.lat_rad + 800 / 6.37e6, lon_rad=s.lon_rad, h_m=s.h_m,
                             horizontal_accuracy_m=3.0)
            rec({"op": "gnss", "fix": fix_json(far)}, ekf.update_gnss(far))
            reset_to_fix(ekf, far)
            rec({"op": "reset", "fix": fix_json(far)}, True)
    return {"generator": GEN, "noise": NOISE_DICT, "P0_diag": P0_DIAG, "initial": state(s0), "ops": ops}


# ---- features and the ONNX model --------------------------------------------------------------

def _windows(n: int, W: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        grav = np.tile(q_rotate(q_normalize(rng.normal(size=4)), [0, 0, 9.81]), (W, 1)) + rng.normal(0, 0.02, (W, 3))
        acc = grav + rng.normal(0, 0.6, (W, 3))
        wz = rng.normal(0, 0.1, W)
        out.append((acc, grav, wz))
    return out


def features() -> dict:
    cases = []
    for acc, grav, wz in _windows(4, 50, 4):
        gyro = np.zeros((len(wz), 3))
        gyro[:, 2] = wz
        x = make_features(acc, grav, gyro, 9.80665, 0.5)
        cases.append({"acc": fl(acc), "grav": fl(grav), "wz": fl(wz), "features": fl(x)})
    return {"generator": GEN, "accel_scale": 9.80665, "gyro_scale": 0.5, "shape": [13, 50], "cases": cases}


def model_a(repo: Path) -> dict:
    from navcore.fusion.ai_models import OnnxSequenceModel
    name = "model_a_cnn1d_w50"
    m = OnnxSequenceModel(repo / "models" / "exported" / f"{name}.onnx")
    cases = []
    for acc, grav, wz in _windows(5, m.window, 5):
        gyro = np.zeros((len(wz), 3))
        gyro[:, 2] = wz
        x = make_features(acc, grav, gyro, m.accel_scale, m.gyro_scale)[None]
        mean, logvar = m.sess.run(["mean", "logvar"], {"window": x})
        pm, pv = m.predict(acc, grav, wz)
        cases.append({"acc": fl(acc), "grav": fl(grav), "wz": fl(wz), "mean": float(mean[0]),
                      "logvar": float(logvar[0]), "predict_mean": pm, "predict_var": pv})
    return {"generator": GEN, "model": name, "tolerance": "|kotlin - python| <= 1e-5 + 1e-6 |python|", "cases": cases}


# ---- the fused navigator and the streaming engine ---------------------------------------------

class StubSpeed:
    """DETERMINISTIC stand-in for Model A (the port's ONNX path is checked separately):
    speed = 12 + 0.5 mean(acc_x) + 10 mean(w_z), variance 0.36 -- depends on the window, so
    the port's windowing is exercised."""

    window = 20

    def predict(self, acc, grav, wz):
        return 12.0 + 0.5 * float(np.mean(acc[:, 0])) + 10.0 * float(np.mean(wz)), 0.36


def _drive(t_end: float, dropouts) -> list:
    return list(synthetic_drive(seed=5, t_end_s=t_end, dropouts=dropouts))


def _step_json(st) -> dict:
    return {"t": st.t, "acc": fl(st.acc), "grav": fl(st.grav), "gyro": fl(st.gyro_raw),
            "fixes": [fix_json(f) for f in st.fixes]}


def navigator() -> dict:
    steps = _drive(180.0, ((60.0, 120.0),))
    s0 = steps[0].truth
    ekf = ErrorStateEKF(s0, np.diag(P0_DIAG), NOISE)
    nav = FusedNavigator(ekf, IOVNBD_AXIS_MAP, FusionConfig(), model_speed=StubSpeed(), R_vb=np.eye(3),
                         lateral_nhc_allowed=True)
    out = []
    for k, st in enumerate(steps):
        nav.step(st.t, st.acc, st.grav, st.gyro_raw, st.fixes)
        if k % 5 == 0 or k == len(steps) - 1:
            out.append({"k": k, "state": state(nav.ekf.nominal), "P_diag": fl(np.diag(nav.ekf.P)),
                        "gnss_state": nav.sm.state.value})
    return {"generator": GEN, "scenario": "SYNTHETIC drive seed 5, 180 s, GNSS dropout 60-120 s; IO-VNBD axis map",
            "config": "FusionConfig() defaults", "noise": NOISE_DICT, "P0_diag": P0_DIAG, "initial": state(s0),
            "R_vb": "identity", "lateral_nhc_allowed": True, "stub": "12 + 0.5 mean(acc_x) + 10 mean(wz), var 0.36, W 20",
            "inputs": [_step_json(st) for st in steps], "outputs": out, "counts": dict(nav.counts)}


def engine() -> dict:
    steps = _drive(150.0, ((70.0, 110.0),))
    eng = NavigationEngine(NOISE, EngineConfig(mh_window_s=40.0), {"model_speed": StubSpeed()})
    out = []
    for k, st in enumerate(steps):
        o = eng.on_sample(st.t, st.acc, st.grav, st.gyro_raw, st.fixes)
        if o is not None and (k % 10 == 0 or k == len(steps) - 1):
            out.append({"k": k, "mode": o.mode, "lat": o.lat_rad, "lon": o.lon_rad, "sigma_h": o.sigma_h_m,
                        "gnss_state": o.gnss_state, "v": fl(o.v_enu_mps)})
    return {"generator": GEN, "scenario": "SYNTHETIC drive seed 5, 150 s, GNSS dropout 70-110 s",
            "config": "EngineConfig(mh_window_s=40) + FusionConfig() defaults, causal mount, display=filter",
            "noise": NOISE_DICT, "inputs": [_step_json(st) for st in steps], "outputs": out,
            "chosen_offset_rad": eng.chosen_offset_rad, "tilt_source": eng.tilt_source, "t_init": eng.t_init}


# ---- map matching ------------------------------------------------------------------------------

def _net(nodes_xy: dict, ways: list) -> RoadNetwork:
    ltp = LocalTangentPlane(math.radians(52.4), math.radians(-1.5), 0.0)
    ll = {}
    for n, (x, y) in nodes_xy.items():
        la, lo, _ = ltp.enu_to_geodetic(np.array([x, y, 0.0]))
        ll[n] = (math.degrees(float(la)), math.degrees(float(lo)))
    return RoadNetwork.from_ways(ll, ways, origin_deg=(52.4, -1.5), cell_m=50.0)


def map_matching(out_dir: Path) -> dict:
    nets = {
        "crossroads": _net({1: (-600, 0), 0: (0, 0), 2: (600, 0), 3: (0, -600), 4: (0, 600)},
                           [(1, [1, 0, 2], {"highway": "primary", "name": "Through Road"}),
                            (2, [3, 0, 4], {"highway": "residential", "name": "Cross Street"})]),
        "dual": _net({1: (-1000, 0), 2: (1000, 0), 3: (1000, 20), 4: (-1000, 20)},
                     [(1, [1, 2], {"highway": "trunk", "oneway": "yes"}),
                      (2, [3, 4], {"highway": "trunk", "oneway": "yes"})]),
    }
    rng = np.random.default_rng(6)
    scen = []
    turn = [(x, 0.0, 0.0) for x in np.arange(-150.0, 0.0, 15.0)] + [(0.0, y, math.pi / 2) for y in np.arange(0.0, 151.0, 15.0)]
    scen.append(("crossroads", "drift_through", [(x, 12.0, 0.0) for x in np.arange(-150.0, 151.0, 15.0)], 15.0, 15.0))
    scen.append(("crossroads", "turn", [(x + rng.normal(0, 6), y + rng.normal(0, 6), h) for x, y, h in turn], 15.0, 15.0))
    scen.append(("crossroads", "off_network", [(x, 3.0, None) for x in np.arange(-500.0, -380.0, 15.0)]
                 + [(-380.0 + 10 * k, 250.0 + 10 * k, None) for k in range(8)]
                 + [(x, 3.0, None) for x in np.arange(-300.0, -180.0, 15.0)], 5.0, 0.0))
    scen.append(("dual", "westbound", [(x, 7.0, math.pi) for x in np.arange(300.0, -1.0, -15.0)], 10.0, 15.0))
    for name, net in nets.items():
        write_bundle(net, out_dir / f"mm_{name}.roads.bin")
    cases = []
    for net_name, label, obs, sigma, speed in scen:
        net = nets[net_name]
        m = HmmMapMatcher(net)
        res = []
        for k, (x, y, h) in enumerate(obs):
            r = m.step(float(k), float(x), float(y), np.eye(2) * sigma ** 2, h, speed if h is not None else 0.0)
            res.append({"matched": r.matched, "mode": r.mode, "segment": r.segment, "x": r.x_m, "y": r.y_m,
                        "lat": r.lat_rad, "lon": r.lon_rad, "confidence": r.map_match_confidence,
                        "radius": r.search_radius_m, "n_candidates": r.n_candidates})
        path = [(t, c.seg, c.frac) for t, c in m.decode()]
        cases.append({"network": f"mm_{net_name}.roads.bin", "label": label, "sigma": sigma, "speed": speed,
                      "obs": [[float(x), float(y), h] for x, y, h in obs], "results": res,
                      "viterbi": [{"t": t, "segment": s, "frac": f} for t, s, f in path], "counts": m.counts})
    # the dead-reckoning wrapper: lat/lon + covariance + velocity in, 1 Hz out
    net = nets["crossroads"]
    dr = DeadReckoningMapMatcher(net)
    feed = []
    for k in range(120):  # 12 s at 10 Hz eastbound along y = 9 m
        t = k * 0.1
        la, lo, _ = net.ltp.enu_to_geodetic(np.array([-100.0 + 14.0 * t, 9.0, 0.0]))
        P = [[64.0, 5.0], [5.0, 49.0]]
        v, Pv = [14.0, 0.2], [[0.25, 0.0], [0.0, 0.3]]
        r = dr.update(t, float(la), float(lo), np.array(P), np.array(v), np.array(Pv))
        feed.append({"t": t, "lat": float(la), "lon": float(lo), "P": fl(P), "v": v, "Pv": fl(Pv),
                     "out": None if r is None else {"matched": r.matched, "segment": r.segment, "lat": r.lat_rad,
                                                    "lon": r.lon_rad, "confidence": r.map_match_confidence}})
    course = [{"v": [8.0, 6.0], "Pv": [0.3, 0.0, 0.0, 0.5], "out": list(course_and_sigma(np.array([8.0, 6.0]),
                                                                                       np.diag([0.3, 0.5])))},
              {"v": [0.0, 0.0], "Pv": [1.0, 0.0, 0.0, 1.0], "out": list(course_and_sigma(np.zeros(2), np.eye(2)))}]
    return {"generator": GEN, "matcher_config": "MatcherConfig() defaults", "cases": cases,
            "dr_matcher": {"network": "mm_crossroads.roads.bin", "feed": feed}, "course": course}


def all_vectors(repo: Path, golden_dir: Path) -> dict[str, dict]:
    return {"nav_geometry.json": geometry(), "nav_ins.json": ins(), "nav_ekf_ops.json": ekf_ops(),
            "nav_features.json": features(), "nav_model_a.json": model_a(repo),
            "nav_navigator.json": navigator(), "nav_engine.json": engine(),
            "nav_map_matching.json": map_matching(golden_dir)}
