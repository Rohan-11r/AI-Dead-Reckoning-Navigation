#!/usr/bin/env python3
"""Strict PyTorch <-> ONNX numerical equivalence check for the exported models.

    .venv/Scripts/python.exe scripts/validate/validate_models.py [--names ...] [--atol 1e-5] [--rtol 1e-6]

For every exported model (default: reports/phase6/selection.json):
1. the .onnx file's SHA-256 must equal the one recorded in its model card
2. identical float32 inputs go through the PyTorch checkpoint (CPU, eval) and the ONNX
   graph (onnxruntime, CPUExecutionProvider):
     * REAL validation windows (the exact feature pipeline, no augmentation)
     * random windows at batch sizes 1, 7 and 256 (dynamic batch axis)
     * a stress set at 10x amplitude (saturating activations, softplus tail)
3. EVERY element of BOTH outputs (mean, logvar) must satisfy
       |torch - onnx| <= atol + rtol * |torch|      (atol 1e-5, rtol 1e-6)
   i.e. 1e-5 absolute for small outputs; for large ones (e.g. 280 m/s on 10x stress input)
   the bound grows with |output|, because 1e-5 is then finer than float32 can represent
   (float32 PyTorch itself differs from float64 by 3.7e-5 there). The raw worst absolute
   gap is still reported. Gate agreed with the project owner, Phase 6.
   EXCEPT the 10x stress set, whose out-of-domain inputs drive hidden activations to ~300
   (speed ~280 m/s, logvar ~33): there ONNX must be as accurate as float32 PyTorch against an
   exact float64 PyTorch reference, per output:
       max|onnx - fp64| <= 2 * max|torch32 - fp64| + atol      (and all outputs finite)
   Also agreed with the project owner.

AGENTS.md 4: "exported models are verified, not assumed" -- any failure exits non-zero and
is a release blocker. Writes reports/phase6/onnx_validation.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from scripts.export.export_models import EXPORT_DIR, load_checkpoint  # noqa: E402
from training.configs.loader import load_training_config  # noqa: E402
from training.datasets.windows import IovnbdWindows  # noqa: E402

OUT = REPO_ROOT / "reports" / "phase6" / "onnx_validation.json"


def real_windows(card: dict, n: int = 512) -> np.ndarray:
    cfg = load_training_config()
    W = card["input"]["shape"][2]
    cfg["data"]["window_samples"] = W
    cfg["data"]["stride_eval"] = W
    key = card["name"].split("_")[0] + "_" + card["name"].split("_")[1]
    verified = cfg["data"]["verified_only"][key]
    ds = IovnbdWindows("val", card["target"], cfg, augment=False, verified_only=verified)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    return np.stack([ds[int(i)][0].numpy() for i in idx]).astype(np.float32)


@torch.no_grad()
def compare(model, sess, x: np.ndarray, atol: float, rtol: float) -> dict:
    tm, tl = model(torch.from_numpy(x))
    om, ol = sess.run(["mean", "logvar"], {"window": x})
    dm = np.abs(tm.numpy() - om)
    dl = np.abs(tl.numpy() - ol)
    # fraction of the allowed bound used, per element; the gate is gate_usage <= 1
    use = max(float((dm / (atol + rtol * np.abs(tm.numpy()))).max()),
              float((dl / (atol + rtol * np.abs(tl.numpy()))).max()))
    return {"batch": int(x.shape[0]), "max_abs_diff_mean": float(dm.max()), "gate_usage": use,
            "max_abs_diff_logvar": float(dl.max()),
            "max_rel_diff_mean": float((dm / np.maximum(np.abs(tm.numpy()), 1e-6)).max()),
            "output_range_mean": [float(tm.min()), float(tm.max())],
            "finite": bool(np.isfinite(om).all() and np.isfinite(ol).all())}


@torch.no_grad()
def compare_fp64(model, sess, x: np.ndarray, atol: float) -> dict:
    """ONNX (fp32) vs an exact fp64 PyTorch reference, relative to fp32 PyTorch's own error."""
    tm, tl = (t.numpy() for t in model(torch.from_numpy(x)))
    em, el = (t.numpy() for t in copy.deepcopy(model).double()(torch.from_numpy(x).double()))
    om, ol = sess.run(["mean", "logvar"], {"window": x})
    out = {"batch": int(x.shape[0]), "rule": "max|onnx-fp64| <= 2*max|torch32-fp64| + atol",
           "max_abs_diff_mean": float(np.abs(tm - om).max()), "max_abs_diff_logvar": float(np.abs(tl - ol).max()),
           "output_range_mean": [float(tm.min()), float(tm.max())],
           "output_range_logvar": [float(tl.min()), float(tl.max())],
           "finite": bool(np.isfinite(om).all() and np.isfinite(ol).all())}
    use = 0.0
    for k, t, o, e in (("mean", tm, om, em), ("logvar", tl, ol, el)):
        err_t, err_o = float(np.abs(t - e).max()), float(np.abs(o - e).max())
        out[f"torch32_vs_fp64_{k}"], out[f"onnx_vs_fp64_{k}"] = err_t, err_o
        use = max(use, err_o / (2 * err_t + atol))
    out["gate_usage"] = use
    return out


def validate(name: str, atol: float, rtol: float) -> dict:
    card = json.loads((EXPORT_DIR / f"{name}.model_card.json").read_text(encoding="utf-8"))
    onnx_path = REPO_ROOT / card["files"]["onnx"]
    sha_ok = hashlib.sha256(onnx_path.read_bytes()).hexdigest() == card["files"]["onnx_sha256"]
    model, _, _ = load_checkpoint(name)
    torch.set_num_threads(1)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    C, W = card["input"]["shape"][1:]
    rng = np.random.default_rng(26168)
    cases = {"real_val_windows": real_windows(card)}
    for b in (1, 7, 256):
        cases[f"random_batch_{b}"] = rng.normal(0, 1, (b, C, W)).astype(np.float32)
    stress = (10 * rng.normal(0, 1, (64, C, W))).astype(np.float32)
    results = {k: compare(model, sess, x, atol, rtol) for k, x in cases.items()}
    results["stress_10x"] = compare_fp64(model, sess, stress, atol)
    worst = max(max(r["max_abs_diff_mean"], r["max_abs_diff_logvar"]) for r in results.values())
    usage = max(r["gate_usage"] for r in results.values())
    passed = sha_ok and usage <= 1.0 and all(r["finite"] for r in results.values())
    return {"name": name, "onnx": card["files"]["onnx"], "sha256_matches_card": sha_ok,
            "tolerance": {"atol": atol, "rtol": rtol, "rule": "|torch-onnx| <= atol + rtol*|torch|"},
            "worst_abs_diff": worst, "worst_gate_usage": usage, "passed": passed,
            "onnxruntime": ort.__version__, "cases": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None)
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()
    if args.names:
        names = args.names.split(",")
    else:
        sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
        names = [v["selected"] for v in sel.values()]
    res = [validate(n, args.atol, args.rtol) for n in names]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"generated": datetime.now().isoformat(timespec="seconds"),
                               "script": "scripts/validate/validate_models.py", "models": res},
                              indent=1) + "\n", encoding="utf-8")
    for r in res:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['name']}: worst |torch - onnx| = {r['worst_abs_diff']:.3e} "
              f"| gate usage {r['worst_gate_usage']:.2f} of 1 (atol 1e-5 + rtol 1e-6*|y|); sha256 matches card: {r['sha256_matches_card']}")
        for k, c in r["cases"].items():
            print(f"      {k:18} batch {c['batch']:4d}  mean {c['max_abs_diff_mean']:.2e}  logvar "
                  f"{c['max_abs_diff_logvar']:.2e}  range [{c['output_range_mean'][0]:.1f}, "
                  f"{c['output_range_mean'][1]:.1f}]  usage {c['gate_usage']:.2f}  finite {c['finite']}")
    return 0 if all(r["passed"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
