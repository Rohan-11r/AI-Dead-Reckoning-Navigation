#!/usr/bin/env python3
"""Phase 6 architecture x window sweep for Model A (speed) and Model B (acceleration).

    .venv/Scripts/python.exe scripts/train/sweep.py [--models a,b]

Grid, epochs and selection rule come from configs/training.yaml `sweep`. Every candidate
is trained on TRAIN (augmented) and scored on the FULL VAL split (non-overlapping windows).
Selection per model: lowest val MAE; every candidate within `tie_tolerance` of it competes
on single-window CPU latency (the on-device cost); latencies within `latency_tolerance` of
the fastest are a tie (measurement noise), broken on parameter count, then val MAE. The TEST split
is NOT touched here -- it is evaluated once, after selection (scripts/evaluate).

Writes reports/phase6/sweep.json (all candidates) and reports/phase6/selection.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from training.configs.loader import load_training_config  # noqa: E402
from training.trainers.run import train_one  # noqa: E402
from training.trainers.trainer import seed_everything  # noqa: E402

OUT = REPO_ROOT / "reports" / "phase6"


@torch.no_grad()
def cpu_latency_ms(model: torch.nn.Module, shape: tuple[int, int], n: int = 200) -> float:
    """Median single-window (batch 1) CPU inference time, 1 thread -- a phone-side proxy."""
    m = model.to("cpu").eval()
    x = torch.randn(1, *shape)
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(20):
            m(x)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.set_num_threads(prev)
    return float(np.median(ts))


def select(cands: list[dict], tol: float, lat_tol: float = 0.0) -> dict:
    """Lowest val MAE; candidates within `tol` of it compete on CPU latency. Latencies within
    `lat_tol` of the fastest are a tie (below measurement noise), broken on params, then MAE."""
    ok = [c for c in cands if c.get("val_mae") is not None]
    best = min(c["val_mae"] for c in ok)
    close = [c for c in ok if c["val_mae"] <= best * (1 + tol)]
    fastest = min(c["cpu_latency_ms"] for c in close)
    fast = [c for c in close if c["cpu_latency_ms"] <= fastest * (1 + lat_tol)]
    chosen = min(fast, key=lambda c: (c["n_params"], c["val_mae"]))
    return {"selected": chosen["name"], "best_val_mae": best, "tie_tolerance": tol,
            "latency_tolerance": lat_tol, "within_tolerance": [c["name"] for c in close],
            "latency_tied": [c["name"] for c in fast],
            "reason": (f"lowest val MAE {best:.4f}; {len(close)} candidate(s) within {tol:.0%}; "
                       f"{len(fast)} within {lat_tol:.0%} of the fastest CPU latency "
                       f"({fastest:.3f} ms/window); chose {chosen['name']} "
                       f"({chosen['cpu_latency_ms']:.3f} ms, {chosen['n_params']:,} params, "
                       f"val MAE {chosen['val_mae']:.4f})")}


def interleaved_latency_ms(models: dict, rounds: int = 7, n: int = 300) -> dict:
    """Round-robin the candidates so background load hits them all alike; keep each model's
    best round median (the least-disturbed estimate of its intrinsic cost)."""
    out = {k: [] for k in models}
    for _ in range(rounds):
        for k, (m, shape) in models.items():
            out[k].append(cpu_latency_ms(m, shape, n=n))
    return {k: {"ms": float(min(v)), "rounds_ms": [round(x, 4) for x in v]} for k, v in out.items()}


def reselect() -> int:
    """Latency measured during the sweep can be inflated by other CPU load; re-measure it for
    every candidate from its checkpoint, then re-apply the unchanged selection rule."""
    from scripts.export.export_models import load_checkpoint

    cfg = load_training_config()
    sel = cfg["sweep"]["selection"]
    report = json.loads((OUT / "sweep.json").read_text(encoding="utf-8"))
    for key, cands in report["candidates"].items():
        models = {}
        for c in cands:
            model, rec, _ = load_checkpoint(c["name"])
            models[c["name"]] = (model, tuple(rec["input_shape"]))
        lat = interleaved_latency_ms(models)
        for c in cands:
            c.setdefault("cpu_latency_ms_sweep", c["cpu_latency_ms"])
            c["cpu_latency_ms"] = lat[c["name"]]["ms"]
            c["cpu_latency_rounds_ms"] = lat[c["name"]]["rounds_ms"]
            print(f"  {c['name']:22} val MAE {c['val_mae']:.4f}  latency {c['cpu_latency_ms']:.3f} ms "
                  f"(rounds {min(c['cpu_latency_rounds_ms']):.3f}-{max(c['cpu_latency_rounds_ms']):.3f}; "
                  f"sweep-time {c['cpu_latency_ms_sweep']:.3f})")
        report["selection"][key] = select(cands, sel["tie_tolerance"], sel.get("latency_tolerance", 0.0))
        print(f"[{key}] SELECTED {report['selection'][key]['selected']}: {report['selection'][key]['reason']}")
    report["latency_remeasured"] = datetime.now().isoformat(timespec="seconds")
    report["latency_method"] = "interleaved round-robin, 7 rounds x 300 batch-1 calls, 1 thread; min of round medians"
    (OUT / "sweep.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (OUT / "selection.json").write_text(json.dumps(report["selection"], indent=1) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="a,b")
    ap.add_argument("--reselect", action="store_true",
                    help="no training: reload every candidate checkpoint, re-measure CPU latency in a "
                         "quiet environment (median of 500) and re-apply the selection rule")
    args = ap.parse_args()
    if args.reselect:
        return reselect()
    cfg = load_training_config()
    sw = cfg["sweep"]
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "python": platform.python_version(),
              "torch": torch.__version__, "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "grid": {"archs": sw["archs"], "windows": sw["windows"]}, "epochs": sw["epochs"],
              "batches_per_epoch": sw["batches_per_epoch"], "candidates": {}, "selection": {}}
    for key in [f"model_{m.strip()}" for m in args.models.split(",")]:
        cands = []
        for W in sw["windows"]:
            for arch in sw["archs"]:
                seed_everything(cfg["seed"])
                name = f"{key}_{arch}_w{W}"
                res = train_one(cfg, key, arch=arch, window=W, hparams=sw["arch_hparams"][arch],
                                epochs=sw["epochs"], max_batches=sw["batches_per_epoch"],
                                val_max_batches=None, name=name)
                model = res.pop("_model")
                best = min((h for h in res["history"] if h["val"].get("n")), key=lambda h: h["val"]["mae"])
                c = {"name": name, "arch": arch, "window_samples": W, "n_params": res["n_params"],
                     "val_mae": best["val"]["mae"], "val_rmse": best["val"]["rmse"],
                     "val_nll": best["val"]["nll"], "val_coverage_1sigma": best["val"]["coverage_1sigma"],
                     "val_coverage_2sigma": best["val"]["coverage_2sigma"], "best_epoch": best["epoch"],
                     "epochs_run": res["epochs_run"], "train_s": res["runtime_s"],
                     "device": res["device_final"], "batch_final": res["batch_size_final"],
                     "fallbacks": res["fallbacks"], "checkpoint": res["checkpoint"],
                     "baseline_constant_val_mae": res["baseline_constant_train_mean"]["val_mae"],
                     "cpu_latency_ms": cpu_latency_ms(model, tuple(res["input_shape"])),
                     "size_kb_fp32": round(res["n_params"] * 4 / 1024, 1),
                     "val_windows": res["val_data"]["windows"], "train_windows": res["train_data"]["windows"]}
                cands.append(c)
                print(f"  -> {name}: val MAE {c['val_mae']:.4f} RMSE {c['val_rmse']:.4f} (const {c['baseline_constant_val_mae']:.4f}) "
                      f"cov1/2 {c['val_coverage_1sigma']:.2f}/{c['val_coverage_2sigma']:.2f} | {c['cpu_latency_ms']:.3f} ms | "
                      f"{c['train_s']:.0f}s on {c['device']}", flush=True)
                report["candidates"][key] = cands
                (OUT / "sweep.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
        report["selection"][key] = select(cands, sw["selection"]["tie_tolerance"],
                                          sw["selection"].get("latency_tolerance", 0.0))
        print(f"[{key}] SELECTED {report['selection'][key]['selected']}: {report['selection'][key]['reason']}")
    (OUT / "sweep.json").write_text(json.dumps(report, indent=1, default=str) + "\n", encoding="utf-8")
    (OUT / "selection.json").write_text(json.dumps(report["selection"], indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
