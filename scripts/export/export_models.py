#!/usr/bin/env python3
"""Export the selected Phase 6 models to ONNX with a model card each.

    .venv/Scripts/python.exe scripts/export/export_models.py [--names model_a_tcn_w50,...]

Default: the models chosen in reports/phase6/selection.json. For each:
* rebuild the architecture from the checkpoint record, load the best-epoch weights
* export to models/exported/<name>.onnx: opset 17, TorchScript exporter (Phase 1 verified;
  no PYTHONUTF8 dependency), dynamic batch axis, inputs "window" (B, C, W), outputs
  "mean" (B,) and "logvar" (B,)
* onnx.checker must pass
* write models/exported/<name>.model_card.json and a tracked copy in models/metadata/
  (the .onnx binaries are gitignored artefacts; the cards are provenance)

The model card is the INPUT CONTRACT the Android side must reproduce bit-for-bit:
feature order, window length, the fixed physical scaling constants and how the invariant
features are computed. Numerical equivalence is checked by scripts/validate/validate_models.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import onnx  # noqa: E402
import torch  # noqa: E402

from training.models.registry import build  # noqa: E402
from training.trainers.trainer import git_commit  # noqa: E402

EXPORT_DIR = REPO_ROOT / "models" / "exported"
META_DIR = REPO_ROOT / "models" / "metadata"
OPSET = 17


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_checkpoint(name: str) -> tuple[torch.nn.Module, dict, Path]:
    ckpt = REPO_ROOT / "models" / "checkpoints" / name / "best.pt"
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    rec = blob["record"]
    model = build(rec["arch"], in_channels=rec["input_shape"][0], positive=rec["output"]["positive"],
                  **rec["hparams"])
    if blob["state_dict"] is None:
        raise RuntimeError(f"{name}: checkpoint holds no weights")
    model.load_state_dict(blob["state_dict"])
    return model.eval(), rec, ckpt


def export(name: str) -> dict:
    model, rec, ckpt = load_checkpoint(name)
    C, W = rec["input_shape"]
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = EXPORT_DIR / f"{name}.onnx"
    dummy = torch.randn(2, C, W)
    torch.onnx.export(model, (dummy,), str(onnx_path), opset_version=OPSET, dynamo=False,
                      input_names=["window"], output_names=["mean", "logvar"],
                      dynamic_axes={"window": {0: "batch"}, "mean": {0: "batch"}, "logvar": {0: "batch"}})
    onnx.checker.check_model(onnx.load(str(onnx_path)), full_check=True)
    card = {
        "name": name, "exported": datetime.now().isoformat(timespec="seconds"),
        "export_commit": git_commit(), "training_commit": rec.get("commit"),
        "format": {"onnx_opset": OPSET, "exporter": "torch.onnx (TorchScript, dynamo=False)",
                   "torch": torch.__version__, "onnx": onnx.__version__, "dtype": "float32"},
        "files": {"onnx": onnx_path.relative_to(REPO_ROOT).as_posix(), "onnx_sha256": sha256(onnx_path),
                  "checkpoint": ckpt.relative_to(REPO_ROOT).as_posix(), "checkpoint_sha256": sha256(ckpt)},
        "architecture": {"arch": rec["arch"], "hparams": rec["hparams"], "n_params": rec["n_params"]},
        "input": {
            "name": "window", "shape": ["batch", C, W], "layout": "channels x time, oldest sample first",
            "sample_rate_hz": 10, "window_s": W * 0.1,
            "features": rec["inputs"],
            "scaling": rec["feature_scaling"],
            "definition": {
                "acc_x..z": "device-frame specific force [m/s^2] / accel_scale_mps2",
                "grav_x..z": "Android GRAVITY vector [m/s^2] / accel_scale_mps2",
                "gyro_x..z": "[0, 0, w_z] / gyro_scale_radps; w_z = logged gyro column 2 (vertical); "
                             "columns 1/3 are not angular rates (reports/phase4/gyro_axis_resolution.json)",
                "f_norm": "|f| / accel_scale_mps2 - 1",
                "f_vertical": "(f . g_hat) / accel_scale_mps2 - 1,  g_hat = grav / |grav|",
                "f_horizontal": "|f x g_hat| / accel_scale_mps2",
                "w_vertical": "(w . g_hat) / gyro_scale_radps",
            },
            "rule": "phone channels only -- no CAN/OBD/VBOX input (SIH26168)",
        },
        "outputs": {
            "mean": {"quantity": rec["output"]["name"], "unit": rec["output"]["unit"],
                     "physical_range": rec["output"]["range"],
                     "note": "softplus-constrained >= 0" if rec["output"]["positive"] else "unconstrained"},
            "logvar": {"quantity": f"log variance of {rec['output']['name']}",
                       "unit": f"log(({rec['output']['unit']})^2)",
                       "note": "clamp to [-8, 6] before use, as in training"},
        },
        "target": rec["target"], "training_data": {k: rec[k] for k in ("train_data", "val_data")},
        "split_manifest_sha256": rec["split_manifest_sha256"], "seed": rec["seed"],
        "best_val_mae": rec["best_val_mae"],
    }
    for d in (EXPORT_DIR, META_DIR):
        (d / f"{name}.model_card.json").write_text(json.dumps(card, indent=1, default=str) + "\n",
                                                   encoding="utf-8")
    print(f"exported {name}: {onnx_path.relative_to(REPO_ROOT)} ({onnx_path.stat().st_size / 1024:.1f} kB), checker OK")
    return card


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", default=None, help="comma list of checkpoint names")
    args = ap.parse_args()
    if args.names:
        names = args.names.split(",")
    else:
        sel = json.loads((REPO_ROOT / "reports" / "phase6" / "selection.json").read_text(encoding="utf-8"))
        names = [v["selected"] for v in sel.values()]
    for n in names:
        export(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
