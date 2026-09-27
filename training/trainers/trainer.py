"""Training loop with device selection and GPU out-of-memory fallback (Phase 5).

* Device: ``auto`` picks CUDA when available, else CPU (the installed torch is the CPU
  build as of Phase 5; requirements-gpu.txt holds the CUDA swap).
* OOM fallback: a CUDA out-of-memory error halves the batch size and restarts the epoch,
  down to ``min_batch_size``; below that the run moves to CPU. Every fallback is logged
  in the result -- never silent.
* Reproducible: seeded numpy/torch; deterministic algorithms where available.
* Checkpoint: best validation epoch -> models/checkpoints/<name>/best.pt with the config,
  seed, commit, split-manifest hash, input feature list and output unit.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from training.losses.regression import huber_nll_loss, regression_metrics
from training.preprocessing.config import REPO_ROOT, load_dataset_config


def seed_everything(seed: int) -> None:
    np.random.seed(seed)  # noqa: NPY002 -- only for third-party code; ours uses seeded Generators
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device: cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def oom_fallback(batch: int, min_batch: int) -> tuple[int, bool, str]:
    """Policy after a CUDA OOM: halve the batch while >= ``min_batch``, else move to CPU.
    Returns (new_batch, move_to_cpu, log message)."""
    if batch // 2 >= min_batch:
        return batch // 2, False, f"CUDA OOM -> batch size {batch // 2}"
    return batch, True, f"CUDA OOM at batch {batch} (< 2 x min {min_batch}) -> CPU"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _loader(ds, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=g, drop_last=False)


@torch.no_grad()
def evaluate(model, ds, device, batch_size: int, logvar_clamp, max_batches: int | None = None) -> dict:
    model.eval()
    means, lvs, ys = [], [], []
    for b, (x, y) in enumerate(_loader(ds, batch_size, False, 0, 0)):
        if max_batches is not None and b >= max_batches:
            break
        m, lv = model(x.to(device))
        means.append(m.cpu())
        lvs.append(lv.cpu())
        ys.append(y)
    if not ys:
        return {"n": 0}
    out = regression_metrics(torch.cat(means), torch.cat(lvs), torch.cat(ys), logvar_clamp)
    out["n"] = int(sum(len(y) for y in ys))
    return out


def fit(model: torch.nn.Module, train_ds, val_ds, tcfg: dict, name: str, meta: dict,
        epochs: int | None = None, max_batches: int | None = None,
        on_batch: Callable[[int], None] | None = None) -> dict:
    """Train ``model``; returns a result dict (history, best metrics, fallbacks, paths)."""
    device = select_device(tcfg["device"])
    batch = int(tcfg["batch_size"])
    epochs = int(epochs or tcfg["epochs"])
    clamp = tuple(tcfg["logvar_clamp"])
    fallbacks: list[str] = []
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    best, best_state, bad_epochs, history = math.inf, None, 0, []
    t0 = time.time()
    epoch = 0
    while epoch < epochs:
        try:
            model.train()
            losses, parts = [], []
            for b, (x, y) in enumerate(_loader(train_ds, batch, True, tcfg["num_workers"], 1000 + epoch)):
                if max_batches is not None and b >= max_batches:
                    break
                x, y = x.to(device), y.to(device)
                mean, logvar = model(x)
                loss, comp = huber_nll_loss(mean, logvar, y, tcfg["huber_delta"], tcfg["nll_weight"], clamp)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch} batch {b}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
                opt.step()
                losses.append(float(loss.detach()))
                parts.append(comp)
                if on_batch:
                    on_batch(b)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not (device.type == "cuda" and is_oom(exc)):
                raise
            torch.cuda.empty_cache()
            batch, to_cpu, msg = oom_fallback(batch, tcfg["min_batch_size"])
            fallbacks.append(f"epoch {epoch}: {msg}")
            if to_cpu:
                device = torch.device("cpu")
                model.to(device)
                opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
            continue  # retry the same epoch
        val = evaluate(model, val_ds, device, batch, clamp, max_batches)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else None,
                        "train_huber": float(np.mean([p["huber"] for p in parts])) if parts else None,
                        "train_nll": float(np.mean([p["nll"] for p in parts])) if parts else None,
                        "train_batches": len(losses), "val": val})
        score = val.get("mae", math.inf) if val.get("n") else float(np.mean(losses))
        if score < best:
            best, bad_epochs = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stopping_patience"]:
                break
        epoch += 1

    ckpt_dir = REPO_ROOT / "models" / "checkpoints" / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_dataset_config().split_manifest
    record = {
        **meta, "name": name, "seed": meta.get("seed"), "commit": git_commit(),
        "split_manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
        "device_final": device.type, "batch_size_final": batch, "fallbacks": fallbacks,
        "epochs_run": len(history), "best_val_mae": best if math.isfinite(best) else None,
        "runtime_s": round(time.time() - t0, 1),
    }
    torch.save({"state_dict": best_state, "record": record}, ckpt_dir / "best.pt")
    return {**record, "history": history, "checkpoint": str((ckpt_dir / "best.pt").relative_to(REPO_ROOT))}


def model_card(name: str, record: dict, out_dir: Path) -> Path:
    """models/metadata/<name>.json: every output's name, unit and physical range (sec 5)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    return p
