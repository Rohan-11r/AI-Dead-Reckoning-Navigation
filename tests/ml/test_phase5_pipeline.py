"""Phase 5 ML pipeline: registry, losses, dataset rules, augmentation, guards, fallback.

Model/loss tests use SYNTHETIC tensors. Dataset tests read the real synced IO-VNBD data
(marked ``dataset``) because the rules they check -- no window across a gap, no ground
truth in the inputs, no test split -- are properties of that data pipeline.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from training.configs.loader import ConfigError, load_training_config
from training.datasets.windows import FEATURE_NAMES, IovnbdWindows, make_features, random_rotation
from training.losses.regression import gaussian_nll, huber_nll_loss, regression_metrics
from training.models.registry import REGISTRY, build, n_params
from training.preprocessing.columns import ColumnRoleError, assert_model_targets
from training.trainers.trainer import is_oom, oom_fallback

REPO = Path(__file__).resolve().parents[2]
CFG = load_training_config()


# ---- registry ----------------------------------------------------------------------------

@pytest.mark.parametrize("arch", sorted(REGISTRY))
@pytest.mark.parametrize("positive", [True, False])
def test_every_model_builds_and_is_small(arch, positive):
    m = build(arch, in_channels=13, positive=positive, hidden=32)
    x = torch.randn(7, 13, 50)
    mean, logvar = m(x)
    assert mean.shape == (7,) and logvar.shape == (7,)
    assert n_params(m) < 1_000_000  # docs/ml_pipeline.md sec 5: phone-sized
    if positive:
        assert torch.all(mean >= 0)


def test_tcn_is_causal():
    m = build("tcn", in_channels=4, positive=False, hidden=16, levels=3).eval()
    x = torch.randn(2, 4, 40)
    y = x.clone()
    y[:, :, 30:] += 5.0  # change only the FUTURE of time step 29
    with torch.no_grad():
        a, b = m.tcn(x)[:, :, :30], m.tcn(y)[:, :, :30]
    torch.testing.assert_close(a, b)


def test_unknown_arch_raises():
    with pytest.raises(KeyError):
        build("transformer_xl", 13, True)


# ---- losses -------------------------------------------------------------------------------

def test_gaussian_nll_formula_and_minimum_at_true_variance():
    y, mu = torch.tensor([1.0]), torch.tensor([0.0])
    lv = torch.linspace(-4, 4, 801)
    nll = gaussian_nll(mu.expand_as(lv), lv, y.expand_as(lv))
    assert abs(float(lv[torch.argmin(nll)])) < 0.02  # optimum sigma^2 = err^2 = 1 -> logvar 0
    torch.testing.assert_close(gaussian_nll(mu, torch.zeros(1), y), torch.tensor([0.5]))


def test_combined_loss_is_finite_and_differentiable():
    mean = torch.randn(64, requires_grad=True)
    lv = torch.randn(64, requires_grad=True)
    loss, parts = huber_nll_loss(mean, lv * 50, torch.randn(64), 1.0, 0.5, (-8.0, 6.0))
    loss.backward()
    assert torch.isfinite(loss) and torch.all(torch.isfinite(mean.grad))
    assert set(parts) == {"huber", "nll"}


def test_coverage_metric_on_a_calibrated_gaussian():
    g = torch.Generator().manual_seed(0)
    y = torch.randn(100_000, generator=g) * 2.0
    m = regression_metrics(torch.zeros_like(y), torch.full_like(y, float(np.log(4.0))), y)
    assert abs(m["coverage_1sigma"] - 0.683) < 0.01 and abs(m["coverage_2sigma"] - 0.954) < 0.01


# ---- guards --------------------------------------------------------------------------------

def test_no_model_target_or_output_is_a_coordinate():
    # static check over the config and the training code (docs/ml_pipeline.md sec 5)
    raw = yaml.safe_load((REPO / "configs" / "training.yaml").read_text(encoding="utf-8"))
    assert_model_targets([m["target"] for m in raw["models"].values()])
    src = (REPO / "scripts" / "train" / "train_all.py").read_text(encoding="utf-8")
    names = re.findall(r'"name": "([a-z_]+)"', src)
    assert names and assert_model_targets(names)
    for p in (REPO / "training").rglob("*.py"):
        for m in re.finditer(r"target\s*=\s*[\"']([A-Za-z_]+)[\"']", p.read_text(encoding="utf-8")):
            assert_model_targets([m.group(1)])
    with pytest.raises(ColumnRoleError):
        assert_model_targets(["gt_lat_deg"])


def test_config_missing_key_raises(tmp_path):
    bad = yaml.safe_load((REPO / "configs" / "training.yaml").read_text(encoding="utf-8"))
    del bad["train"]["lr"]
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"train.lr"):
        load_training_config(p)


def test_oom_policy_halves_then_moves_to_cpu():
    assert oom_fallback(256, 16) == (128, False, "CUDA OOM -> batch size 128")
    b, to_cpu, _ = oom_fallback(16, 16)
    assert (b, to_cpu) == (16, True)
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_oom(RuntimeError("shape mismatch"))


# ---- augmentation ------------------------------------------------------------------------

def test_rotation_augmentation_preserves_the_invariant_features():
    rng = np.random.default_rng(1)
    acc = rng.normal(0, 1, (50, 3)) + np.array([0.0, 0.0, 9.8])
    grav = np.tile([0.1, -0.2, 9.8], (50, 1))
    gyro = np.zeros((50, 3))
    gyro[:, 2] = rng.normal(0, 0.1, 50)
    base = make_features(acc, grav, gyro, CFG["features"])
    for mode in ("yaw_tilt", "full"):
        R = random_rotation(rng, mode, 15.0)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        rot = make_features(acc @ R.T, grav @ R.T, gyro @ R.T, CFG["features"])
        inv = [FEATURE_NAMES.index(n) for n in ("f_norm", "f_vertical", "f_horizontal", "w_vertical")]
        np.testing.assert_allclose(rot[inv], base[inv], atol=1e-5)


# ---- dataset (real data) -----------------------------------------------------------------

@pytest.mark.dataset
def test_test_split_is_refused():
    with pytest.raises(PermissionError):
        IovnbdWindows("test", "gt_speed_mps", CFG, augment=False, verified_only=False)


@pytest.mark.dataset
def test_windows_never_cross_gaps_and_inputs_hold_no_ground_truth():
    ds = IovnbdWindows("val", "gt_speed_mps", CFG, augment=False, verified_only=False)
    assert len(ds) > 100 and sum(ds.exclusions.values()) > 0
    x, _ = ds[0]
    assert x.shape == (len(FEATURE_NAMES), CFG["data"]["window_samples"]) and torch.isfinite(x).all()
    assert all(not c.startswith("gt_") for c in [*ds.raw_channels, ds.gyro_channel])
    import pandas as pd

    from training.preprocessing.config import load_dataset_config
    d = pd.read_parquet(load_dataset_config().synced_root / f"{ds.sessions[0]}.parquet",
                        columns=["dt_s", "segment_id"])
    W = ds.W
    for si, end in ds.index:
        if si != 0:
            break
        w = d.iloc[end - W + 1:end + 1]
        assert w["segment_id"].nunique() == 1
        assert (w["dt_s"].iloc[1:] <= CFG["data"]["max_window_dt_s"]).all()


@pytest.mark.dataset
def test_augmentation_only_changes_training_inputs_not_targets():
    cfg = copy.deepcopy(CFG)
    ds_aug = IovnbdWindows("val", "gt_speed_mps", cfg, augment=True, verified_only=False, seed=3)
    ds_raw = IovnbdWindows("val", "gt_speed_mps", cfg, augment=False, verified_only=False)
    (xa, ya), (xr, yr) = ds_aug[5], ds_raw[5]
    assert float(ya) == float(yr)
    assert not torch.allclose(xa, xr)
