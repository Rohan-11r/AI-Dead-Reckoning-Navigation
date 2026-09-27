"""Every registered architecture must export to ONNX and match PyTorch to < 1e-5.

Models here have random (SYNTHETIC) weights and inputs: this pins exportability of each
architecture before any expensive training depends on it. The trained, selected models
are checked separately by scripts/validate/validate_models.py.
"""

from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from training.models.registry import REGISTRY, build

HPARAMS = {"tcn": {"hidden": 16, "levels": 3, "kernel": 3}, "cnn1d": {"hidden": 16, "layers": 2, "kernel": 5},
           "gru": {"hidden": 16, "layers": 1}, "lstm": {"hidden": 16, "layers": 1}}


@pytest.mark.parametrize("arch", sorted(REGISTRY))
@pytest.mark.parametrize("positive", [True, False])
def test_architecture_exports_and_matches_onnxruntime(arch, positive, tmp_path):
    torch.manual_seed(0)
    model = build(arch, in_channels=13, positive=positive, dropout=0.1, **HPARAMS[arch]).eval()
    path = tmp_path / f"{arch}.onnx"
    torch.onnx.export(model, (torch.randn(2, 13, 30),), str(path), opset_version=17, dynamo=False,
                      input_names=["window"], output_names=["mean", "logvar"],
                      dynamic_axes={"window": {0: "batch"}, "mean": {0: "batch"}, "logvar": {0: "batch"}})
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    for batch in (1, 9, 64):
        x = rng.normal(0, 1, (batch, 13, 30)).astype(np.float32)
        with torch.no_grad():
            tm, tl = model(torch.from_numpy(x))
        om, ol = sess.run(["mean", "logvar"], {"window": x})
        assert np.max(np.abs(tm.numpy() - om)) < 1e-5
        assert np.max(np.abs(tl.numpy() - ol)) < 1e-5
