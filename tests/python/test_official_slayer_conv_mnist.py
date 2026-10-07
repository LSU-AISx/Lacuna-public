"""Optional geometry and protocol checks for the official convolutional pilot."""

from __future__ import annotations

import io

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lava.lib.dl.slayer")

from examples import train_official_slayer_conv_mnist as pilot  # noqa: E402
from examples.train_official_slayer_mnist import encode_images  # noqa: E402
from lacuna.importers.slayer import import_slayer_feedforward  # noqa: E402


def test_spatial_encoder_preserves_dense_pilot_events():
    pixels = np.tile(np.array([0, 64, 128, 255], dtype=np.uint8), (2, 196))
    expected = encode_images(pixels, bins=4, seed=17)
    actual = pilot.encode_spatial(pixels, bins=4, seed=17)

    assert actual.shape == (2, 1, 28, 28, 4)
    assert torch.equal(actual.reshape(2, 784, 4), expected)


def test_model_geometry_and_checkpoint_preserve_import_fingerprint():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, _ = pilot.build_model()
    model.eval()
    inputs = pilot.encode_spatial(np.full((2, 784), 128), bins=3, seed=23)
    shapes = []
    with torch.no_grad():
        output = inputs
        for block in model:
            output = block(output)
            shapes.append(tuple(output.shape))
    assert shapes == [
        (2, 4, 12, 12, 3),
        (2, 8, 5, 5, 3),
        (2, 200, 3),
        (2, 10, 3),
    ]
    weight_counts = [
        block.synapse.weight.numel() for block in model if hasattr(block, "synapse")
    ]
    assert weight_counts == [100, 288, 2000]
    assert all(
        block.synapse.weight.requires_grad
        for block in model
        if hasattr(block, "synapse")
    )
    imported = import_slayer_feedforward(
        model, input_shape=(1, 28, 28), acknowledge_quantization=True
    )
    assert len(imported.deployment.network.graph.nodes) == 1570
    assert len(imported.deployment.network.graph.edges) == 23600
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored, _ = pilot.build_model()
    restored.load_state_dict(torch.load(stream, weights_only=True))
    restored.eval()
    reimported = import_slayer_feedforward(
        restored, input_shape=(1, 28, 28), acknowledge_quantization=True
    )
    assert reimported.source_fingerprint == imported.source_fingerprint
    assert reimported.network_sha256 == imported.network_sha256
    with torch.no_grad():
        assert torch.equal(restored(inputs), model(inputs))


def test_resume_rejects_completed_pilot(tmp_path):
    pilot.write_json(tmp_path / "report.json", {})
    args = pilot.parser().parse_args(["--resume", str(tmp_path)])
    with pytest.raises(ValueError, match="completed report"):
        pilot.run(args)


@pytest.mark.parametrize(
    "field,replacement,message",
    [
        ("architecture", [], "architecture changed"),
        ("training_sources_sha256", {}, "training code changed"),
        ("source_provenance", {}, "source or environment changed"),
        ("dataset_sha256", {}, "dataset changed"),
        ("checkpoint_sha256", "incorrect", "selected checkpoint changed"),
    ],
)
def test_resume_guards_training_identity(
    tmp_path, monkeypatch, field, replacement, message
):
    provenance = {"official_source_dirty": False, "official_source_commit": "test"}
    config = vars(pilot.parser().parse_args([])) | {"data_dir": str(tmp_path)}
    protocol = {
        "config": config,
        "architecture": pilot.ARCHITECTURE,
        "training_sources_sha256": {"training.py": "a"},
        "source_provenance": provenance,
        "dataset_sha256": {"dataset.gz": "b"},
    }
    summary = {"checkpoint_sha256": "incorrect"}
    if field == "checkpoint_sha256":
        summary[field] = replacement
    else:
        protocol[field] = replacement
    pilot.write_json(tmp_path / "protocol.json", protocol)
    pilot.write_json(tmp_path / "training_summary.json", summary)
    (tmp_path / "official_checkpoint.pt").write_bytes(b"checkpoint")
    monkeypatch.setattr(pilot, "source_provenance", lambda: provenance)
    monkeypatch.setattr(pilot, "training_source_hashes", lambda: {"training.py": "a"})
    monkeypatch.setattr(pilot, "data_hashes", lambda path: {"dataset.gz": "b"})
    monkeypatch.setattr(
        pilot,
        "load_mnist",
        lambda *args, **kwargs: (
            np.zeros((10, 784)),
            np.arange(10),
            np.zeros((10, 784)),
            np.arange(10),
        ),
    )
    args = pilot.parser().parse_args(["--resume", str(tmp_path)])
    with pytest.raises(ValueError, match=message):
        pilot.run(args)
