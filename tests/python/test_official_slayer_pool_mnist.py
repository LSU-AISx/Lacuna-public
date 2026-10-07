"""Optional geometry, fixed-pooling, and provenance checks for the trained pilot."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lava.lib.dl.slayer")

from examples import train_official_slayer_pool_mnist as pilot  # noqa: E402
from lacuna.importers.slayer import import_slayer_feedforward  # noqa: E402


def test_default_is_one_bounded_prior_protocol_with_fixed_seed():
    args = pilot.parser().parse_args([])
    assert (
        args.train_samples,
        args.validation_samples,
        args.test_samples,
    ) == (10000, 1000, 1000)
    assert (args.epochs, args.bins, args.batch_size) == (10, 32, 128)
    assert (args.learning_rate, args.seed, args.split_seed) == (0.003, 0, 137)


def test_model_geometry_fixed_pools_and_checkpoint_import_identity():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model, params = pilot.build_model()
    model.eval()
    assert params["current_decay"] == 1.0
    assert params["voltage_decay"] == 512 / 4096
    assert params["scale"] == 4096
    inputs = pilot.encode_spatial(np.full((2, 784), 128), bins=3, seed=23)
    shapes = []
    with torch.no_grad():
        output = inputs
        for block in model:
            output = block(output)
            shapes.append(tuple(output.shape))
    assert shapes == [
        (2, 4, 28, 28, 3),
        (2, 4, 14, 14, 3),
        (2, 8, 14, 14, 3),
        (2, 8, 7, 7, 3),
        (2, 392, 3),
        (2, 10, 3),
    ]
    assert [
        model[index].synapse.weight.numel() for index in pilot.TRAINED_BLOCKS
    ] == [36, 288, 3920]
    for index in pilot.NEURAL_BLOCKS:
        block = model[index]
        assert block.synapse.weight.requires_grad == (
            index in pilot.TRAINED_BLOCKS
        )
        assert block.synapse.pre_hook_fx is None
        assert block.delay is None
        assert block.delay_shift is False
    for index in pilot.POOL_BLOCKS:
        synapse = model[index].synapse
        assert synapse.weight.numel() == 4
        assert torch.all(synapse.weight == 1.0)
        assert synapse.kernel_size == synapse.stride == (2, 2, 1)
        assert synapse.padding == (0, 0, 0)
        assert synapse.dilation == (1, 1, 1)
    imported = import_slayer_feedforward(
        model, input_shape=(1, 28, 28), acknowledge_quantization=True
    )
    assert len(imported.metadata["layers"]) == 5
    assert len(imported.deployment.network.graph.nodes) == 6674
    assert len(imported.deployment.network.graph.edges) == 86720
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


def test_end_to_end_synthetic_preflight_updates_every_trained_layer_and_no_pool():
    torch.set_num_threads(1)
    rng_before = torch.random.get_rng_state()
    preflight = pilot.synthetic_preflight()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert preflight["data"].startswith("synthetic")
    diagnostic = preflight["frozen_default_sum_pool_diagnostic"]
    assert all(
        count > 0 for count in diagnostic["spike_totals_by_source_block"]
    )
    assert all(
        value > 0 for value in diagnostic["gradient_max_abs_by_trained_block"]
    )
    audit = diagnostic["weight_audit"]
    assert [row["source_block"] for row in audit] == [0, 1, 2, 3, 5]
    assert all(row["weight_change_l2"] > 0 for row in audit if row["trained"])
    assert all(
        row["unchanged"] and row["weight_change_l2"] == 0
        for row in audit
        if not row["trained"]
    )


@pytest.mark.parametrize("index", pilot.POOL_BLOCKS)
def test_weight_audit_rejects_changed_pool(index):
    model, _ = pilot.build_model()
    before = pilot.weight_snapshot(model)
    with torch.no_grad():
        model[index].synapse.weight.add_(0.125)
    with pytest.raises(RuntimeError, match="fixed pool weights changed"):
        pilot.weight_audit(model, before)


def test_weight_audit_rejects_unchanged_learned_layer():
    model, _ = pilot.build_model()
    with pytest.raises(RuntimeError, match="trained weights did not change"):
        pilot.weight_audit(
            model, pilot.weight_snapshot(model), require_learned_change=True
        )


def test_validation_selection_uses_accuracy_loss_then_earlier_epoch():
    key = pilot.selection_key
    assert key({"accuracy": 0.9, "loss": 0.8}, 3) > key(
        {"accuracy": 0.8, "loss": 0.1}, 1
    )
    assert key({"accuracy": 0.9, "loss": 0.1}, 3) > key(
        {"accuracy": 0.9, "loss": 0.8}, 1
    )
    assert key({"accuracy": 0.9, "loss": 0.1}, 1) > key(
        {"accuracy": 0.9, "loss": 0.1}, 3
    )


@pytest.mark.parametrize(
    "field",
    [
        "architecture",
        "source_provenance",
        "official_sources_sha256",
        "workflow_sources_sha256",
        "dataset_sha256",
        "lacuna_library_sha256",
    ],
)
def test_frozen_protocol_rejects_changed_training_or_deployment_identity(
    field,
):
    current = {
        "source_provenance": {"official_source_commit": "unit-test"},
        "official_sources_sha256": {"official.py": "a"},
        "workflow_sources_sha256": {"importer.py": "b"},
        "dataset_sha256": {"dataset.gz": "c"},
        "lacuna_library_sha256": "d",
    }
    protocol = {"architecture": pilot.ARCHITECTURE, **current}
    pilot.verify_frozen_protocol(protocol, current)
    protocol[field] = "changed"
    with pytest.raises(
        ValueError, match="changed since the protocol was frozen"
    ):
        pilot.verify_frozen_protocol(protocol, current)


def test_source_manifest_contains_pilot_importers_and_core():
    hashes = pilot.workflow_source_hashes()
    assert {
        "examples/train_official_slayer_pool_mnist.py",
        "examples/train_official_slayer_conv_mnist.py",
        "examples/train_official_slayer_mnist.py",
        "src/lacuna/importers/slayer.py",
        "src/lacuna/importers/feedforward_lif.py",
        "src/lacuna/importers/dense_lif.py",
    } <= hashes.keys()
    assert any(
        name.startswith("c/") and name.endswith(".c") for name in hashes
    )
    assert all(len(value) == 64 for value in hashes.values())
    assert (
        "src/lava/lib/dl/slayer/synapse/layer.py"
        in pilot.official_source_hashes()
    )


def test_protocol_is_persisted_before_training_without_test_encoding(
    tmp_path, monkeypatch
):
    class TrainingReached(Exception):
        pass

    labels = np.tile(np.arange(10), 3)
    monkeypatch.setattr(
        pilot,
        "load_mnist",
        lambda *args, **kwargs: (
            np.full((30, 784), 128),
            labels,
            np.full((30, 784), 200),
            labels,
        ),
    )
    monkeypatch.setattr(
        pilot,
        "frozen_provenance",
        lambda *args: {"test_provenance": "fixture"},
    )
    monkeypatch.setattr(
        pilot, "synthetic_preflight", lambda: {"data": "unit-test fixture"}
    )
    calls = []
    encode = pilot.encode_spatial

    def record_encode(images, *, bins, seed):
        calls.append(seed)
        return encode(images, bins=bins, seed=seed)

    def observe_train(
        args, output, images, targets, validation_inputs, validation_y
    ):
        protocol = json.loads((output / "protocol.json").read_text())
        assert protocol["test_used_for_selection"] is False
        assert protocol["test_provenance"] == "fixture"
        assert protocol["neural_source_blocks"] == [0, 1, 2, 3, 5]
        assert not set(protocol["train_indices"]) & set(
            protocol["validation_indices"]
        )
        assert calls == [200000]
        assert len(images) == len(validation_inputs) == 10
        raise TrainingReached

    monkeypatch.setattr(pilot, "encode_spatial", record_encode)
    monkeypatch.setattr(pilot, "train", observe_train)
    args = pilot.parser().parse_args(
        [
            "--output",
            str(tmp_path / "pilot"),
            "--train-samples",
            "10",
            "--validation-samples",
            "10",
            "--test-samples",
            "10",
            "--bins",
            "2",
        ]
    )
    with pytest.raises(TrainingReached):
        pilot.run(args)


def test_resume_rejects_completed_pilot(tmp_path):
    pilot.write_json(tmp_path / "report.json", {})
    args = pilot.parser().parse_args(["--resume", str(tmp_path)])
    with pytest.raises(ValueError, match="completed report"):
        pilot.run(args)


@pytest.mark.parametrize("target", ["checkpoint", "protocol"])
def test_restore_rejects_artifact_hash_changes(tmp_path, target):
    checkpoint = tmp_path / "official_checkpoint.pt"
    protocol = tmp_path / "protocol.json"
    checkpoint.write_bytes(b"unit-test checkpoint placeholder")
    protocol.write_text("{}\n")
    summary = {
        "checkpoint_sha256": pilot.sha256(checkpoint),
        "protocol_sha256": pilot.sha256(protocol),
    }
    (checkpoint if target == "checkpoint" else protocol).write_bytes(
        b"changed"
    )
    with pytest.raises(ValueError, match=f"{target} changed since training"):
        pilot.restore_model(tmp_path, summary)
