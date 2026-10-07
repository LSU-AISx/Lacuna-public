"""Optional geometry, learning, and transfer checks for the AlexNet pilot."""

from __future__ import annotations

import io
import json
from copy import deepcopy

import numpy as np
import pytest


torch = pytest.importorskip("torch")
slayer = pytest.importorskip("lava.lib.dl.slayer")

from examples import train_official_slayer_alexnet_mnist as pilot  # noqa: E402
from examples.train_official_slayer_conv_mnist import (
    encode_spatial,
)  # noqa: E402
from lacuna.importers.slayer import (  # noqa: E402
    import_slayer_feedforward,
    validate_slayer_feedforward,
)


def test_default_protocol_is_one_bounded_fixed_seed_experiment():
    args = pilot.parser().parse_args([])
    assert (
        args.train_samples,
        args.validation_samples,
        args.test_samples,
    ) == (10000, 1000, 1000)
    assert (args.epochs, args.bins, args.batch_size) == (10, 32, 16)
    assert (args.learning_rate, args.seed, args.split_seed) == (0.001, 0, 137)
    config = pilot.AlexNetConfig()
    assert config.channels == (8, 16, 24, 24, 16)
    assert config.hidden == (64, 32)
    assert (config.tau_grad, config.scale_grad, config.weight_scale) == (
        0.01,
        10.0,
        2.0,
    )


def test_encoder_preserves_events_and_adds_only_zero_border():
    pixels = np.tile(np.array([0, 64, 128, 255], dtype=np.uint8), (2, 196))
    expected = encode_spatial(pixels, bins=5, seed=17)
    actual = pilot.encode_inputs(pixels, bins=5, seed=17)
    assert actual.shape == (2, 1, 32, 32, 5)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual[:, :, 2:-2, 2:-2, :], expected)
    assert torch.equal(actual, pilot.encode_inputs(pixels, bins=5, seed=17))
    assert bool(torch.all((actual == 0) | (actual == 1)))
    assert torch.count_nonzero(actual[:, :, :2, :, :]) == 0
    assert torch.count_nonzero(actual[:, :, -2:, :, :]) == 0
    assert torch.count_nonzero(actual[:, :, :, :2, :]) == 0
    assert torch.count_nonzero(actual[:, :, :, -2:, :]) == 0
    assert not torch.equal(
        actual, pilot.encode_inputs(pixels, bins=5, seed=18)
    )


def test_default_geometry_has_five_convolutions_and_three_dense_layers():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        config = pilot.AlexNetConfig()
        model, params = pilot.build_model(config)
        model.eval()
        inputs = pilot.encode_inputs(np.full((1, 784), 128), bins=3, seed=23)
        shapes = []
        with torch.no_grad():
            output = inputs
            for block in model:
                output = block(output)
                shapes.append(tuple(output.shape[1:-1]))
    assert shapes == [
        (8, 32, 32),
        (8, 16, 16),
        (16, 16, 16),
        (16, 8, 8),
        (24, 8, 8),
        (24, 8, 8),
        (16, 8, 8),
        (16, 4, 4),
        (256,),
        (64,),
        (32,),
        (10,),
    ]
    assert pilot.TRAINED_BLOCKS == (0, 2, 4, 5, 6, 9, 10, 11)
    assert pilot.POOL_BLOCKS == (1, 3, 7)
    assert pilot.NEURAL_BLOCKS == (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11)
    assert [
        model[index].synapse.weight.numel() for index in pilot.TRAINED_BLOCKS
    ] == [200, 3200, 3456, 5184, 3456, 16384, 2048, 320]
    assert params["threshold"] == params["current_decay"] == 1.0
    assert params["voltage_decay"] == 512 / 4096
    assert params["scale"] == 4096
    assert params["tau_grad"] == config.tau_grad
    assert params["scale_grad"] == config.scale_grad
    assert params["persistent_state"] is False
    assert params["requires_grad"] is False
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
    documented = {
        row["source_block"]: tuple(row["output_shape"])
        for row in pilot.architecture(config)
        if "source_block" in row
    }
    assert documented == dict(enumerate(shapes))


def test_compact_model_checkpoint_and_all_eleven_imported_layers(core):
    torch.set_num_threads(1)
    config = pilot.AlexNetConfig(channels=(2, 2, 2, 2, 2), hidden=(4, 3))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model, _ = pilot.build_model(config)
        model.eval()
        generator = torch.Generator().manual_seed(31)
        with torch.no_grad():
            for index in pilot.TRAINED_BLOCKS:
                weight = model[index].synapse.weight
                weight.copy_(
                    torch.randint(-2, 6, weight.shape, generator=generator) / 4
                )
        inputs = (
            torch.rand(2, 1, 32, 32, 3, generator=generator) < 0.2
        ).float()
        imported = import_slayer_feedforward(
            model, input_shape=(1, 32, 32), acknowledge_quantization=True
        )
        report = validate_slayer_feedforward(
            model, imported, inputs, library=core._lib._name
        )
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        restored, _ = pilot.build_model(config)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        restored.eval()
        reimported = import_slayer_feedforward(
            restored, input_shape=(1, 32, 32), acknowledge_quantization=True
        )
        with torch.no_grad():
            assert torch.equal(restored(inputs), model(inputs))
    assert imported.metadata["source_blocks"] == list(pilot.NEURAL_BLOCKS)
    assert imported.deployment.layer_shapes == (
        (2, 32, 32),
        (2, 16, 16),
        (2, 16, 16),
        (2, 8, 8),
        (2, 8, 8),
        (2, 8, 8),
        (2, 8, 8),
        (2, 4, 4),
        (4,),
        (3,),
        (10,),
    )
    assert len(imported.metadata["layers"]) == len(report["layers"]) == 11
    assert report["exact_spike_match_on_batch"], report
    assert all(layer["mismatched_bins"] == 0 for layer in report["layers"])
    assert report["prediction_agreement"] == 1.0
    assert report["off_grid_spikes"] == []
    assert reimported.source_fingerprint == imported.source_fingerprint
    assert reimported.network_sha256 == imported.network_sha256


def test_synthetic_step_updates_all_eight_learned_weights_and_no_pool():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model, _ = pilot.build_model(pilot.AlexNetConfig())
        before = pilot.weight_snapshot(model)
        inputs = pilot.encode_inputs(np.full((16, 784), 200), bins=16, seed=2)
        loss_function = slayer.loss.SpikeRate(
            true_rate=0.2, false_rate=0.02, reduction="mean"
        )
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ],
            lr=0.003,
        )
        optimizer.zero_grad()
        loss = loss_function(model(inputs), torch.arange(16) % 10)
        assert bool(torch.isfinite(loss))
        loss.backward()
        for index in pilot.TRAINED_BLOCKS:
            gradient = model[index].synapse.weight.grad
            assert gradient is not None, index
            assert bool(torch.isfinite(gradient).all()), index
            assert float(gradient.abs().max()) > 0, index
        for index in pilot.POOL_BLOCKS:
            assert model[index].synapse.weight.grad is None
        optimizer.step()
        audit = pilot.weight_audit(model, before)
    assert [row["source_block"] for row in audit] == list(pilot.NEURAL_BLOCKS)
    assert sum(row["trained"] for row in audit) == 8
    assert all(row["weight_change_l2"] > 0 for row in audit if row["trained"])
    assert all(
        row["unchanged"] and row["weight_change_l2"] == 0
        for row in audit
        if not row["trained"]
    )


@pytest.mark.parametrize("index", (1, 3, 7))
def test_weight_audit_rejects_changed_pool(index):
    model, _ = pilot.build_model(pilot.AlexNetConfig())
    before = pilot.weight_snapshot(model)
    with torch.no_grad():
        model[index].synapse.weight.add_(0.125)
    with pytest.raises(RuntimeError, match="fixed pool weights changed"):
        pilot.weight_audit(model, before)


def test_weight_audit_reports_unchanged_weights_without_inventing_updates():
    model, _ = pilot.build_model(pilot.AlexNetConfig())
    audit = pilot.weight_audit(model, pilot.weight_snapshot(model))
    assert all(row["unchanged"] for row in audit)
    assert all(row["weight_change_l2"] == 0 for row in audit)
    assert all(
        row["initial_sha256"] == row["selected_sha256"] for row in audit
    )


@pytest.mark.parametrize("sizes", [(151, 47, 63), (550, 50, 100)])
def test_splits_are_exact_deterministic_disjoint_and_support_full_data(sizes):
    train_labels = np.arange(600) % 10
    test_labels = np.arange(100) % 10
    actual = pilot.split_indices(train_labels, test_labels, *sizes, seed=137)
    repeated = pilot.split_indices(train_labels, test_labels, *sizes, seed=137)
    changed = pilot.split_indices(train_labels, test_labels, *sizes, seed=138)
    assert tuple(map(len, actual)) == sizes
    assert all(np.array_equal(a, b) for a, b in zip(actual, repeated))
    assert all(not np.array_equal(a, b) for a, b in zip(actual, changed))
    train_ids, val_ids, test_ids = actual
    assert not set(train_ids) & set(val_ids)
    for indices, available in zip(actual, (600, 600, 100)):
        assert len(np.unique(indices)) == len(indices)
        assert indices.min() >= 0 and indices.max() < available
    if sizes == (550, 50, 100):
        assert set(train_ids) | set(val_ids) == set(range(600))
        assert set(test_ids) == set(range(100))


@pytest.mark.parametrize(
    "sizes,message",
    [
        ((0, 10, 10), "positive integers"),
        ((10, -1, 10), "positive integers"),
        ((10, 10, True), "positive integers"),
        ((10.5, 10, 10), "positive integers"),
        ((591, 10, 10), "exceed the training partition"),
        ((10, 10, 101), "exceed the canonical test partition"),
    ],
)
def test_split_rejects_invalid_counts(sizes, message):
    with pytest.raises(ValueError, match=message):
        pilot.split_indices(
            np.arange(600) % 10, np.arange(100) % 10, *sizes, 137
        )


def test_split_keeps_previous_training_diagnostics_out_of_validation():
    train_labels, test_labels = np.arange(600) % 10, np.arange(100) % 10
    reserved = (5, 117, 231, 404, 599)
    kwargs = dict(seed=137, training_only=reserved)
    actual = pilot.split_indices(
        train_labels, test_labels, 550, 50, 100, **kwargs
    )
    repeated = pilot.split_indices(
        train_labels, test_labels, 550, 50, 100, **kwargs
    )
    train_ids, val_ids, test_ids = actual
    assert all(np.array_equal(a, b) for a, b in zip(actual, repeated))
    assert tuple(map(len, actual)) == (550, 50, 100)
    assert set(reserved) <= set(train_ids)
    assert not set(reserved) & set(val_ids)
    assert not set(train_ids) & set(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(600))
    assert set(test_ids) == set(range(100))


@pytest.mark.parametrize(
    "reserved,message",
    [
        ((1, 1), "unique valid integers"),
        ((-1,), "unique valid integers"),
        ((600,), "unique valid integers"),
        ((True,), "unique valid integers"),
        ((1.0,), "unique valid integers"),
        (tuple(range(11)), "cannot hold all training-only indices"),
    ],
)
def test_split_rejects_invalid_training_only_indices(reserved, message):
    with pytest.raises(ValueError, match=message):
        pilot.split_indices(
            np.arange(600) % 10,
            np.arange(100) % 10,
            10,
            10,
            10,
            137,
            training_only=reserved,
        )


def _selected_checkpoint(output):
    config = pilot.AlexNetConfig(channels=(2, 2, 2, 2, 2), hidden=(4, 3))
    model, params = pilot.build_model(config)
    before = pilot.weight_snapshot(model)
    with torch.no_grad():
        for index in pilot.TRAINED_BLOCKS:
            model[index].synapse.weight.add_(0.125)
    pilot.write_json(output / "protocol.json", {"fixture": "unit test"})
    protocol_hash = pilot.sha256(output / "protocol.json")
    state = {
        "model_state_dict": model.state_dict(),
        "neuron_params": params,
        "architecture": pilot.architecture(config),
        "selected_epoch": 2,
        "protocol_sha256": protocol_hash,
        "initial_weights": before,
    }
    path = output / "official_checkpoint.pt"
    pilot.save_checkpoint(path, state)
    summary = {
        "checkpoint_sha256": pilot.sha256(path),
        "protocol_sha256": protocol_hash,
        "selected_epoch": 2,
        "weight_audit": pilot.weight_audit(model, before),
    }
    return config, model, state, summary


def test_selected_checkpoint_restores_identical_tensors_and_eval_mode(
    tmp_path,
):
    config, model, _, summary = _selected_checkpoint(tmp_path)
    restored = pilot.restore_selected(tmp_path, config, summary)
    assert restored.training is False
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name]), name
    assert not (tmp_path / "official_checkpoint.pt.tmp").exists()


@pytest.mark.parametrize(
    "field,message",
    [
        ("checkpoint_hash", "selected checkpoint changed"),
        ("protocol", "selected checkpoint protocol changed"),
        ("architecture", "model contract changed"),
        ("neuron_params", "model contract changed"),
        ("selected_epoch", "selection metadata changed"),
        ("protocol_sha256", "selection metadata changed"),
        ("weight_audit", "weight audit changed"),
    ],
)
def test_selected_checkpoint_rejects_tampered_identity(
    tmp_path, field, message
):
    config, _, state, summary = _selected_checkpoint(tmp_path)
    checkpoint_path = tmp_path / "official_checkpoint.pt"
    if field == "checkpoint_hash":
        summary["checkpoint_sha256"] = "incorrect"
    elif field == "protocol":
        pilot.write_json(tmp_path / "protocol.json", {"fixture": "changed"})
    elif field == "weight_audit":
        summary["weight_audit"][0]["weight_count"] += 1
    else:
        if field == "architecture":
            state[field] = []
        elif field == "neuron_params":
            state[field]["voltage_decay"] = 0.5
        elif field == "selected_epoch":
            state[field] = 3
        else:
            state[field] = "incorrect"
        pilot.save_checkpoint(checkpoint_path, state)
        summary["checkpoint_sha256"] = pilot.sha256(checkpoint_path)
    with pytest.raises(ValueError, match=message):
        pilot.restore_selected(tmp_path, config, summary)


def test_resume_state_preserves_tensors_and_binds_complete_epoch(tmp_path):
    state = {
        "protocol_sha256": "frozen-protocol",
        "epoch": 3,
        "model_state_dict": {"weights": torch.tensor([1.0, -2.0])},
        "torch_rng_state": torch.random.get_rng_state(),
    }
    pilot.save_resume_state(tmp_path, state)
    restored = pilot.load_resume_state(tmp_path, "frozen-protocol")
    assert restored["epoch"] == state["epoch"]
    assert restored["protocol_sha256"] == state["protocol_sha256"]
    assert torch.equal(
        restored["model_state_dict"]["weights"],
        state["model_state_dict"]["weights"],
    )
    assert torch.equal(restored["torch_rng_state"], state["torch_rng_state"])
    manifest = json.loads((tmp_path / "resume.json").read_text())
    assert manifest == {
        "checkpoint_file": "epoch-000003.pt",
        "checkpoint_sha256": pilot.sha256(tmp_path / "epoch-000003.pt"),
        "protocol_sha256": "frozen-protocol",
        "epoch": 3,
    }
    assert not (tmp_path / "epoch-000003.pt.tmp").exists()


@pytest.mark.parametrize(
    "field,message",
    [
        ("protocol_hash", "resume protocol does not match"),
        ("checkpoint_hash", "checkpoint changed or is incomplete"),
        ("checkpoint_file", "filename does not match epoch"),
        ("epoch", "checkpoint metadata does not match"),
        ("protocol_sha256", "checkpoint metadata does not match"),
    ],
)
def test_resume_state_rejects_tampering(tmp_path, field, message):
    state = {"protocol_sha256": "frozen-protocol", "epoch": 3}
    pilot.save_resume_state(tmp_path, state)
    manifest_path = tmp_path / "resume.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "protocol_hash":
        manifest["protocol_sha256"] = "incorrect"
    elif field == "checkpoint_hash":
        manifest["checkpoint_sha256"] = "incorrect"
    elif field == "checkpoint_file":
        manifest["checkpoint_file"] = "../outside-checkpoint.pt"
    else:
        state[field] = 4 if field == "epoch" else "incorrect"
        path = tmp_path / "epoch-000003.pt"
        pilot.save_checkpoint(path, state)
        manifest["checkpoint_sha256"] = pilot.sha256(path)
    pilot.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match=message):
        pilot.load_resume_state(tmp_path, "frozen-protocol")


def test_interrupted_resume_publication_retains_previous_complete_epoch(
    tmp_path, monkeypatch
):
    first = {
        "protocol_sha256": "frozen-protocol",
        "epoch": 1,
        "weights": torch.tensor([1.0]),
    }
    pilot.save_resume_state(tmp_path, first)
    manifest_before = (tmp_path / "resume.json").read_bytes()
    checkpoint_before = (tmp_path / "epoch-000001.pt").read_bytes()
    original_write = pilot.write_json

    def interrupt_manifest(path, value):
        if path.name == "resume.json":
            raise OSError("simulated interruption before manifest publication")
        original_write(path, value)

    monkeypatch.setattr(pilot, "write_json", interrupt_manifest)
    with pytest.raises(OSError, match="simulated interruption"):
        pilot.save_resume_state(
            tmp_path,
            {
                "protocol_sha256": "frozen-protocol",
                "epoch": 2,
                "weights": torch.tensor([2.0]),
            },
        )
    assert (tmp_path / "epoch-000002.pt").exists()
    assert (tmp_path / "resume.json").read_bytes() == manifest_before
    assert (tmp_path / "epoch-000001.pt").read_bytes() == checkpoint_before
    restored = pilot.load_resume_state(tmp_path, "frozen-protocol")
    assert restored["epoch"] == 1
    assert torch.equal(restored["weights"], first["weights"])


def _assert_identical(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_identical(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_identical(first, second)
    else:
        assert left == right


def test_synthetic_two_epoch_resume_matches_uninterrupted_training(tmp_path):
    torch.set_num_threads(1)
    config = pilot.AlexNetConfig(channels=(2, 2, 2, 2, 2), hidden=(4, 3))
    args = pilot.parser().parse_args(
        ["--epochs", "2", "--bins", "4", "--batch-size", "2"]
    )
    images = np.full((6, 784), 200, dtype=np.uint8)
    labels = np.arange(6, dtype=np.int64)
    validation = pilot.encode_inputs(images[:2], bins=4, seed=200000)
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    for output in (continuous, resumed):
        output.mkdir()
        pilot.write_json(
            output / "protocol.json", {"fixture": "same protocol"}
        )
    with torch.random.fork_rng(devices=[]):
        expected = pilot.train(
            args, continuous, config, images, labels, validation, labels[:2]
        )
        paused_args = deepcopy(args)
        paused_args.max_training_seconds = 1e-12
        paused = pilot.train(
            paused_args,
            resumed,
            config,
            images,
            labels,
            validation,
            labels[:2],
        )
        assert paused["epochs_run"] == 1
        assert paused["training_completed"] is False
        assert paused["stop_reason"] == "time_budget"
        actual = pilot.train(
            args, resumed, config, images, labels, validation, labels[:2]
        )
    assert actual["epochs_run"] == expected["epochs_run"] == 2
    assert (
        actual["training_completed"] is expected["training_completed"] is True
    )
    for field in ("selected_epoch", "selected_validation", "weight_audit"):
        _assert_identical(actual[field], expected[field])
    protocol_hash = pilot.sha256(continuous / "protocol.json")
    expected_state = pilot.load_resume_state(continuous, protocol_hash)
    actual_state = pilot.load_resume_state(resumed, protocol_hash)
    for state in (expected_state, actual_state):
        for epoch in state["history"]:
            epoch.pop("seconds")
    _assert_identical(actual_state, expected_state)
    expected_model = pilot.restore_selected(continuous, config, expected)
    actual_model = pilot.restore_selected(resumed, config, actual)
    _assert_identical(actual_model.state_dict(), expected_model.state_dict())


def _chunk_report(start):
    second = start == 2
    return {
        "samples": 2,
        "bins": 3,
        "source_batch_size": 2,
        "source_fingerprint": "same-source",
        "network_sha256": "same-network",
        "exact_spike_match_on_batch": False,
        "source_predictions": [0, 1] if not second else [1, 0],
        "lacuna_predictions": [0, 1] if not second else [0, 0],
        "source_output_counts": (
            [[2, 0], [0, 3]] if not second else [[0, 2], [1, 0]]
        ),
        "lacuna_output_counts": (
            [[2, 0], [0, 3]] if not second else [[1, 0], [1, 0]]
        ),
        "prediction_agreement": 1.0 if not second else 0.5,
        "off_grid_spikes": (
            [[0, 1, 3, 0.5]] if not second else [[1, 0, 2, 1.5]]
        ),
        "layers": [
            {
                "layer": 0,
                "source_block": 0,
                "shape": [4],
                "source_spikes": 3 if not second else 5,
                "lacuna_spikes": 3 if not second else 4,
                "mismatched_bins": 0 if not second else 1,
                "first_mismatch_sample_channel_bin": (
                    None if not second else [1, 3, 1]
                ),
            },
            {
                "layer": 1,
                "source_block": 2,
                "shape": [2],
                "source_spikes": 5 if not second else 3,
                "lacuna_spikes": 5 if not second else 2,
                "mismatched_bins": 2 if not second else 1,
                "first_mismatch_sample_channel_bin": (
                    [1, 1, 0] if not second else [0, 1, 1]
                ),
            },
        ],
        "tie_policy": "lowest output index wins",
    }


def test_chunk_comparison_aggregates_counts_and_global_mismatch_offsets(
    monkeypatch,
):
    reports = [_chunk_report(0), _chunk_report(2)]
    originals = deepcopy(reports)
    calls = []

    def validate(model, imported, inputs, *, source_batch_size):
        calls.append((inputs.flatten().tolist(), source_batch_size))
        return reports[len(calls) - 1]

    monkeypatch.setattr(pilot, "validate_slayer_feedforward", validate)
    result = pilot.compare_in_chunks(
        object(), object(), torch.arange(4), batch_size=2, chunk_size=2
    )
    assert calls == [([0, 1], 2), ([2, 3], 2)]
    assert result["samples"] == 4
    assert result["comparison_chunk_size"] == 2
    assert result["prediction_agreement"] == 0.75
    assert result["exact_spike_match_on_batch"] is False
    assert result["off_grid_spikes"] == [[0, 1, 3, 0.5], [3, 0, 2, 1.5]]
    for field in (
        "source_predictions",
        "lacuna_predictions",
        "source_output_counts",
        "lacuna_output_counts",
    ):
        assert result[field] == originals[0][field] + originals[1][field]
    assert result["layers"][0] == {
        "layer": 0,
        "source_block": 0,
        "shape": [4],
        "source_spikes": 8,
        "lacuna_spikes": 7,
        "mismatched_bins": 1,
        "first_mismatch_sample_channel_bin": [3, 3, 1],
    }
    assert result["layers"][1]["source_spikes"] == 8
    assert result["layers"][1]["lacuna_spikes"] == 7
    assert result["layers"][1]["mismatched_bins"] == 3
    assert result["layers"][1]["first_mismatch_sample_channel_bin"] == [
        1,
        1,
        0,
    ]
    assert reports == originals


@pytest.mark.parametrize("chunk_size", [0, 1, 3])
def test_chunk_comparison_rejects_misaligned_batch_boundaries(chunk_size):
    with pytest.raises(ValueError, match="multiple of batch size"):
        pilot.compare_in_chunks(
            object(),
            object(),
            torch.arange(4),
            batch_size=2,
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    "field", ["source_fingerprint", "network_sha256", "bins"]
)
def test_chunk_comparison_rejects_changed_identity(monkeypatch, field):
    def validate(model, imported, inputs, *, source_batch_size):
        report = _chunk_report(int(inputs[0]))
        if int(inputs[0]) == 2:
            report[field] = "changed"
        return report

    monkeypatch.setattr(pilot, "validate_slayer_feedforward", validate)
    with pytest.raises(ValueError, match="identity changed between chunks"):
        pilot.compare_in_chunks(
            object(), object(), torch.arange(4), batch_size=2, chunk_size=2
        )


def test_chunk_comparison_matches_single_real_source_validation(core):
    torch.set_num_threads(1)
    params = {
        "threshold": 1.0,
        "current_decay": 1.0,
        "voltage_decay": 512 / 4096,
        "scale": 4096,
        "persistent_state": False,
        "requires_grad": False,
    }
    model = torch.nn.Sequential(
        slayer.block.cuba.Dense(
            params,
            2,
            2,
            pre_hook_fx=None,
            weight_norm=False,
            delay=False,
            delay_shift=False,
        )
    ).eval()
    with torch.no_grad():
        model[0].synapse.weight.copy_(
            torch.tensor([[1.25, -0.25], [-0.5, 1.5]]).reshape_as(
                model[0].synapse.weight
            )
        )
    inputs = torch.tensor(
        [
            [[1, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [1, 0, 1]],
            [[1, 1, 0], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 1]],
        ],
        dtype=torch.float32,
    )
    imported = import_slayer_feedforward(
        model, input_shape=(2,), acknowledge_quantization=True
    )
    expected = validate_slayer_feedforward(
        model, imported, inputs, library=core._lib._name, source_batch_size=2
    )
    actual = pilot.compare_in_chunks(
        model, imported, inputs, batch_size=2, chunk_size=2
    )
    assert actual.pop("comparison_chunk_size") == 2
    assert actual == expected
