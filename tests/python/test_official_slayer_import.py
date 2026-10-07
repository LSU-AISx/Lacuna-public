"""Optional checks against the unmodified official SLAYER CPU implementation."""

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
slayer = pytest.importorskip("lava.lib.dl.slayer")

from lacuna.importers.slayer import (  # noqa: E402
    import_slayer_dense,
    validate_slayer_dense,
)


def _model(*weights, **neuron_overrides):
    parameters = {
        "threshold": 1.0,
        "current_decay": 1.0,
        "voltage_decay": 0.5,
        "persistent_state": False,
        "requires_grad": False,
    }
    parameters.update(neuron_overrides)
    layers = []
    for values in weights:
        matrix = torch.as_tensor(values, dtype=torch.float32)
        block = slayer.block.cuba.Dense(
            parameters,
            matrix.shape[1],
            matrix.shape[0],
            pre_hook_fx=None,
            weight_norm=False,
            delay=False,
            delay_shift=False,
        )
        with torch.no_grad():
            block.synapse.weight.copy_(matrix.reshape(block.synapse.weight.shape))
        layers.append(block)
    return torch.nn.Sequential(*layers).eval()


def _compare(model, inputs, core, *, timestep=1.0):
    imported = import_slayer_dense(
        model, timestep=timestep, acknowledge_quantization=True
    )
    report = validate_slayer_dense(
        model, imported, inputs, library=core._lib._name
    )
    return imported, report


def test_official_import_requires_rounding_acknowledgment():
    with pytest.raises(ValueError, match="acknowledge_quantization"):
        import_slayer_dense(_model([[1.25]]))


def test_official_import_uses_effective_decay_and_threshold():
    model = _model([[1.25, -0.5]], voltage_decay=0.12345, threshold=1.013)
    imported = import_slayer_dense(
        model, timestep=0.25, acknowledge_quantization=True
    )
    neuron = model[0].neuron
    layer = imported.metadata["layers"][0]
    decay = int(slayer.utils.quantize(neuron.voltage_decay).item())
    effective_threshold = torch.tensor(
        neuron.threshold + neuron.threshold_eps, dtype=torch.float32
    ).item()
    assert layer["decay_integer"] == decay
    assert layer["tau_m"] == pytest.approx(-0.25 / math.log1p(-decay / 4096))
    assert layer["threshold"] == effective_threshold
    assert layer["weights"] == [[1.25, -0.5]]
    assert imported.metadata["universal_exact_equivalence"] is False
    assert imported.metadata["source_files_sha256"]


def test_official_import_signed_inputs_and_simultaneous_cancellation(core):
    model = _model([[1.5, -1.5], [-1.5, 1.5]])
    inputs = torch.tensor([
        [[1, 0, 1, 0], [0, 0, 1, 0]],
        [[0, 0, 1, 0], [1, 0, 1, 0]],
    ], dtype=torch.float32)
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[1.0, 0.0], [0.0, 1.0]]
    assert report["lacuna_output_counts"] == report["source_output_counts"]
    assert report["source_predictions"] == [0, 1]
    assert report["prediction_agreement"] == 1.0
    assert report["off_grid_spikes"] == []


def test_official_import_decay_reset_and_independent_samples(core):
    model = _model([[0.75]])
    inputs = torch.tensor([
        [[1, 1, 1, 0, 1, 0, 1, 0]],
        [[1, 1, 1, 1, 1, 1, 1, 1]],
        [[0, 0, 0, 0, 0, 0, 0, 0]],
    ], dtype=torch.float32)
    _, report = _compare(model, inputs, core, timestep=0.25)
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[1.0], [4.0], [0.0]]
    assert report["layers"][0]["source_spikes"] == 5


@pytest.mark.parametrize("seed", [11, 29, 47])
def test_official_import_random_feedforward_all_layers(core, seed):
    generator = torch.Generator().manual_seed(seed)
    weights = [
        torch.randint(-12, 21, shape, generator=generator).float() / 16
        for shape in ((5, 4), (3, 5), (2, 3))
    ]
    model = _model(*weights, scale=1024)
    inputs = (torch.rand(6, 4, 24, generator=generator) < 0.3).float()
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"], report
    assert len(report["layers"]) == 3
    assert all(layer["mismatched_bins"] == 0 for layer in report["layers"])
    assert report["prediction_agreement"] == 1.0


def test_official_import_reports_known_quantization_disagreement(core):
    # This input rounds down to threshold in SLAYER but exceeds it in Lacuna.
    model = _model([[1.0001]])
    inputs = torch.ones(1, 1, 1)
    _, report = _compare(model, inputs, core)
    assert not report["exact_spike_match_on_batch"]
    assert report["layers"][0]["mismatched_bins"] == 1
    assert report["layers"][0]["first_mismatch_sample_channel_bin"] == [0, 0, 0]
    assert report["source_output_counts"] == [[0.0]]
    assert report["lacuna_output_counts"] == [[1.0]]
    # One output class agrees trivially even when its spike trains disagree.
    assert report["prediction_agreement"] == 1.0


@pytest.mark.parametrize("scale, spike_count", [(64, 0.0), (4096, 1.0)])
def test_official_import_matches_effective_threshold_equality(
    core, scale, spike_count
):
    model = _model([[1.0]], scale=scale)
    imported, report = _compare(model, torch.ones(1, 1, 1), core)
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[spike_count]]
    assert report["lacuna_output_counts"] == [[spike_count]]
    if scale == 4096:
        assert imported.metadata["layers"][0]["threshold"] == 1.0
    else:
        assert imported.metadata["layers"][0]["threshold"] > 1.0


@pytest.mark.parametrize("state", ["current_state", "voltage_state"])
@pytest.mark.parametrize("value", [0.25, float("nan"), float("inf")])
def test_official_import_rejects_invalid_initial_state(state, value):
    model = _model([[1.25]])
    getattr(model[0].neuron, state).fill_(value)
    with pytest.raises(ValueError, match="initial states must be zero"):
        import_slayer_dense(model, acknowledge_quantization=True)


@pytest.mark.parametrize("field, value", [
    ("groups", 2),
    ("stride", (1, 1, 2)),
    ("padding", (0, 0, 1)),
    ("dilation", (1, 1, 2)),
])
def test_official_import_rejects_modified_dense_geometry(field, value):
    model = _model([[1.25]])
    setattr(model[0].synapse, field, value)
    with pytest.raises(ValueError, match="convolution geometry"):
        import_slayer_dense(model, acknowledge_quantization=True)


@pytest.mark.parametrize("change, message", [
    ("train", "eval"),
    ("delay_shift", "delay"),
    ("quantized_weights", "pre_hook_fx"),
    ("bias", "bias"),
    ("dropout", "fixed shared neurons"),
    ("normalization", "fixed shared neurons"),
    ("mask", "masks"),
    ("count_log", "count_log"),
    ("axonal_delay", "delay"),
    ("persistent_state", "fixed shared neurons"),
    ("graded_spike", "fixed shared neurons"),
    ("requires_grad", "fixed shared neurons"),
    ("current_tail", "current_decay"),
    ("no_voltage_decay", "voltage decay"),
    ("full_voltage_decay", "voltage decay"),
    ("hook", "forward hooks"),
    ("double", "float32"),
])
def test_official_import_rejects_unsupported_source_features(change, message):
    model = _model([[1.25]])
    block = model[0]
    if change == "train":
        model.train()
    elif change == "delay_shift":
        block.delay_shift = True
    elif change == "quantized_weights":
        block.synapse.pre_hook_fx = block.neuron.quantize_8bit
    elif change == "bias":
        block.synapse.bias = torch.nn.Parameter(torch.zeros(1))
    elif change == "dropout":
        block.neuron.drop = torch.nn.Dropout(0.1).eval()
    elif change == "normalization":
        block.neuron.norm = torch.nn.Identity().eval()
    elif change == "mask":
        block.mask = torch.ones_like(block.synapse.weight)
    elif change == "count_log":
        block.count_log = True
    elif change == "axonal_delay":
        block.delay = slayer.axon.Delay(max_delay=2).eval()
    elif change in ("persistent_state", "graded_spike", "requires_grad"):
        setattr(block.neuron, change, True)
    elif change == "current_tail":
        with torch.no_grad():
            block.neuron.current_decay.fill_(2048)
    elif change in ("no_voltage_decay", "full_voltage_decay"):
        with torch.no_grad():
            block.neuron.voltage_decay.fill_(
                0 if change == "no_voltage_decay" else 4096
            )
    elif change == "hook":
        block.register_forward_hook(lambda module, args, output: output)
    elif change == "double":
        model.double()
    with pytest.raises(ValueError, match=message):
        import_slayer_dense(model, acknowledge_quantization=True)


def test_official_import_rejects_nonsequential_container():
    with pytest.raises(ValueError, match="Sequential"):
        import_slayer_dense(_model([[1.25]])[0], acknowledge_quantization=True)


def test_official_import_rejects_non_cuba_layer():
    model = torch.nn.Sequential(torch.nn.Linear(1, 1)).eval()
    with pytest.raises(ValueError, match="official CUBA Dense"):
        import_slayer_dense(model, acknowledge_quantization=True)


@pytest.mark.parametrize("changed_field", ["weight", "decay", "threshold"])
def test_official_validation_rejects_model_changed_after_export(changed_field):
    model = _model([[1.25]])
    imported = import_slayer_dense(model, acknowledge_quantization=True)
    with torch.no_grad():
        if changed_field == "weight":
            model[0].synapse.weight.add_(0.125)
        elif changed_field == "decay":
            model[0].neuron.voltage_decay.add_(1)
        else:
            model[0].neuron._threshold += 0.125
    with pytest.raises(ValueError, match="source model changed"):
        validate_slayer_dense(model, imported, torch.ones(1, 1, 2))


def test_official_validation_rejects_changed_deployment_bindings():
    model = _model([[1.25]])
    imported = import_slayer_dense(model, acknowledge_quantization=True)
    network = imported.deployment.network
    assert imported.network_sha256 == network.semantic_sha256
    network.graph.nodes[-1].bindings["v_threshold"] = 2.0
    assert imported.network_sha256 != network.semantic_sha256
    with pytest.raises(ValueError, match="deployment network changed"):
        validate_slayer_dense(model, imported, torch.ones(1, 1, 2))


@pytest.mark.parametrize("nested", [False, True])
def test_official_validation_rejects_changed_source_metadata(nested):
    model = _model([[1.25]])
    imported = import_slayer_dense(model, acknowledge_quantization=True)
    if nested:
        imported.metadata["layers"][0]["weights"][0][0] = 2.0
    else:
        imported.metadata["universal_exact_equivalence"] = True
    with pytest.raises(ValueError, match="source metadata changed"):
        validate_slayer_dense(model, imported, torch.ones(1, 1, 2))


@pytest.mark.parametrize("field, message", [
    ("weight", "source integer-state bound"),
    ("state_scale", "state representation"),
])
def test_official_import_rejects_unsafe_source_integer_state_bound(field, message):
    model = _model([[1.25]])
    if field == "weight":
        with torch.no_grad():
            model[0].synapse.weight.fill_(1e12)
    else:
        model[0].neuron.s_scale = 1 << 60
    with pytest.raises(ValueError, match=message):
        import_slayer_dense(model, acknowledge_quantization=True)


@pytest.mark.parametrize("inputs", [
    torch.ones(1, 1),
    torch.empty(0, 1, 2),
    torch.ones(1, 2, 2),
    torch.full((1, 1, 2), 0.5),
    torch.full((1, 1, 2), float("nan")),
])
def test_official_validation_rejects_invalid_input_contract(inputs):
    model = _model([[1.25]])
    imported = import_slayer_dense(model, acknowledge_quantization=True)
    with pytest.raises(ValueError, match="inputs"):
        validate_slayer_dense(model, imported, inputs)
