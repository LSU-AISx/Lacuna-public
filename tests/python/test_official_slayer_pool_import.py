"""Optional bounded Pool import checks against the real official SLAYER."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
slayer = pytest.importorskip("lava.lib.dl.slayer")

from lacuna import Engine, MixedInputSpike, Network, SpikeTrain  # noqa: E402
from lacuna.errors import ResolutionError  # noqa: E402
from lacuna.importers.slayer import (  # noqa: E402
    import_slayer_dense,
    import_slayer_feedforward,
    validate_slayer_feedforward,
)


def _parameters(**overrides):
    parameters = {
        "threshold": 1.0,
        "current_decay": 1.0,
        "voltage_decay": 0.5,
        "scale": 4096,
        "persistent_state": False,
        "requires_grad": False,
    }
    parameters.update(overrides)
    return parameters


def _pool(kernel, *, scale=4096, **geometry):
    return slayer.block.cuba.Pool(
        _parameters(scale=scale),
        kernel,
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
        **geometry,
    )


def _conv(inputs, outputs):
    return slayer.block.cuba.Conv(
        _parameters(),
        inputs,
        outputs,
        1,
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
    )


def _dense(inputs, outputs):
    return slayer.block.cuba.Dense(
        _parameters(),
        inputs,
        outputs,
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
    )


def _import(model, shape):
    return import_slayer_feedforward(
        model, input_shape=shape, acknowledge_quantization=True
    )


def _compare(model, inputs, core):
    imported = _import(model, tuple(inputs.shape[1:-1]))
    report = validate_slayer_feedforward(
        model, imported, inputs, library=core._lib._name
    )
    return imported, report


@pytest.mark.parametrize(
    "shape, kernel, weight_scale",
    [
        ((1, 4, 6), (2, 3), 1.0),
        ((2, 4, 6), (2, 3), 0.25),
        ((3, 6, 8), (3, 2), 1.5),
        ((4, 2, 6), (1, 3), -0.5),
    ],
)
def test_official_pool_graph_matches_independent_window_sums(
    shape, kernel, weight_scale
):
    model = torch.nn.Sequential(
        _pool(kernel, weight_scale=weight_scale)
    ).eval()
    imported = _import(model, shape)
    channels, rows, columns = shape
    kh, kw = kernel
    output_shape = (channels, rows // kh, columns // kw)
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    expected = values.reshape(channels, rows // kh, kh, columns // kw, kw)
    expected = expected.sum(dim=(2, 4)) * weight_scale
    source = model[0].synapse(values[None, ..., None]).detach()[0, ..., 0]
    assert torch.equal(source, expected)

    targets = imported.deployment.layer_nodes[0]
    target_index = {node: index for index, node in enumerate(targets)}
    actual = torch.zeros(math.prod(output_shape), dtype=torch.float64)
    graph = imported.deployment.network.graph
    for edge in graph.edges:
        output_index = target_index[edge.post]
        assert edge.pre // (rows * columns) == output_index // (
            output_shape[1] * output_shape[2]
        )
        actual[output_index] += (
            edge.weight * values.reshape(-1)[edge.pre].item()
        )
    assert len(graph.edges) == math.prod(shape)
    assert torch.equal(actual, expected.reshape(-1).double())
    assert imported.deployment.layer_shapes == (output_shape,)
    layer = imported.metadata["layers"][0]
    assert layer["kind"] == "pool2d"
    assert layer["lowered_kind"] == "conv2d"
    assert layer["groups"] == channels
    assert layer["stride"] == list(kernel)
    assert layer["padding"] == [0, 0]
    assert layer["dilation"] == [1, 1]
    assert layer["output_shape"] == list(output_shape)
    assert torch.equal(
        torch.tensor(layer["weights"]),
        torch.full((channels, 1, kh, kw), weight_scale),
    )
    assert imported.metadata["universal_exact_equivalence"] is False


def test_official_pool_copies_actual_asymmetric_signed_kernel():
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    kernel = torch.tensor([[0.25, -0.5, 0.75], [1.0, -1.25, 1.5]])
    with torch.no_grad():
        model[0].synapse.weight.copy_(kernel.reshape(1, 1, 2, 3, 1))
    imported = _import(model, (2, 4, 6))
    targets = imported.deployment.layer_nodes[0]
    actual = {
        (edge.pre, edge.post): edge.weight
        for edge in imported.deployment.network.graph.edges
    }
    expected = {}
    for channel in range(2):
        for row in range(2):
            for column in range(2):
                target = targets[(channel * 2 + row) * 2 + column]
                for kr in range(2):
                    for kc in range(3):
                        source = (
                            (channel * 4 + row * 2 + kr) * 6 + column * 3 + kc
                        )
                        expected[(source, target)] = kernel[kr, kc].item()
    assert actual == expected


def test_official_pool_analytic_spike_sequence_and_channel_isolation(core):
    model = torch.nn.Sequential(_pool((2, 3), weight_scale=0.25)).eval()
    inputs = torch.zeros(1, 2, 2, 6, 8)
    inputs[0, 0, 0, :3, [0, 1, 2, 4, 6]] = 1
    inputs[0, 0, :, 3:, [0, 3]] = 1
    inputs[0, 1, 0, 0, :] = 1
    inputs[0, 1, 0, 3:, :] = 1
    expected = torch.zeros(1, 2, 1, 2, 8)
    expected[0, 0, 0, 0, 1] = 1
    expected[0, 0, 0, 1, [0, 3]] = 1
    expected[0, 1, 0, 1, [1, 3, 5, 7]] = 1
    with torch.no_grad():
        assert torch.equal(model(inputs), expected)
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"], report
    assert report["source_output_counts"] == [[1.0, 2.0, 0.0, 4.0]]
    assert report["lacuna_output_counts"] == report["source_output_counts"]
    assert report["off_grid_spikes"] == []


def test_official_pool_flatten_dense_preserves_channel_row_column_order(core):
    model = torch.nn.Sequential(
        _pool((2, 3), weight_scale=1.25),
        slayer.block.cuba.Flatten(),
        _dense(12, 12),
    ).eval()
    with torch.no_grad():
        model[2].synapse.weight.copy_(
            (1.25 * torch.eye(12)).reshape_as(model[2].synapse.weight)
        )
    inputs = torch.zeros(12, 3, 4, 6, 1)
    for index in range(12):
        channel, position = divmod(index, 4)
        row, column = divmod(position, 2)
        inputs[index, channel, row * 2, column * 3, 0] = 1
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"], report
    assert report["source_predictions"] == list(range(12))
    assert report["lacuna_predictions"] == list(range(12))
    assert report["source_output_counts"] == torch.eye(12).tolist()


def test_official_conv_pool_conv_pool_flatten_dense_matches_all_layers(core):
    model = torch.nn.Sequential(
        _conv(2, 3),
        _pool((2, 3), weight_scale=0.25),
        _conv(3, 2),
        _pool(2, weight_scale=0.5),
        slayer.block.cuba.Flatten(),
        _dense(8, 3),
    ).eval()
    generator = torch.Generator().manual_seed(31)
    with torch.no_grad():
        for index in (0, 2, 5):
            weight = model[index].synapse.weight
            weight.copy_(
                torch.randint(-2, 6, weight.shape, generator=generator) / 4
            )
    inputs = (torch.rand(3, 2, 8, 12, 8, generator=generator) < 0.2).float()
    imported, report = _compare(model, inputs, core)
    assert imported.deployment.layer_shapes == (
        (3, 8, 12),
        (3, 4, 4),
        (2, 4, 4),
        (2, 2, 2),
        (3,),
    )
    assert imported.metadata["source_blocks"] == [0, 1, 2, 3, 5]
    assert [layer["kind"] for layer in imported.metadata["layers"]] == [
        "conv2d",
        "pool2d",
        "conv2d",
        "pool2d",
        "dense",
    ]
    assert len(report["layers"]) == 5
    assert report["exact_spike_match_on_batch"], report
    assert all(layer["mismatched_bins"] == 0 for layer in report["layers"])
    assert report["prediction_agreement"] == 1.0


def test_official_pool_retains_reported_near_threshold_disagreement(core):
    model = torch.nn.Sequential(_pool((2, 3), scale=64)).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(1.0001)
    inputs = torch.zeros(1, 1, 2, 3, 1)
    inputs[0, 0, 0, 0, 0] = 1
    _, report = _compare(model, inputs, core)
    assert not report["exact_spike_match_on_batch"]
    assert report["layers"][0]["mismatched_bins"] == 1
    assert report["layers"][0]["first_mismatch_sample_channel_bin"] == [
        0,
        0,
        0,
    ]
    assert report["source_output_counts"] == [[0.0]]
    assert report["lacuna_output_counts"] == [[1.0]]
    assert report["off_grid_spikes"] == []


def test_official_pool_json_roundtrip_and_binary_image_replay(core):
    model = torch.nn.Sequential(
        _pool((2, 3), weight_scale=0.75),
        slayer.block.cuba.Flatten(),
        _dense(2, 2),
    ).eval()
    with torch.no_grad():
        model[2].synapse.weight.copy_(
            torch.tensor([[1.25, -0.25], [-0.5, 1.5]]).reshape_as(
                model[2].synapse.weight
            )
        )
    imported = _import(model, (2, 2, 3))
    deployment = imported.deployment
    restored = Network.from_text(deployment.network.to_text())
    assert restored.to_text() == deployment.network.to_text()
    assert restored.semantic_sha256 == imported.network_sha256
    assert json.loads(json.dumps(imported.metadata)) == imported.metadata
    inputs = {
        deployment.input_ports[0]: SpikeTrain((1.0, 2.0)),
        deployment.input_ports[1]: SpikeTrain((1.0,)),
        deployment.input_ports[6]: SpikeTrain((2.0,)),
        deployment.input_ports[7]: SpikeTrain((2.0,)),
    }
    with Engine(core._lib._name).compile(deployment.network) as simulation:
        expected = simulation.run(3.0, inputs=inputs)
        image = simulation.compiled_graph_image()
    with Engine(core._lib._name).compile(restored) as simulation:
        roundtrip = simulation.run(3.0, inputs=inputs)
    assert roundtrip.raw.core.spikes == expected.raw.core.spikes
    assert roundtrip.raw.core.states == expected.raw.core.states
    with core.load_compiled_graph_image(image) as loaded:
        actual = loaded.run(
            (0.0,) * len(deployment.network.graph.nodes),
            inputs=(
                MixedInputSpike(1.0, 0, 1.0),
                MixedInputSpike(1.0, 1, 1.0),
                MixedInputSpike(2.0, 0, 1.0),
                MixedInputSpike(2.0, 6, 1.0),
                MixedInputSpike(2.0, 7, 1.0),
            ),
            t_end=3.0,
        )
    assert actual.spikes == expected.raw.core.spikes
    assert actual.states == expected.raw.core.states


@pytest.mark.parametrize(
    "field, value",
    [
        ("stride", (1, 3, 1)),
        ("stride", (3, 3, 1)),
        ("stride", (2, 3, 2)),
        ("padding", (1, 0, 0)),
        ("padding", (0, 0, 1)),
        ("dilation", (2, 1, 1)),
        ("dilation", (1, 1, 2)),
        ("kernel_size", (2, 3, 2)),
        ("kernel_size", (0, 3, 1)),
        ("kernel_size", (2, 2, 1)),
        ("groups", 2),
        ("in_channels", 2),
        ("out_channels", 2),
    ],
)
def test_official_pool_rejects_unsupported_or_inconsistent_geometry(
    field, value
):
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    setattr(model[0].synapse, field, value)
    with pytest.raises((ValueError, ResolutionError)):
        _import(model, (2, 4, 6))


@pytest.mark.parametrize(
    "shape", [(24,), (2, 3, 6), (2, 4, 5), (2, 1, 6), (2, 4, 2)]
)
def test_official_pool_rejects_flat_nondivisible_or_undersized_input(shape):
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    with pytest.raises((ValueError, ResolutionError)):
        _import(model, shape)


@pytest.mark.parametrize(
    "change",
    [
        "train",
        "trainable_kernel",
        "pre_hook",
        "weight_norm",
        "bias",
        "delay_shift",
        "delay",
        "count_log",
        "mask",
        "dropout",
        "norm",
        "persistent_state",
        "graded_spike",
        "requires_grad",
        "shared_param",
        "current_tail",
        "voltage_decay",
        "initial_state",
        "hook",
        "double",
        "nonfinite_weight",
    ],
)
def test_official_pool_rejects_unsupported_source_features(change):
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    block = model[0]
    if change == "train":
        model.train()
    elif change == "trainable_kernel":
        block.synapse.weight.requires_grad_(True)
    elif change == "pre_hook":
        block.synapse.pre_hook_fx = block.neuron.quantize_8bit
    elif change == "weight_norm":
        block.synapse.enable_weight_norm()
    elif change == "bias":
        block.synapse.bias = torch.nn.Parameter(torch.zeros(1))
    elif change == "delay_shift":
        block.delay_shift = True
    elif change == "delay":
        block.delay = slayer.axon.Delay(max_delay=2).eval()
    elif change == "count_log":
        block.count_log = True
    elif change == "mask":
        block.mask = torch.ones_like(block.synapse.weight)
    elif change == "dropout":
        block.neuron.drop = torch.nn.Dropout(0.1).eval()
    elif change == "norm":
        block.neuron.norm = torch.nn.Identity().eval()
    elif change in ("persistent_state", "graded_spike", "requires_grad"):
        setattr(block.neuron, change, True)
    elif change == "shared_param":
        block.neuron.shared_param = False
    elif change == "current_tail":
        block.neuron.current_decay.fill_(2048)
    elif change == "voltage_decay":
        block.neuron.voltage_decay.zero_()
    elif change == "initial_state":
        block.neuron.voltage_state.fill_(0.25)
    elif change == "hook":
        block.register_forward_hook(lambda module, args, output: output)
    elif change == "double":
        model.double()
    elif change == "nonfinite_weight":
        block.synapse.weight.fill_(float("nan"))
    with pytest.raises((ValueError, ResolutionError)):
        _import(model, (2, 4, 6))


def test_official_pool_requires_explicit_flatten_before_dense():
    model = torch.nn.Sequential(_pool((2, 3)), _dense(8, 2)).eval()
    with pytest.raises(ValueError, match="Flatten"):
        _import(model, (2, 4, 6))


def test_official_pool_is_not_accepted_by_dense_only_importer():
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    with pytest.raises(ValueError):
        import_slayer_dense(model, acknowledge_quantization=True)


@pytest.mark.parametrize("block_type", [torch.nn.MaxPool2d, torch.nn.AvgPool2d])
def test_official_pool_rejects_nonspiking_tensor_pooling(block_type):
    model = torch.nn.Sequential(block_type(2)).eval()
    with pytest.raises(ValueError):
        _import(model, (2, 4, 6))


def test_official_pool_rejects_custom_subclass():
    class CustomPool(slayer.block.cuba.Pool):
        pass

    block = CustomPool(
        _parameters(),
        (2, 3),
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
    )
    with pytest.raises(ValueError):
        _import(torch.nn.Sequential(block).eval(), (2, 4, 6))


def test_official_pool_rejects_warmed_incompatible_neuron_shape():
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    with torch.no_grad():
        model(torch.zeros(1, 2, 2, 12, 2))
    with pytest.raises(ValueError, match="neuron shape"):
        _import(model, (2, 4, 6))


@pytest.mark.parametrize("change", ["weight", "threshold", "source_file"])
def test_official_pool_validation_rejects_source_fingerprint_change(
    change, monkeypatch
):
    model = torch.nn.Sequential(_pool((2, 3))).eval()
    imported = _import(model, (2, 4, 6))
    if change == "weight":
        model[0].synapse.weight.add_(0.25)
    elif change == "threshold":
        model[0].neuron._threshold += 0.25
    else:
        original = Path.read_bytes

        def changed_source(path):
            data = original(path)
            if path.name == "layer.py" and path.parent.name == "synapse":
                return data + b"\n# Test-only source fingerprint change\n"
            return data

        monkeypatch.setattr(Path, "read_bytes", changed_source)
    with pytest.raises(ValueError, match="source model changed"):
        validate_slayer_feedforward(model, imported, torch.ones(1, 2, 4, 6, 2))
