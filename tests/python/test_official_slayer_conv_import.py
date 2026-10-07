"""Optional spatial importer checks against the official SLAYER implementation."""

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")
slayer = pytest.importorskip("lava.lib.dl.slayer")

from lacuna import Engine, MixedInputSpike, SpikeTrain  # noqa: E402
from lacuna.errors import ResolutionError  # noqa: E402
from lacuna.importers.slayer import (  # noqa: E402
    import_slayer_feedforward,
    validate_slayer_feedforward,
)


def _parameters():
    return {
        "threshold": 1.0,
        "current_decay": 1.0,
        "voltage_decay": 0.5,
        "scale": 4096,
        "persistent_state": False,
        "requires_grad": False,
    }


def _conv(in_channels, out_channels, kernel, **geometry):
    return slayer.block.cuba.Conv(
        _parameters(),
        in_channels,
        out_channels,
        kernel,
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
        **geometry,
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


def _fill_dyadic_weights(model, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in model:
            if hasattr(block, "synapse"):
                weights = block.synapse.weight
                values = (
                    torch.randint(-4, 7, weights.shape, generator=generator).float() / 4
                )
                weights.copy_(values)


@pytest.mark.parametrize(
    "shape, outputs, kernel, geometry",
    [
        ((1, 4, 5), 2, (2, 3), {}),
        ((2, 5, 7), 3, (2, 3), {"stride": (2, 1), "padding": (1, 2)}),
        ((2, 6, 7), 4, (2, 2), {"dilation": (2, 1), "padding": (1, 0)}),
        (
            (4, 7, 9),
            6,
            (2, 3),
            {"stride": (2, 1), "padding": (1, 2), "dilation": (2, 1), "groups": 2},
        ),
        ((3, 4, 5), 3, (2, 2), {"groups": 3}),
    ],
)
def test_official_conv_graph_deposits_match_source_cross_correlation(
    shape, outputs, kernel, geometry
):
    model = torch.nn.Sequential(_conv(shape[0], outputs, kernel, **geometry)).eval()
    synapse = model[0].synapse
    # Distinct coefficients expose channel permutations and flipped kernels.
    with torch.no_grad():
        values = torch.arange(synapse.weight.numel()).reshape_as(synapse.weight)
        synapse.weight.copy_((values % 17 - 8).float() / 4)
    imported = _import(model, shape)
    graph = imported.deployment.network.graph
    targets = imported.deployment.layer_nodes[0]
    target_index = {node: index for index, node in enumerate(targets)}
    matrix = torch.zeros(len(targets), math.prod(shape), dtype=torch.float64)
    for edge in graph.edges:
        matrix[target_index[edge.post], edge.pre] += edge.weight
    values = torch.arange(math.prod(shape), dtype=torch.float32).reshape(1, *shape, 1)
    expected = synapse(values).detach().reshape(-1).double()
    actual = matrix @ values.reshape(-1).double()
    assert torch.equal(actual, expected)
    assert imported.deployment.input_shape == shape
    assert imported.deployment.layer_shapes[0] == tuple(synapse(values).shape[1:-1])


def test_official_conv_asymmetric_impulse_has_no_kernel_flip(core):
    model = torch.nn.Sequential(_conv(1, 1, (2, 3))).eval()
    with torch.no_grad():
        model[0].synapse.weight.zero_()
        model[0].synapse.weight[0, 0, 0, 2, 0] = 1.25
    inputs = torch.zeros(1, 1, 3, 5, 3)
    inputs[0, 0, 0, 2, 0] = 1
    inputs[0, 0, 1, 4, 1] = 1
    imported, report = _compare(model, inputs, core)
    assert imported.deployment.layer_shapes == ((1, 2, 3),)
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_official_conv_grouped_layer_spikes_match(core, groups):
    model = torch.nn.Sequential(
        _conv(4, 4, (2, 3), stride=(2, 1), padding=(1, 0), groups=groups)
    ).eval()
    _fill_dyadic_weights(model, 7)
    generator = torch.Generator().manual_seed(19)
    inputs = (torch.rand(3, 4, 5, 6, 8, generator=generator) < 0.3).float()
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"], report
    assert report["off_grid_spikes"] == []


def test_official_conv_depthwise_channels_do_not_cross(core):
    model = torch.nn.Sequential(_conv(3, 3, 1, groups=3)).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(1.25)
    inputs = torch.zeros(1, 3, 2, 2, 2)
    inputs[0, 1, 0, 1, 0] = 1
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[0.0] * 5 + [1.0] + [0.0] * 6]


def test_official_two_convolutions_flatten_and_dense_compare_neural_layers(core):
    model = torch.nn.Sequential(
        _conv(2, 3, (2, 3), padding=(1, 0)),
        _conv(3, 2, (2, 2), stride=(2, 1)),
        slayer.block.cuba.Flatten(),
        _dense(2 * 3 * 4, 4),
    ).eval()
    _fill_dyadic_weights(model, 11)
    generator = torch.Generator().manual_seed(29)
    inputs = (torch.rand(3, 2, 6, 7, 8, generator=generator) < 0.2).float()
    imported, report = _compare(model, inputs, core)
    assert imported.deployment.layer_shapes == ((3, 7, 5), (2, 3, 4), (4,))
    assert len(imported.deployment.layer_nodes) == 3
    assert len(report["layers"]) == 3
    assert report["exact_spike_match_on_batch"], report
    assert report["prediction_agreement"] == 1.0


def test_official_flatten_preserves_channel_row_column_order(core):
    model = torch.nn.Sequential(
        _conv(2, 2, 1, groups=2),
        slayer.block.cuba.Flatten(),
        _dense(12, 12),
    ).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(1.25)
        model[2].synapse.weight.copy_(
            (1.25 * torch.eye(12)).reshape_as(model[2].synapse.weight)
        )
    inputs = torch.eye(12).reshape(12, 2, 2, 3, 1)
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"]
    assert report["source_predictions"] == list(range(12))
    assert report["lacuna_predictions"] == list(range(12))


def test_official_feedforward_accepts_flat_dense_input(core):
    model = torch.nn.Sequential(_dense(2, 2)).eval()
    with torch.no_grad():
        model[0].synapse.weight.copy_(
            (1.25 * torch.eye(2)).reshape_as(model[0].synapse.weight)
        )
    _, report = _compare(model, torch.eye(2).reshape(2, 2, 1), core)
    assert report["exact_spike_match_on_batch"]
    assert report["source_predictions"] == [0, 1]


def test_official_conv_validation_chunks_preserve_independent_samples(core):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(0.75)
    inputs = torch.ones(65, 1, 1, 1, 4)
    inputs[-1].zero_()
    _, report = _compare(model, inputs, core)
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[2.0]] * 64 + [[0.0]]


def test_official_conv_import_accepts_warmed_matching_geometry(core):
    model = torch.nn.Sequential(_conv(1, 1, (2, 3))).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(0.25)
        model(torch.zeros(2, 1, 4, 5, 3))
    _, report = _compare(model, torch.ones(1, 1, 4, 5, 3), core)
    assert report["exact_spike_match_on_batch"]


def test_official_conv_import_rejects_warmed_incompatible_geometry():
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 3, 2, 2))
    with pytest.raises(ValueError, match="neuron shape"):
        _import(model, (1, 2, 3))


def test_official_conv_reports_source_state_rounding_disagreement(core):
    parameters = _parameters()
    parameters["scale"] = 64
    model = torch.nn.Sequential(
        slayer.block.cuba.Conv(
            parameters,
            1,
            1,
            1,
            pre_hook_fx=None,
            weight_norm=False,
            delay=False,
            delay_shift=False,
        )
    ).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(1.0001)
    _, report = _compare(model, torch.ones(1, 1, 1, 1, 1), core)
    assert not report["exact_spike_match_on_batch"]
    assert report["layers"][0]["mismatched_bins"] == 1
    assert report["layers"][0]["first_mismatch_sample_channel_bin"] == [0, 0, 0]
    assert report["source_output_counts"] == [[0.0]]
    assert report["lacuna_output_counts"] == [[1.0]]
    assert report["off_grid_spikes"] == []


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "2"])
def test_official_conv_validation_rejects_invalid_source_batch_size(batch_size):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    imported = _import(model, (1, 1, 1))
    with pytest.raises(ValueError, match="source_batch_size"):
        validate_slayer_feedforward(
            model,
            imported,
            torch.ones(1, 1, 1, 1, 1),
            source_batch_size=batch_size,
        )


@pytest.mark.parametrize("batch_size", [1, 3])
def test_official_conv_validation_records_source_batch_size(core, batch_size):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(1.25)
    imported = _import(model, (1, 1, 1))
    report = validate_slayer_feedforward(
        model,
        imported,
        torch.ones(5, 1, 1, 1, 2),
        library=core._lib._name,
        source_batch_size=batch_size,
    )
    assert report["source_batch_size"] == batch_size
    assert report["exact_spike_match_on_batch"]
    assert report["source_output_counts"] == [[2.0]] * 5


def test_official_conv_dense_binary_image_replays_signed_network(core):
    model = torch.nn.Sequential(
        _conv(1, 2, (1, 2)),
        slayer.block.cuba.Flatten(),
        _dense(4, 2),
    ).eval()
    with torch.no_grad():
        model[0].synapse.weight.copy_(
            torch.tensor([[1.5, -0.5], [0.25, 1.25]]).reshape_as(
                model[0].synapse.weight
            )
        )
        model[2].synapse.weight.copy_(
            torch.tensor([[1.25, 0.5, -0.25, 0.0], [0.0, -0.25, 0.5, 1.25]]).reshape_as(
                model[2].synapse.weight
            )
        )
    deployment = _import(model, (1, 1, 3)).deployment
    with Engine(core._lib._name).compile(deployment.network) as simulation:
        expected = simulation.run(
            3.0,
            inputs={
                deployment.input_ports[0]: SpikeTrain((1.0, 2.0)),
                deployment.input_ports[1]: SpikeTrain((1.0,)),
                deployment.input_ports[2]: SpikeTrain((2.0,)),
            },
        )
        image = simulation.compiled_graph_image()
    with core.load_compiled_graph_image(image) as loaded:
        actual = loaded.run(
            (0.0,) * len(deployment.network.graph.nodes),
            inputs=(
                MixedInputSpike(1.0, 0, 1.0),
                MixedInputSpike(1.0, 1, 1.0),
                MixedInputSpike(2.0, 0, 1.0),
                MixedInputSpike(2.0, 2, 1.0),
            ),
            t_end=3.0,
        )
    assert actual.spikes == expected.raw.core.spikes
    assert actual.states == expected.raw.core.states


@pytest.mark.parametrize(
    "field, value",
    [
        ("stride", (1, 1, 2)),
        ("padding", (0, 0, 1)),
        ("dilation", (1, 1, 2)),
        ("kernel_size", (1, 1, 2)),
    ],
)
def test_official_conv_rejects_temporal_geometry(field, value):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    setattr(model[0].synapse, field, value)
    with pytest.raises(ValueError):
        _import(model, (1, 2, 3))


@pytest.mark.parametrize(
    "shape", [(1,), (2, 3), (2, 2, 3), (1, 0, 3), (1, 2, -3), (1, True, 3)]
)
def test_official_conv_rejects_invalid_or_mismatched_input_shape(shape):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    with pytest.raises((ValueError, ResolutionError)):
        _import(model, shape)


@pytest.mark.parametrize(
    "inputs",
    [
        torch.ones(1, 6, 2),
        torch.ones(1, 1, 3, 2, 2),
        torch.ones(0, 1, 2, 3, 2),
        torch.full((1, 1, 2, 3, 2), 0.5),
        torch.full((1, 1, 2, 3, 2), float("nan")),
    ],
)
def test_official_conv_validation_rejects_invalid_inputs(inputs):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    imported = _import(model, (1, 2, 3))
    with pytest.raises(ValueError, match="inputs"):
        validate_slayer_feedforward(model, imported, inputs)


@pytest.mark.parametrize("change", ["stride", "padding", "dilation", "weights"])
def test_official_conv_validation_rejects_source_mutation(change):
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    imported = _import(model, (1, 4, 5))
    synapse = model[0].synapse
    if change == "weights":
        with torch.no_grad():
            synapse.weight.add_(0.25)
    elif change == "padding":
        synapse.padding = (1, 1, 0)
    else:
        setattr(synapse, change, (2, 2, 1))
    with pytest.raises(ValueError):
        validate_slayer_feedforward(model, imported, torch.ones(1, 1, 4, 5, 2))


@pytest.mark.parametrize(
    "unsupported", ["torch_flatten", "pool_defaults", "conv_transpose"]
)
def test_official_conv_rejects_unsupported_block_configurations(unsupported):
    if unsupported == "torch_flatten":
        block = torch.nn.Flatten(1, 3)
    elif unsupported == "pool_defaults":
        block = slayer.block.cuba.Pool(_parameters(), 2)
    else:
        block = slayer.block.cuba.ConvT(_parameters(), 1, 1, 2)
    model = torch.nn.Sequential(block).eval()
    with pytest.raises(ValueError):
        _import(model, (1, 4, 5))


@pytest.mark.parametrize(
    "sequence", ["flatten_twice", "dense_without_flatten", "conv_after_flatten"]
)
def test_official_conv_rejects_invalid_spatial_flat_transitions(sequence):
    if sequence == "flatten_twice":
        blocks = [
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Flatten(),
            _dense(6, 1),
        ]
    elif sequence == "dense_without_flatten":
        blocks = [_conv(1, 1, 1), _dense(6, 1)]
    else:
        blocks = [slayer.block.cuba.Flatten(), _conv(6, 1, 1)]
    with pytest.raises(ValueError):
        _import(torch.nn.Sequential(*blocks).eval(), (1, 2, 3))


def test_official_conv_import_requires_quantization_acknowledgment():
    model = torch.nn.Sequential(_conv(1, 1, 1)).eval()
    with pytest.raises(ValueError, match="acknowledge_quantization"):
        import_slayer_feedforward(model, input_shape=(1, 2, 3))
