from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from lacuna import Engine, Network, NeuronPolarity, SpikeTrain
from lacuna.errors import ResolutionError
from lacuna.importers.dense_lif import DenseLIFLayer
from lacuna.importers.feedforward_lif import (
    Conv2dLIFLayer,
    build_feedforward_lif,
    normalize_input_shape,
)


def projection(deployment, index=0):
    """Recover local edge coordinates without assuming edge insertion order."""

    if index:
        sources = deployment.layer_nodes[index - 1]
    else:
        sources = tuple(range(len(deployment.input_ports)))
    targets = deployment.layer_nodes[index]
    source_index = {node: local for local, node in enumerate(sources)}
    target_index = {node: local for local, node in enumerate(targets)}
    return {
        (source_index[edge.pre], target_index[edge.post]): edge.weight
        for edge in deployment.network.graph.edges
        if edge.pre in source_index and edge.post in target_index
    }


def test_conv_layer_normalizes_and_freezes_kernel():
    weights = [[[[1, -2], [3, 4]]]]
    layer = Conv2dLIFLayer(weights, 2, 1, stride=[2, 1])
    weights[0][0][0][0] = 9
    assert layer.weights == ((((1.0, -2.0), (3.0, 4.0)),),)
    assert layer.stride == (2, 1)
    assert layer.output_shape((1, 4, 3)) == (1, 2, 2)
    with pytest.raises(FrozenInstanceError):
        layer.threshold = 2


def test_conv_uses_cross_correlation_without_flipping_kernel():
    layer = Conv2dLIFLayer([[[[1, -2], [3, 4]]]], 2, 1, delay=0.25)
    deployment = build_feedforward_lif([layer], (1, 3, 3))
    assert deployment.input_shape == (1, 3, 3)
    assert deployment.layer_shapes == ((1, 2, 2),)
    assert projection(deployment) == {
        (0, 0): 1,
        (1, 0): -2,
        (3, 0): 3,
        (4, 0): 4,
        (1, 1): 1,
        (2, 1): -2,
        (4, 1): 3,
        (5, 1): 4,
        (3, 2): 1,
        (4, 2): -2,
        (6, 2): 3,
        (7, 2): 4,
        (4, 3): 1,
        (5, 3): -2,
        (7, 3): 3,
        (8, 3): 4,
    }
    graph = deployment.network.graph
    assert all(edge.delay == 0.25 for edge in graph.edges)
    assert all(node.polarity is NeuronPolarity.MIXED for node in graph.nodes)
    assert all(
        edge.plasticity is None and edge.weight_group is None for edge in graph.edges
    )
    assert all(edge.synapse is None for edge in graph.edges)


def test_conv_multiple_channels_follow_channel_major_order():
    layer = Conv2dLIFLayer([[[[1]], [[2]]], [[[3]], [[4]]]], 2, 1)
    deployment = build_feedforward_lif([layer], (2, 1, 2))
    assert deployment.layer_shapes == ((2, 1, 2),)
    assert projection(deployment) == {
        (0, 0): 1,
        (2, 0): 2,
        (1, 1): 1,
        (3, 1): 2,
        (0, 2): 3,
        (2, 2): 4,
        (1, 3): 3,
        (3, 3): 4,
    }


def test_conv_groups_map_only_within_matching_channel_group():
    layer = Conv2dLIFLayer(
        [[[[1]], [[2]]], [[[3]], [[4]]], [[[5]], [[6]]], [[[7]], [[8]]]],
        2,
        1,
        groups=2,
    )
    deployment = build_feedforward_lif([layer], (4, 1, 1))
    assert projection(deployment) == {
        (0, 0): 1,
        (1, 0): 2,
        (0, 1): 3,
        (1, 1): 4,
        (2, 2): 5,
        (3, 2): 6,
        (2, 3): 7,
        (3, 3): 8,
    }


def test_conv_stride_padding_and_dilation_omit_out_of_field_edges():
    layer = Conv2dLIFLayer(
        [[[[1, 2], [3, 4]]]],
        2,
        1,
        stride=(2, 2),
        padding=(1, 1),
        dilation=(2, 2),
    )
    deployment = build_feedforward_lif([layer], (1, 3, 3))
    assert layer.output_shape((1, 3, 3)) == (1, 2, 2)
    assert projection(deployment) == {
        (4, 0): 4,
        (4, 1): 3,
        (4, 2): 2,
        (4, 3): 1,
    }


def test_conv_nonsquare_geometry_and_floor_output_size():
    layer = Conv2dLIFLayer(
        [[[[1, 2, 3], [4, 5, 6]]]],
        2,
        1,
        stride=(2, 3),
        padding=(1, 0),
        dilation=(1, 2),
    )
    deployment = build_feedforward_lif([layer], (1, 4, 8))
    assert deployment.layer_shapes == ((1, 3, 2),)
    assert projection(deployment) == {
        (0, 0): 4,
        (2, 0): 5,
        (4, 0): 6,
        (3, 1): 4,
        (5, 1): 5,
        (7, 1): 6,
        (8, 2): 1,
        (10, 2): 2,
        (12, 2): 3,
        (16, 2): 4,
        (18, 2): 5,
        (20, 2): 6,
        (11, 3): 1,
        (13, 3): 2,
        (15, 3): 3,
        (19, 3): 4,
        (21, 3): 5,
        (23, 3): 6,
        (24, 4): 1,
        (26, 4): 2,
        (28, 4): 3,
        (27, 5): 1,
        (29, 5): 2,
        (31, 5): 3,
    }


def test_conv_zeros_do_not_create_inert_edges():
    deployment = build_feedforward_lif([Conv2dLIFLayer([[[[0, -1]]]], 2, 1)], (1, 1, 3))
    assert projection(deployment) == {(1, 0): -1, (2, 1): -1}
    all_zero = build_feedforward_lif(
        [Conv2dLIFLayer([[[[0]]]], 2, 1), DenseLIFLayer([[0, 0]], 2, 1)],
        (1, 1, 2),
    )
    assert all_zero.network.graph.edges == ()


def test_conv_conv_dense_flattens_all_spatial_channels():
    deployment = build_feedforward_lif(
        [
            Conv2dLIFLayer([[[[1]]], [[[2]]]], 2, 1),
            Conv2dLIFLayer([[[[3]], [[4]]]], 2, 1),
            DenseLIFLayer([[5, 6]], 2, 1),
        ],
        (1, 1, 2),
    )
    assert deployment.layer_shapes == ((2, 1, 2), (1, 1, 2), (1,))
    assert projection(deployment, 1) == {
        (0, 0): 3,
        (2, 0): 4,
        (1, 1): 3,
        (3, 1): 4,
    }
    assert projection(deployment, 2) == {(0, 0): 5, (1, 0): 6}
    flatten = build_feedforward_lif(
        [Conv2dLIFLayer([[[[1]]], [[[2]]]], 2, 1), DenseLIFLayer([[3, 4, 5, 6]], 2, 1)],
        (1, 1, 2),
    )
    assert projection(flatten, 1) == {(0, 0): 3, (1, 0): 4, (2, 0): 5, (3, 0): 6}


def test_dense_vector_input_is_supported():
    deployment = build_feedforward_lif([DenseLIFLayer([[2, -3]], 2, 1)], (2,))
    assert deployment.input_shape == (2,)
    assert deployment.layer_shapes == ((1,),)
    assert projection(deployment) == {(0, 0): 2, (1, 0): -3}


def test_spatial_graph_json_round_trip(tmp_path):
    deployment = build_feedforward_lif([Conv2dLIFLayer([[[[1, -2]]]], 2, 1)], (1, 2, 3))
    path = tmp_path / "conv.json"
    deployment.network.save(path)
    loaded = Network.load(path)
    assert loaded.graph.to_text() == deployment.network.graph.to_text()
    assert loaded.semantic_sha256 == deployment.network.semantic_sha256


def test_conv_dense_execute_as_ordinary_c_graph(core):
    deployment = build_feedforward_lif(
        [
            Conv2dLIFLayer([[[[1, -1]]]], 2, 1, delay=0.25),
            DenseLIFLayer([[1, 1]], 2, 1, delay=0.5),
        ],
        (1, 1, 3),
    )
    with Engine(core._lib._name).compile(deployment.network) as simulation:
        result = simulation.run(
            2.0,
            inputs={deployment.input_ports[0]: SpikeTrain((1.0,))},
        )
    assert [
        (spike.t, spike.node) for spike in result.spikes if spike.node in (3, 4, 5)
    ] == [(1.25, 3), (1.75, 5)]


@pytest.mark.parametrize(
    "weights",
    [
        (),
        ((),),
        (((),),),
        ((((),),),),
        [1],
        [[1]],
        [[[1]]],
        [[[[1]], [[1, 2]]]],
        [[[[1]], [[1], [2]]]],
        [[[[1]]], [[[1]], [[2]]]],
        [[[[math.nan]]]],
        [[[[math.inf]]]],
        [[[[True]]]],
        [[[["1"]]]],
        {"weights": 1},
    ],
)
def test_conv_rejects_empty_ragged_or_invalid_weights(weights):
    with pytest.raises(ResolutionError):
        Conv2dLIFLayer(weights, 2, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tau_m", 0),
        ("tau_m", math.nan),
        ("tau_m", True),
        ("threshold", 0),
        ("threshold", math.inf),
        ("threshold", "1"),
        ("delay", -1),
        ("delay", math.nan),
        ("stride", (0, 1)),
        ("stride", (1.0, 1)),
        ("stride", (True, 1)),
        ("stride", 1),
        ("stride", (1,)),
        ("padding", (-1, 0)),
        ("padding", (0, 0.0)),
        ("dilation", (1, 0)),
        ("dilation", (1, math.inf)),
        ("groups", 0),
        ("groups", 1.0),
        ("groups", True),
        ("groups", 2),
    ],
)
def test_conv_rejects_invalid_parameters(field, value):
    parameters = {"tau_m": 2, "threshold": 1}
    parameters[field] = value
    with pytest.raises(ResolutionError):
        Conv2dLIFLayer([[[[1]]]], **parameters)


@pytest.mark.parametrize(
    "shape",
    [(), (0,), (-1, 2, 3), (1, 2), (1, 2, 3, 4), (1, True, 2), (1, 2.0, 3), "123"],
)
def test_invalid_input_shapes_are_rejected(shape):
    with pytest.raises(ResolutionError):
        normalize_input_shape(shape)


@pytest.mark.parametrize(
    "layers,shape",
    [
        ([], (1, 1, 1)),
        ([None], (1, 1, 1)),
        ([Conv2dLIFLayer([[[[1]]]], 2, 1)], (2, 1, 1)),
        ([Conv2dLIFLayer([[[[1, 2]]]], 2, 1)], (1, 1, 1)),
        ([DenseLIFLayer([[1, 2]], 2, 1)], (1, 2, 2)),
        ([DenseLIFLayer([[1]], 2, 1), Conv2dLIFLayer([[[[1]]]], 2, 1)], (1, 1, 1)),
    ],
)
def test_invalid_layer_chains_are_rejected(layers, shape):
    with pytest.raises(ResolutionError):
        build_feedforward_lif(layers, shape)
