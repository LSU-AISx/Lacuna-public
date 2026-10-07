from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from lacuna import Engine, MixedInputSpike, Network, NeuronPolarity, SpikeTrain
from lacuna.errors import ResolutionError
from lacuna.importers import DenseLIFLayer, build_dense_lif


def test_dense_layer_normalizes_and_freezes_weights() -> None:
    weights = [[1, -2], [3, 4]]
    layer = DenseLIFLayer(weights, tau_m=2, threshold=1)
    weights[0][0] = 99
    assert layer.weights == ((1.0, -2.0), (3.0, 4.0))
    assert layer.tau_m == 2.0
    with pytest.raises(FrozenInstanceError):
        layer.threshold = 2.0


def test_dense_layer_accepts_array_like_sequences_without_numpy() -> None:
    class ArrayLike:
        def __init__(self, values):
            self.values = values

        def __len__(self):
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

    layer = DenseLIFLayer(
        ArrayLike([ArrayLike([1.0, -2.0]), ArrayLike([3.0, 4.0])]),
        tau_m=2,
        threshold=1,
    )
    assert layer.weights == ((1.0, -2.0), (3.0, 4.0))


def test_dense_edges_preserve_sign_and_output_input_orientation() -> None:
    deployment = build_dense_lif(
        (
            DenseLIFLayer(((1, -2), (3, 4), (-5, 6)), 4, 1, 0.25),
            DenseLIFLayer(((7, -8, 9),), 7, 2, 0.5),
        ),
        name="signed-import",
    )
    graph = deployment.network.graph
    assert deployment.layer_nodes == ((2, 3, 4), (5,))
    assert len(deployment.input_ports) == 2
    assert [(edge.pre, edge.post, edge.weight) for edge in graph.edges] == [
        (0, 2, 1.0),
        (0, 3, 3.0),
        (0, 4, -5.0),
        (1, 2, -2.0),
        (1, 3, 4.0),
        (1, 4, 6.0),
        (2, 5, 7.0),
        (3, 5, -8.0),
        (4, 5, 9.0),
    ]
    assert [edge.delay for edge in graph.edges] == [0.25] * 6 + [0.5] * 3
    assert all(node.polarity is NeuronPolarity.MIXED for node in graph.nodes)
    assert all(edge.plasticity is None for edge in graph.edges)
    assert all(edge.synapse is None for edge in graph.edges)


def test_dense_network_json_round_trip(tmp_path) -> None:
    deployment = build_dense_lif(
        [DenseLIFLayer([[0.5, -0.25], [-0.5, 0.75]], 2, 1)]
    )
    path = tmp_path / "dense-lif.json"
    deployment.network.save(path)
    loaded = Network.load(path)
    assert loaded.graph.to_text() == deployment.network.graph.to_text()
    assert loaded.semantic_sha256 == deployment.network.semantic_sha256


def test_dense_c_inference_aggregates_signed_inputs_and_resets(core) -> None:
    deployment = build_dense_lif(
        [
            DenseLIFLayer([[1.0, -0.5], [-0.5, 1.0]], 4, 1),
            DenseLIFLayer([[1.0, 1.0]], 7, 1, 0.25),
        ]
    )
    with Engine(core._lib._name).compile(deployment.network) as simulation:
        result = simulation.run(
            3.5,
            inputs={
                deployment.input_ports[0]: SpikeTrain((1.0, 2.0, 3.0)),
                deployment.input_ports[1]: SpikeTrain((2.0,)),
            },
        )
    learned = {node for layer in deployment.layer_nodes for node in layer}
    observed = [
        (spike.t, spike.node)
        for spike in result.spikes
        if spike.node in learned
    ]
    assert observed == [
        (1.0, 2),
        (1.25, 4),
        (3.0, 2),
        (3.25, 4),
    ]


def test_dense_c_binary_image_preserves_signed_network(core) -> None:
    deployment = build_dense_lif(
        [DenseLIFLayer([[1.0, -0.25]], 4, 1, 0.5)]
    )
    with Engine(core._lib._name).compile(deployment.network) as simulation:
        expected = simulation.run(
            3.0,
            inputs={
                deployment.input_ports[0]: SpikeTrain((1.0, 2.0)),
                deployment.input_ports[1]: SpikeTrain((1.0,)),
            },
        )
        image = simulation.compiled_graph_image()
    with core.load_compiled_graph_image(image) as loaded:
        actual = loaded.run(
            (0.0, 0.0, 0.0),
            inputs=(
                MixedInputSpike(1.0, 0, 1.0),
                MixedInputSpike(1.0, 1, 1.0),
                MixedInputSpike(2.0, 0, 1.0),
            ),
            t_end=3.0,
        )
    assert actual.spikes == expected.raw.core.spikes
    assert actual.states == expected.raw.core.states


@pytest.mark.parametrize(
    "weights",
    [
        (),
        ((),),
        ((1.0,), (1.0, 2.0)),
        (1.0,),
        "1",
        {"row": [1.0]},
        ((math.nan,),),
        ((math.inf,),),
        ((-math.inf,),),
        ((True,),),
        (("1",),),
        (((1.0,),),),
    ],
)
def test_dense_layer_rejects_invalid_weights(weights) -> None:
    with pytest.raises(ResolutionError):
        DenseLIFLayer(weights, 2, 1)


@pytest.mark.parametrize("field", ["tau_m", "threshold", "delay"])
@pytest.mark.parametrize(
    "value", [math.nan, math.inf, -math.inf, True, "1", [1]]
)
def test_dense_layer_rejects_nonfinite_and_nonnumeric_parameters(
    field, value
) -> None:
    parameters = {"tau_m": 2, "threshold": 1, "delay": 0}
    parameters[field] = value
    with pytest.raises(ResolutionError):
        DenseLIFLayer(((1,),), **parameters)


@pytest.mark.parametrize(
    "parameters",
    [
        {"tau_m": 0},
        {"tau_m": -1},
        {"threshold": 0},
        {"threshold": -1},
        {"delay": -1},
    ],
)
def test_dense_layer_rejects_invalid_parameter_ranges(parameters) -> None:
    values = {"tau_m": 2, "threshold": 1, "delay": 0}
    values.update(parameters)
    with pytest.raises(ResolutionError):
        DenseLIFLayer(((1,),), **values)


@pytest.mark.parametrize("layers", [(), [None], "layers"])
def test_dense_import_rejects_missing_or_invalid_layers(layers) -> None:
    with pytest.raises(ResolutionError):
        build_dense_lif(layers)


def test_dense_import_rejects_disconnected_layer_widths() -> None:
    with pytest.raises(ResolutionError, match="previous output width"):
        build_dense_lif(
            [
                DenseLIFLayer(((1, 2),), 2, 1),
                DenseLIFLayer(((1, 2),), 2, 1),
            ]
        )
