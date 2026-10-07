"""Check mixed signs across kinetic tiers, execution modes and inspection."""

from dataclasses import replace

import pytest

from lacuna import (
    AdEx,
    AdaptiveLIF,
    Engine,
    Graph,
    GraphEdge,
    InputMode,
    InputPort,
    LIF,
    NetworkBuilder,
    NeuronPolarity,
    OutputPort,
    SpikeInput,
)

from .test_per_edge import _alpha_graph, _graph


def _with_source_polarity(graph, polarity):
    return replace(
        graph,
        nodes=(replace(graph.nodes[0], polarity=polarity),) + graph.nodes[1:],
    )


def _assert_same_trajectory(actual, expected):
    assert actual.outputs == expected.outputs
    assert actual.core.spikes == expected.core.spikes
    assert actual.core.states == expected.core.states


@pytest.mark.parametrize("kinetic", ["exponential", "alpha"])
@pytest.mark.parametrize("taus", [(5.0, 5.0), (5.0, 7.0), (10.0, 10.0)])
@pytest.mark.parametrize("prepared", [False, True])
def test_signed_filtered_delivery_matches_typed_inhibition(
    core, kinetic, taus, prepared
):
    factory = _alpha_graph if kinetic == "alpha" else _graph
    signed = _with_source_polarity(
        factory(taus=taus, weights=(-7.0, -9.0)), NeuronPolarity.MIXED
    ).resolve()
    typed = _with_source_polarity(
        factory(taus=taus, weights=(7.0, 9.0)), NeuronPolarity.INHIBITORY
    ).resolve()
    inputs = (SpikeInput(1.0, "stimulus", 20.0),)
    if prepared:
        with signed.compile(core) as a, typed.compile(core) as b:
            actual = a.run(spike_inputs=inputs, t_end=6.0)
            expected = b.run(spike_inputs=inputs, t_end=6.0)
            assert a.last_execution_path == "compiled_sparse"
    else:
        actual = signed.run(core, spike_inputs=inputs, t_end=6.0)
        expected = typed.run(core, spike_inputs=inputs, t_end=6.0)
    _assert_same_trajectory(actual, expected)
    assert actual.core.states[1].values[0] < -65.0


@pytest.mark.parametrize(
    "target",
    [
        LIF(name="target_lif", drive=25.0),
        AdaptiveLIF(name="target_adaptive", drive=25.0),
        AdEx(name="target_adex", drive=500.0),
    ],
)
def test_signed_delta_preserves_analytical_and_stepped_targets(core, target):
    source = LIF(name="source")
    graph = Graph(
        models=(source.model, target.model),
        nodes=(
            source.node(0, polarity=NeuronPolarity.MIXED),
            target.node(1),
        ),
        edges=(GraphEdge(0, 0, 1, -5.0, 0.5),),
        input_ports=(InputPort("input", 0, InputMode.SPIKE),),
        output_ports=(OutputPort("target", 1),),
    )
    typed = _with_source_polarity(
        replace(graph, edges=(replace(graph.edges[0], weight=5.0),)),
        NeuronPolarity.INHIBITORY,
    )
    inputs = (SpikeInput(1.0, "input", 20.0),)
    with (
        graph.resolve().compile(core) as a,
        typed.resolve().compile(core) as b,
    ):
        actual = a.run(spike_inputs=inputs, t_end=20.0)
        expected = b.run(spike_inputs=inputs, t_end=20.0)
    _assert_same_trajectory(actual, expected)


def test_visualizer_keeps_mixed_signs_and_nonnegative_magnitudes(core):
    builder = NetworkBuilder("mixed-visualizer")
    source = builder.neuron("source", LIF(), polarity="mixed")
    targets = builder.population("targets", 2, LIF())
    builder.connect(source, targets, weight=(2.0, -3.0))
    with Engine(core._lib._name).compile(builder.build()) as compiled:
        with compiled.visualizer(1.0) as viewer:
            payload = viewer.graph_payload()
    assert payload["nodes"][0]["polarity"] == "MIXED"
    assert [edge["weight"] for edge in payload["edges"]] == [2.0, -3.0]
    assert [edge["magnitude"] for edge in payload["edges"]] == [2.0, 3.0]
