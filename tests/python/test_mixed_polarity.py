"""Signed weights are opt-in and do not weaken Dale-typed connections."""

from dataclasses import replace
import json
import math

import pytest

from lacuna import (
    AdEx,
    AdaptiveLIF,
    AlphaCurrent,
    Delta,
    DeltaEdge,
    Engine,
    ExponentialCurrent,
    Graph,
    InputSpike,
    LIF,
    MixedEdge,
    MixedInputSpike,
    ModulatedSTDP,
    NetworkBuilder,
    NeuronPolarity,
    PairSTDP,
    RecordingConfig,
    SoftExcursionModulated,
    SpikeTrain,
    TripletSTDP,
    VoltageModulatedSTDP,
)
from lacuna.audit import audit_causal_trace
from lacuna.errors import ResolutionError

from .test_compiled_graph import _reactive


def _network(
    polarity=NeuronPolarity.MIXED, weight=-4.0, model=None, synapse=None
):
    builder = NetworkBuilder("mixed-weights")
    source = builder.neuron(
        "source", LIF(name="source_lif"), polarity=polarity
    )
    target = builder.neuron("target", model or LIF())
    builder.connect(
        source,
        target,
        weight=weight,
        delay=0.3,
        synapse=synapse or Delta(),
    )
    builder.input("in", source)
    builder.output("out", target)
    return builder.build()


def test_polarity_mapping_and_default():
    assert [polarity.runtime_code for polarity in NeuronPolarity] == [0, 1, 2]
    assert [polarity.sign for polarity in NeuronPolarity] == [1.0, -1.0, 1.0]
    assert LIF().node(0).polarity is NeuronPolarity.EXCITATORY
    builder = NetworkBuilder("defaults")
    builder.neuron("default", LIF())
    builder.neuron("mixed", LIF(), polarity="mixed")
    graph = builder.build().graph
    assert graph.nodes[0].polarity is NeuronPolarity.EXCITATORY
    assert graph.nodes[1].polarity is NeuronPolarity.MIXED


def test_signed_graph_round_trip_and_schema_compatibility():
    graph = _network().graph
    text = graph.to_text()
    document = json.loads(text)
    assert document["schema"] == 11
    restored = Graph.from_text(text)
    assert restored.to_text() == text
    assert restored.nodes[0].polarity is NeuronPolarity.MIXED
    assert restored.edges[0].weight == -4.0
    assert restored.resolve().effective_edges[0].weight == -4.0
    for schema in (9, 10):
        document["schema"] = schema
        with pytest.raises(ResolutionError, match="schema 11"):
            Graph.from_text(json.dumps(document))
    typed = json.loads(
        _network(NeuronPolarity.EXCITATORY, 4.0).graph.to_text()
    )
    assert typed["schema"] == 10
    for schema in (9, 10, 11):
        typed["schema"] = schema
        assert (
            Graph.from_text(json.dumps(typed)).nodes[0].polarity
            is NeuronPolarity.EXCITATORY
        )
    del typed["nodes"][0]["polarity"]
    with pytest.raises(ResolutionError):
        Graph.from_text(json.dumps(typed))


@pytest.mark.parametrize(
    "polarity", [NeuronPolarity.EXCITATORY, NeuronPolarity.INHIBITORY]
)
def test_typed_sources_still_reject_negative_weights(core, polarity):
    with pytest.raises(ResolutionError, match="nonnegative"):
        _network(polarity, -4.0)
    graph = _network().graph
    graph = replace(
        graph,
        nodes=(replace(graph.nodes[0], polarity=polarity), graph.nodes[1]),
    )
    with pytest.raises(ResolutionError, match="nonnegative"):
        graph.resolve()
    with pytest.raises(ValueError):
        core.compile_mixed(
            (_reactive(),) * 2,
            edges=(MixedEdge(0, 1, -4.0, 0.3),),
            polarities=(polarity, NeuronPolarity.EXCITATORY),
        )
    with pytest.raises(ValueError):
        core.run_delta(
            (_reactive(),) * 2,
            (-65.0,) * 2,
            edges=(DeltaEdge(0, 1, -4.0, 0.3),),
            polarities=(polarity, NeuronPolarity.EXCITATORY),
            t_end=2.0,
        )


@pytest.mark.parametrize("weight", [math.nan, math.inf, -math.inf])
def test_mixed_weight_must_be_finite(weight):
    with pytest.raises(ResolutionError):
        _network(weight=weight)


@pytest.mark.parametrize(
    "model,synapse",
    [
        (LIF(), Delta()),
        (AdaptiveLIF(), Delta()),
        (AdEx(), Delta()),
        (LIF(synaptic_input=True), ExponentialCurrent(5.0)),
        (LIF(synaptic_input=True), AlphaCurrent(5.0)),
        (LIF(synaptic_input=True), AlphaCurrent(20.0)),
    ],
)
def test_signed_negative_delivery_matches_inhibitory_magnitude(
    core, model, synapse
):
    engine = Engine(core._lib._name)
    results = []
    for polarity, weight in (
        (NeuronPolarity.MIXED, -4.0),
        (NeuronPolarity.INHIBITORY, 4.0),
    ):
        with engine.compile(
            _network(polarity, weight, model, synapse)
        ) as compiled:
            results.append(
                compiled.run(8.0, inputs={"in": SpikeTrain((0.2, 4.0), 20.0)})
            )
    assert results[0].raw.core.spikes == results[1].raw.core.spikes
    assert results[0].raw.core.states == results[1].raw.core.states


def test_all_polarities_share_one_graph_and_trace_audit(core):
    models = (_reactive(),) * 5
    polarities = (
        NeuronPolarity.MIXED,
        NeuronPolarity.EXCITATORY,
        NeuronPolarity.INHIBITORY,
        NeuronPolarity.EXCITATORY,
        NeuronPolarity.EXCITATORY,
    )
    edges = (
        MixedEdge(0, 1, 20.0, 0.1),
        MixedEdge(0, 2, 20.0, 0.1),
        MixedEdge(0, 3, -4.0, 0.1),
        MixedEdge(1, 4, 3.0, 0.2),
        MixedEdge(2, 4, 3.0, 0.2),
    )
    result = core.run_mixed(
        models,
        (-65.0,) * 5,
        edges=edges,
        polarities=polarities,
        inputs=(MixedInputSpike(0.2, 0, 20.0),),
        recording=RecordingConfig(capacity=300),
        t_end=2.0,
    )
    assert [spike.node for spike in result.spikes] == [0, 1, 2]
    assert result.states[3].values[0] < -65.0
    assert result.states[4].values[0] == pytest.approx(-65.0)
    audit = audit_causal_trace(
        result, models=models, edges=edges, polarities=polarities, t_end=2.0
    )
    assert audit.delivery_records == 5
    scalar = core.run_delta(
        models,
        (-65.0,) * 5,
        edges=tuple(
            DeltaEdge(e.pre, e.post, e.weight, e.delay) for e in edges
        ),
        polarities=polarities,
        inputs=(InputSpike(0.2, 0, 20.0),),
        t_end=2.0,
    )
    assert scalar.spikes == result.spikes
    assert tuple(state.value for state in scalar.states) == tuple(
        state.values[0] for state in result.states
    )


@pytest.mark.parametrize(
    "rule",
    [
        PairSTDP(),
        TripletSTDP(),
        ModulatedSTDP(),
        VoltageModulatedSTDP(),
        SoftExcursionModulated(),
    ],
)
def test_mixed_online_plasticity_rejected_at_api_and_native_binding(
    core, rule
):
    graph = _network(weight=0.5).graph
    graph = replace(graph, edges=(replace(graph.edges[0], plasticity=rule),))
    with pytest.raises(ResolutionError, match="online plasticity.*MIXED"):
        graph.resolve()
    builder = NetworkBuilder("mixed-learning")
    source = builder.neuron("source", LIF(), polarity=NeuronPolarity.MIXED)
    target = builder.neuron("target", LIF())
    with pytest.raises(ResolutionError, match="online plasticity.*MIXED"):
        builder.connect(source, target, weight=0.5, plasticity=rule)
    with pytest.raises(ValueError, match="online plasticity.*MIXED"):
        core.compile_mixed(
            (_reactive(),) * 2,
            edges=(MixedEdge(0, 1, 0.5, 0.1),),
            polarities=(NeuronPolarity.MIXED, NeuronPolarity.EXCITATORY),
            plasticity=(rule,),
        )


def test_shared_signed_group_requires_matching_polarities(core):
    graph = _network().graph
    graph = replace(
        graph,
        nodes=graph.nodes + (replace(graph.nodes[0], id=2),),
        edges=(
            replace(graph.edges[0], weight_group=0),
            replace(graph.edges[0], id=1, pre=2, weight_group=0),
        ),
    )
    graph.resolve()
    incompatible = replace(
        graph,
        nodes=graph.nodes[:2]
        + (replace(graph.nodes[2], polarity=NeuronPolarity.EXCITATORY),),
    )
    with pytest.raises(ResolutionError, match="polarity"):
        incompatible.resolve()
    with pytest.raises(ValueError, match="polarity"):
        core.compile_mixed(
            (_reactive(),) * 3,
            edges=(MixedEdge(0, 1, 1.0, 0.1), MixedEdge(2, 1, 1.0, 0.1)),
            polarities=(
                NeuronPolarity.MIXED,
                NeuronPolarity.EXCITATORY,
                NeuronPolarity.EXCITATORY,
            ),
            weight_groups=(0, 0),
        )
