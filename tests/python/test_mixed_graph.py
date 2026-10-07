from __future__ import annotations

import json

import pytest

from lacuna import (
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    GraphSynapse,
    InputMode,
    InputPort,
    OutputPort,
    SpikeInput,
)
from lacuna.errors import ResolutionError
from lacuna.ffi import CoreEvaluator, MixedRunResult

from .test_alpha import ALPHA_LIF, ALPHA_SYNAPSE
from .test_dsl_resolver import LIF


def _mixed_graph() -> Graph:
    return Graph(
        models=(GraphModel("alpha_lif", ALPHA_LIF), GraphModel("lif", LIF)),
        nodes=(
            GraphNode(0, "lif", -65.0, {"drive": 0.0}),
            GraphNode(
                1,
                "alpha_lif",
                (-65.0, 0.0, 0.0),
                {},
                synapse="alpha",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": 5.0},
            ),
        ),
        edges=(GraphEdge(0, 0, 1, 40.0, 1.0),),
        input_ports=(InputPort("stimulus", 0, InputMode.SPIKE),),
        output_ports=(OutputPort("decoded", 1),),
        synapses=(GraphSynapse("alpha", ALPHA_SYNAPSE),),
    )


def test_mixed_graph_runs_through_named_input_and_output_ports(
    core: CoreEvaluator,
) -> None:
    result = _mixed_graph().resolve().run(
        core,
        spike_inputs=[SpikeInput(1.0, "stimulus", 20.0)],
        t_end=12.0,
    )
    assert isinstance(result.core, MixedRunResult)
    assert [(event.port, event.node) for event in result.outputs] == [("decoded", 1)]
    assert [event.t for event in result.outputs] == pytest.approx(
        [11.230709364362863], abs=5e-10
    )
    assert result.core.stats.deliveries_scheduled == 1
    assert result.core.stats.deliveries_processed == 1


def test_mixed_graph_serialization_is_byte_stable_with_separate_hashes() -> None:
    first = _mixed_graph().to_text()
    second = Graph.from_text(first).to_text()
    assert second == first
    document = json.loads(first)
    assert document["schema"] == 10
    assert len(document["models"][0]["model_hash"]) == 64
    assert len(document["synapses"][0]["synapse_hash"]) == 64
    assert document["nodes"][1]["initial"] == [
        "-0x1.0400000000000p+6",
        "0x0.0p+0",
        "0x0.0p+0",
    ]


def test_synapse_hash_is_verified_independently() -> None:
    document = json.loads(_mixed_graph().to_text())
    document["synapses"][0]["synapse_hash"] = "0" * 64
    with pytest.raises(ResolutionError, match="synapse hash mismatch"):
        Graph.from_text(json.dumps(document))


def test_alpha_node_requires_explicit_mapping() -> None:
    graph = Graph(
        models=(GraphModel("alpha_lif", ALPHA_LIF),),
        nodes=(GraphNode(0, "alpha_lif", -65.0, {}, synapse="alpha"),),
        synapses=(GraphSynapse("alpha", ALPHA_SYNAPSE),),
    )
    with pytest.raises(ResolutionError, match="requires receptor and output"):
        graph.resolve()
