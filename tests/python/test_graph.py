from __future__ import annotations

import json
import math

import pytest

from lacuna import (
    DriveInput,
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    InputMode,
    InputPort,
    NeuronPolarity,
    OutputPort,
    RecordingConfig,
    SpikeInput,
    StateInspectionRequest,
    TraceKind,
)
from lacuna.errors import CapabilityError, ResolutionError
from lacuna.ffi import CoreEvaluator

from .test_dsl_resolver import LIF


def _graph() -> Graph:
    return Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(
            GraphNode(20, "lif", -65.0, {"drive": 0.0}),
            GraphNode(10, "lif", -65.0, {"drive": 0.0}),
        ),
        edges=(GraphEdge(0, 10, 20, 20.0, 1.0),),
        input_ports=(
            InputPort("stimulus", 20, InputMode.SPIKE),
            InputPort("bias", 10, InputMode.DRIVE, "drive"),
        ),
        output_ports=(OutputPort("sink", 20), OutputPort("pacemaker", 10)),
    )


def test_graph_round_trip_is_byte_stable_and_expands_bindings() -> None:
    graph = _graph()
    first = graph.to_text()
    loaded = Graph.from_text(first)
    second = loaded.to_text()
    assert second == first
    document = json.loads(first)
    assert [node["id"] for node in document["nodes"]] == [10, 20]
    assert set(document["nodes"][0]["bindings"]) == {
        "drive",
        "tau_m",
        "v_reset",
        "v_rest",
        "v_th",
    }
    assert document["nodes"][0]["initial"].startswith("-0x")


def test_schema_9_graphs_remain_readable_and_upgrade_to_schema_10() -> None:
    document = json.loads(_graph().to_text())
    document["schema"] = 9
    restored = Graph.from_text(json.dumps(document))
    assert json.loads(restored.to_text())["schema"] == 10


def test_dale_polarity_signs_every_outgoing_edge_and_round_trips() -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(
            GraphNode(
                0,
                "lif",
                -65.0,
                {"drive": 0.0},
                polarity=NeuronPolarity.INHIBITORY,
            ),
            GraphNode(1, "lif", -65.0, {"drive": 0.0}),
            GraphNode(2, "lif", -65.0, {"drive": 0.0}),
        ),
        edges=(
            GraphEdge(0, 0, 1, 3.0),
            GraphEdge(1, 0, 2, 5.0),
        ),
    )

    resolved = graph.resolve()
    assert [edge.weight for edge in resolved.edges] == [3.0, 5.0]
    assert [edge.weight for edge in resolved.effective_edges] == [-3.0, -5.0]
    restored = Graph.from_text(graph.to_text())
    assert restored.nodes[0].polarity is NeuronPolarity.INHIBITORY
    assert restored.to_text() == graph.to_text()


def test_graph_rejects_signed_edge_weights() -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        edges=(GraphEdge(0, 0, 0, -1.0),),
    )
    with pytest.raises(ResolutionError, match="nonnegative magnitude"):
        graph.resolve()


def test_embedded_model_hash_is_verified() -> None:
    document = json.loads(_graph().to_text())
    document["models"][0]["model_hash"] = "0" * 64
    with pytest.raises(ResolutionError, match="model hash mismatch"):
        Graph.from_text(json.dumps(document))


def test_named_ports_drive_and_filter_outputs(core: CoreEvaluator) -> None:
    result = _graph().resolve().run(
        core,
        spike_inputs=[SpikeInput(1.0, "stimulus", 20.0)],
        drive_inputs=[DriveInput(5.0, "bias", 20.0)],
        t_end=19.0,
    )
    assert [(event.port, event.node) for event in result.outputs] == [
        ("sink", 20),
        ("pacemaker", 10),
    ]
    assert [event.t for event in result.outputs] == pytest.approx(
        [1.0, 5.0 + 10.0 * math.log(4.0)], abs=1e-12
    )


def test_graph_trace_uses_public_node_identifiers(core: CoreEvaluator) -> None:
    result = _graph().resolve().run(
        core,
        spike_inputs=[SpikeInput(1.0, "stimulus", 20.0)],
        t_end=2.0,
        recording=RecordingConfig(
            kinds=frozenset(
                {
                    TraceKind.INPUT_SPIKE,
                    TraceKind.DEPOSIT_APPLY,
                    TraceKind.SPIKE,
                    TraceKind.RESET,
                    TraceKind.FINAL_STATE,
                }
            ),
            nodes=(20,),
            capacity=8,
        ),
    )
    assert [record.kind for record in result.trace] == [
        TraceKind.INPUT_SPIKE,
        TraceKind.DEPOSIT_APPLY,
        TraceKind.SPIKE,
        TraceKind.RESET,
        TraceKind.FINAL_STATE,
    ]
    assert {record.node for record in result.trace} == {20}
    assert result.trace[1].state_names == ("v",)


def test_graph_inspection_uses_public_node_and_state_names(
    core: CoreEvaluator,
) -> None:
    result = _graph().resolve().run(
        core,
        spike_inputs=[SpikeInput(1.0, "stimulus", 20.0)],
        t_end=2.0,
        inspections=(StateInspectionRequest(1.0, 20, (0,)),),
    )
    assert len(result.inspections) == 1
    inspection = result.inspections[0]
    assert inspection.node == 20
    assert inspection.state_indices == (0,)
    assert inspection.state_names == ("v",)
    assert inspection.values == (-65.0,)
    assert inspection.clamped is True


def test_drive_port_cannot_change_structural_coefficients(core: CoreEvaluator) -> None:
    with pytest.raises(CapabilityError, match="changes more than the affine b term"):
        Graph(
            models=(GraphModel("lif", LIF),),
            nodes=(GraphNode(0, "lif", -65.0, {}),),
            input_ports=(InputPort("bad", 0, InputMode.DRIVE, "tau_m"),),
        ).resolve()
