from __future__ import annotations

import math

import pytest

from lacuna import (
    AugmentedState,
    DriveInput,
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    InputMode,
    InputPort,
    OutputPort,
    RecordingConfig,
    StateInspectionRequest,
    SpikeInput,
    TraceKind,
    audit_causal_trace,
    parse_neuron,
    resolve_stepped_neuron,
)
from lacuna.ffi import CoreEvaluator
from lacuna.ir import DispatchForm, ResolvedScalarLIF, ResolvedSteppedNeuron

from .test_dsl_resolver import LIF


QIF = """
neuron qif {
  params {
    drive = 1
    threshold = 1
    reset_value = 0
  }
  state { v: membrane }
  dynamics { dv/dt = v*v + drive }
  threshold { v > threshold }
  reset { v <- reset_value }
}
"""


ADEX = """
neuron adex {
  params {
    C: positive = 200
    gL: positive = 10
    EL = -70
    VT = -50
    DeltaT: positive = 2
    tauw: positive = 30
    a = 2
    b = 40
    I = 500
    v_reset = -58
    v_spike = -40
  }
  state {
    v: membrane
    w: adaptation
  }
  dynamics {
    dv/dt = (-gL*(v-EL) + gL*DeltaT*exp((v-VT)/DeltaT) - w + I)/C
    dw/dt = (a*(v-EL)-w)/tauw
  }
  threshold { v > v_spike }
  reset {
    v <- v_reset
    w <- w+b
  }
  refractory { duration = 2 }
}
"""


def test_nonlinear_qif_resolves_to_generic_stepper_and_hits_exact_crossing(
    core: CoreEvaluator,
) -> None:
    graph = Graph(
        models=(GraphModel("qif", QIF),),
        nodes=(GraphNode(0, "qif", 0.0, {}),),
        output_ports=(OutputPort("out", 0),),
    )
    resolved = graph.resolve()
    model = resolved.models[0]
    assert isinstance(model, ResolvedSteppedNeuron)
    assert model.dispatch is DispatchForm.STEPPED

    prediction = core.predict_stepped(model, AugmentedState((0.0,), 0.0), 1.0)
    assert prediction.t_spike == pytest.approx(math.pi / 4.0, abs=2e-8)
    assert prediction.diagnostics.accepted_steps > 0
    assert prediction.diagnostics.rhs_evaluations >= 7

    result = resolved.run(core, t_end=1.7)
    assert [spike.t for spike in result.core.spikes] == pytest.approx(
        [math.pi / 4.0, math.pi / 2.0], abs=3e-8
    )


def test_adex_executes_end_to_end_and_round_trips_schema_10(
    core: CoreEvaluator,
) -> None:
    graph = Graph(
        models=(GraphModel("adex", ADEX),),
        nodes=(GraphNode(0, "adex", (-70.0, 0.0), {}),),
        output_ports=(OutputPort("out", 0),),
    )
    resolved = graph.resolve()
    assert isinstance(resolved.models[0], ResolvedSteppedNeuron)
    result = resolved.run(core, t_end=100.0)
    assert [item.t for item in result.core.spikes] == pytest.approx(
        [
            14.0920641071,
            25.7942401681,
            38.4308839488,
            51.7613929965,
            65.5535183975,
            79.6286522023,
            93.8681553071,
        ],
        abs=5e-7,
    )
    assert len(result.core.states[0].values) == 2
    text = graph.to_text()
    assert Graph.from_text(text).to_text() == text
    assert '"schema": 10' in text


def test_mixed_analytical_and_stepped_nodes_share_zero_delay_scheduler(
    core: CoreEvaluator,
) -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF), GraphModel("adex", ADEX)),
        nodes=(
            GraphNode(10, "lif", -65.0, {"drive": 0.0}),
            GraphNode(20, "adex", (-70.0, 0.0), {}),
        ),
        edges=(GraphEdge(0, 10, 20, 35.0, 0.0),),
        input_ports=(InputPort("stimulus", 10, InputMode.SPIKE),),
        output_ports=(OutputPort("lif_out", 10), OutputPort("adex_out", 20)),
    )
    resolved = graph.resolve()
    assert isinstance(resolved.models[0], ResolvedScalarLIF)
    assert isinstance(resolved.models[1], ResolvedSteppedNeuron)
    result = resolved.run(
        core,
        spike_inputs=(SpikeInput(0.0, "stimulus", 20.0),),
        t_end=1.0,
    )
    assert [(item.node, item.t) for item in result.core.spikes] == [
        (0, 0.0),
        (1, 0.0),
    ]


def test_safe_nonlinear_intrinsics_lower_without_host_callbacks() -> None:
    model = resolve_stepped_neuron(parse_neuron(ADEX))
    assert model.dispatch is DispatchForm.STEPPED
    assert any(node.op.name == "EXP" for node in model.propagation_dag.nodes)


def test_piecewise_constant_drive_invalidates_a_stepped_prediction(
    core: CoreEvaluator,
) -> None:
    graph = Graph(
        models=(GraphModel("adex", ADEX),),
        nodes=(GraphNode(0, "adex", (-70.0, 0.0), {}),),
        input_ports=(InputPort("current", 0, InputMode.DRIVE, "I"),),
    ).resolve()
    active = graph.run(core, t_end=20.0)
    silent = graph.run(
        core,
        drive_inputs=(DriveInput(0.0, "current", 0.0),),
        t_end=20.0,
    )
    assert len(active.core.spikes) == 1
    assert silent.core.spikes == ()
    assert silent.core.stats.drive_updates_processed == 1


def test_stepped_population_uses_one_deterministic_same_time_batch(
    core: CoreEvaluator,
) -> None:
    count = 32
    resolved = Graph(
        models=(GraphModel("qif", QIF),),
        nodes=tuple(GraphNode(index, "qif", 0.0, {}) for index in range(count)),
    ).resolve()
    result = resolved.run(core, t_end=1.0, output_capacity=count)
    assert [item.node for item in result.core.spikes] == list(range(count))
    assert [item.t for item in result.core.spikes] == pytest.approx(
        [math.pi / 4.0] * count, abs=2e-8
    )


def test_complete_stepped_trace_passes_the_independent_scheduler_audit(
    core: CoreEvaluator,
) -> None:
    resolved = Graph(
        models=(GraphModel("qif", QIF),),
        nodes=(GraphNode(0, "qif", 0.0, {}),),
    ).resolve()
    result = resolved.run(
        core,
        t_end=1.0,
        recording=RecordingConfig(capacity=64),
    ).core
    audit = audit_causal_trace(
        result,
        models=resolved.models,
        edges=resolved.edges,
        t_end=1.0,
    )
    assert audit.spike_records == 1
    assert audit.final_state_records == 1


def test_stepped_state_inspection_and_trace_use_the_existing_observability_path(
    core: CoreEvaluator,
) -> None:
    resolved = Graph(
        models=(GraphModel("qif", QIF),),
        nodes=(GraphNode(7, "qif", 0.0, {}),),
    ).resolve()
    result = resolved.run(
        core,
        t_end=0.6,
        inspections=(StateInspectionRequest(0.5, 7, (0,)),),
        recording=RecordingConfig(
            kinds=frozenset({TraceKind.FINAL_STATE}),
            nodes=(7,),
            capacity=2,
        ),
    )
    assert result.inspections[0].state_names == ("v",)
    assert result.inspections[0].values == pytest.approx((math.tan(0.5),), abs=2e-8)
    assert result.trace[0].kind is TraceKind.FINAL_STATE
    assert result.trace[0].state_names == ("v",)
