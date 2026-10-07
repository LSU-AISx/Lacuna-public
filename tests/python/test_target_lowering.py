"""Target lowering must use native arithmetic before building network state."""

from dataclasses import replace
import math
import struct

import pytest

from lacuna import AdaptiveLIF, CoreEvaluator, Graph, LIF, NetworkBuilder
from lacuna.errors import PrecisionResolutionError
from lacuna.target_lowering import resolve_target_graph

from .test_execution_plan_matrix import PLAN_CASES, _matrix_inputs
from .test_per_edge import _graph, _alpha_graph


@pytest.fixture(params=("float32", "float32-time64"))
def target_core(request):
    return CoreEvaluator(precision=request.param)


def _f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


@pytest.mark.parametrize("name,factory", PLAN_CASES, ids=[name for name, _ in PLAN_CASES])
def test_diverse_target_matrix_executes_authored_graphs(target_core, core, name, factory):
    graph = factory()
    original_text = graph.to_text()
    resolved = resolve_target_graph(graph, target_core)
    assert resolved.precision is target_core.precision
    assert resolved.target_binding_key
    assert graph.to_text() == original_text
    plan = resolved.execution_plan()
    with target_core.compile_execution_plan(plan) as compiled:
        result = compiled.run(resolved.initial_values, inputs=_matrix_inputs(resolved), t_end=8.0)
    reference = graph.resolve()
    with core.compile_execution_plan(reference.execution_plan()) as compiled:
        expected = compiled.run(reference.initial_values, inputs=_matrix_inputs(reference), t_end=8.0)
    assert len(result.spikes) == len(expected.spikes)
    assert [spike.node for spike in result.spikes] == [spike.node for spike in expected.spikes]
    assert [spike.t for spike in result.spikes] == pytest.approx(
        [spike.t for spike in expected.spikes], abs=2e-3,
    )
    for actual_state, expected_state in zip(result.states, expected.states):
        assert actual_state.values == pytest.approx(expected_state.values, abs=3e-3, rel=1e-4)
        assert all(math.isfinite(value) and value == _f32(value) for value in actual_state.values)


def test_scalar_coefficient_is_evaluated_not_cast(target_core):
    model = LIF(tau_m=20.1, v_rest=-65.3, drive=24.7)
    graph = Graph(models=(model.model,), nodes=(model.node(1),))
    resolved = resolve_target_graph(graph, target_core).models[0]
    expected = _f32(_f32(_f32(-65.3) + _f32(24.7)) * _f32(1 / _f32(20.1)))
    assert resolved.b == expected
    assert resolved.a == _f32(-1 / _f32(20.1))


def test_resolved_graph_is_not_relabelled_for_another_target(target_core):
    builder = NetworkBuilder("target")
    builder.neuron("n", LIF())
    graph = builder.build().graph
    host = graph.resolve()
    target = resolve_target_graph(graph, target_core)
    assert host.precision.value == "float64"
    assert host.execution_plan().target_binding_key is None
    assert target.execution_plan().precision is target_core.precision
    with pytest.raises(ValueError, match="precision differs"):
        from lacuna.execution_plan import lower_execution_plan
        lower_execution_plan(host, precision=target_core.precision)


def test_native_rate_collision_is_rejected(target_core):
    model = AdaptiveLIF(tau_m=24.00001335144043, tau_adaptation=24.000015258789062)
    graph = Graph(models=(model.model,), nodes=(model.node(0),))
    with pytest.raises(PrecisionResolutionError, match="collapse"):
        resolve_target_graph(graph, target_core)


def test_rounding_cannot_silently_downgrade_analytical_execution(target_core):
    model = AdaptiveLIF(tau_m=20.0, tau_adaptation=20.00000001)
    graph = Graph(models=(model.model,), nodes=(model.node(0),))
    with pytest.raises(PrecisionResolutionError, match="analytical dynamics to stepped"):
        resolve_target_graph(graph, target_core)


def test_target_initial_threshold_separation_is_checked(target_core):
    from lacuna import GraphModel, GraphNode

    model = GraphModel("m", """
neuron TargetThreshold {
    params { eps = 0.0000000298023223876953125 }
    state { v : membrane }
    dynamics { dv/dt = -v }
    threshold { v > 1 + eps }
    reset { v <- 0 }
}
""")
    graph = Graph(models=(model,), nodes=(GraphNode(0, "m", 1.0, {}),))
    with pytest.raises(PrecisionResolutionError, match="initial state.*target threshold"):
        resolve_target_graph(graph, target_core)


@pytest.mark.parametrize("factory", (_graph, _alpha_graph))
def test_per_edge_noncontiguous_ids_and_equal_rates(target_core, factory):
    graph = factory(taus=(10.0, 5.0), weights=(10.0, 10.0))
    graph = replace(graph, edges=tuple(
        replace(edge, id=10 + 3 * index) for index, edge in enumerate(graph.edges)
    ))
    resolved = resolve_target_graph(graph, target_core)
    with resolved.compile(target_core) as compiled:
        result = compiled.run(t_end=4.0)
    assert result.core.states


def test_grouped_edge_initials_are_added_in_native_precision(target_core):
    graph = _graph(taus=(5.0, 5.0, 5.0), weights=(1.0, 1.0, 1.0))
    graph = replace(graph, edges=tuple(
        replace(edge, initial=value)
        for edge, value in zip(graph.edges, (2**24, 1.0, -(2**24)))
    ))
    assert graph.resolve().models[1].group_initials == (1.0,)
    target = resolve_target_graph(graph, target_core)
    assert target.models[1].group_initials == (0.0,)
    assert target.initial_values[1][1:] == (0.0,)


def test_unordered_nodes_keep_their_parameter_bindings(target_core):
    one = LIF(name="one", drive=24.0)
    two = LIF(name="two", drive=25.0)
    graph = Graph(models=(one.model, two.model), nodes=(two.node(99), one.node(4)))
    target = resolve_target_graph(graph, target_core)
    assert target.node_ids == (4, 99)
    assert target.models[0].bindings["drive"] == 24.0
    assert target.models[1].bindings["drive"] == 25.0
