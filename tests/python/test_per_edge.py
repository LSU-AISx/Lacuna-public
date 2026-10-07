from __future__ import annotations

import json

import mpmath
import pytest

from lacuna import (
    AugmentedState,
    CoreEvaluator,
    DriveInput,
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
from lacuna.errors import CapabilityError, ResolutionError
from lacuna.ir import ExpPolyRootHint, ResolvedPerEdgeLIF, SynapseTier

from .test_alpha import ALPHA_LIF, ALPHA_SYNAPSE
from .test_dsl_resolver import LIF


EXP_SYNAPSE = """
synapse exp_exc {
    params { tau_s : positive = 5.0 }
    state { s }
    dynamics { ds/dt = -s/tau_s }
    on_spike { s <- s + w }
    output { current = s }
}
"""


def _graph(*, taus: tuple[float, ...], weights: tuple[float, ...]) -> Graph:
    return Graph(
        models=(GraphModel("source", LIF), GraphModel("target", ALPHA_LIF)),
        nodes=(
            GraphNode(0, "source", -65.0, {"drive": 0.0}),
            GraphNode(1, "target", -65.0, {}),
        ),
        edges=tuple(
            GraphEdge(
                index,
                0,
                1,
                weight,
                0.0,
                synapse="exp",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": tau},
            )
            for index, (tau, weight) in enumerate(zip(taus, weights))
        ),
        input_ports=(InputPort("stimulus", 0, InputMode.SPIKE),),
        output_ports=(OutputPort("response", 1),),
        synapses=(GraphSynapse("exp", EXP_SYNAPSE),),
    )


def _first_crossing(taus: tuple[float, ...], weights: tuple[float, ...]) -> float:
    mpmath.mp.dps = 80

    def voltage(delta):
        membrane_rate = mpmath.mpf(-1) / 10
        total = mpmath.mpf(-65)
        for tau, weight in zip(taus, weights):
            rate = -mpmath.mpf(1) / tau
            if rate == membrane_rate:
                total += weight * delta * mpmath.exp(rate * delta)
            else:
                total += weight * (
                    mpmath.exp(rate * delta) - mpmath.exp(membrane_rate * delta)
                ) / (rate - membrane_rate)
        return total

    low = mpmath.mpf("0")
    high = mpmath.mpf("1")
    while voltage(high) < -50:
        high *= 2
    for _ in range(260):
        middle = (low + high) / 2
        if voltage(middle) < -50:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def _alpha_graph(
    *, taus: tuple[float, ...], weights: tuple[float, ...], initials=()
) -> Graph:
    base = _graph(taus=(5.0,), weights=(1.0,))
    return Graph(
        models=base.models,
        nodes=base.nodes,
        edges=tuple(
            GraphEdge(
                index,
                0,
                1,
                weight,
                synapse="alpha",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": tau},
                initial=initials[index] if initials else 0.0,
            )
            for index, (tau, weight) in enumerate(zip(taus, weights))
        ),
        input_ports=base.input_ports,
        output_ports=base.output_ports,
        synapses=(GraphSynapse("alpha", ALPHA_SYNAPSE),),
    )


def _first_alpha_crossing(
    taus: tuple[float, ...], weights: tuple[float, ...]
) -> float:
    mpmath.mp.dps = 80
    membrane_rate = -mpmath.mpf(1) / 10

    def voltage(delta):
        total = mpmath.mpf(-65)
        for tau, weight in zip(taus, weights):
            rate = -mpmath.mpf(1) / tau
            gap = (rate - membrane_rate) * delta
            derivative = (
                mpmath.mpf("0.5")
                if gap == 0
                else (gap * mpmath.exp(gap) - mpmath.expm1(gap)) / gap**2
            )
            total += (
                rate**2
                * weight
                * mpmath.exp(membrane_rate * delta)
                * delta**2
                * derivative
            )
        return total

    low = mpmath.mpf("0")
    high = mpmath.mpf("0.125")
    while voltage(high) < -50:
        low = high
        high *= 2
        if high > 4096:
            raise AssertionError("reference trajectory did not cross")
    for _ in range(260):
        middle = (low + high) / 2
        if voltage(middle) < -50:
            low = middle
        else:
            high = middle
    return float((low + high) / 2)


def test_equal_edge_modes_fold_into_one_shared_state() -> None:
    resolved = _graph(taus=(5.0, 5.0), weights=(10.0, 10.0)).resolve()
    model = resolved.models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    assert model.tier is SynapseTier.FOLDED_SHARED
    assert model.state_names == ("v", "edge_mode[0].exp_exc.s")
    assert model.group_edge_ids == ((0, 1),)
    assert [edge.target for edge in resolved.edges] == [1, 1]
    single = _graph(taus=(5.0,), weights=(20.0,)).resolve().models[1]
    distinct = _graph(taus=(5.0, 7.0), weights=(10.0, 10.0)).resolve().models[1]
    assert single.model_hash == model.model_hash == distinct.model_hash
    assert single.resolution_key == model.resolution_key
    assert distinct.resolution_key != model.resolution_key


def test_distinct_edge_modes_remain_separate_and_cross_analytically(
    core: CoreEvaluator,
) -> None:
    taus = (5.0, 7.0)
    weights = (10.0, 10.0)
    resolved = _graph(taus=taus, weights=weights).resolve()
    model = resolved.models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    assert model.tier is SynapseTier.PER_EDGE
    assert model.state_names == (
        "v",
        "edge_mode[0].exp_exc.s",
        "edge_mode[1].exp_exc.s",
    )
    assert [edge.target for edge in resolved.edges] == [1, 2]

    result = resolved.run(
        core,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
        t_end=2.0,
    )
    assert len(result.outputs) == 1
    expected = 1.0 + _first_crossing(taus, weights)
    assert result.outputs[0].t == pytest.approx(expected, abs=2e-9)


def test_equal_membrane_and_scalar_synapse_rate_uses_linear_exp_mode(
    core: CoreEvaluator,
) -> None:
    taus = (10.0,)
    weights = (20.0,)
    resolved = _graph(taus=taus, weights=weights).resolve()
    result = resolved.run(
        core,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
        t_end=30.0,
    )
    assert result.outputs[0].t == pytest.approx(
        1.0 + _first_crossing(taus, weights), abs=2e-9
    )


@pytest.mark.parametrize(
    ("tau", "weight"),
    ((5.0, 40.0), (10.0, 80.0)),
)
def test_per_edge_alpha_crosses_for_distinct_and_equal_membrane_rates(
    core: CoreEvaluator, tau: float, weight: float
) -> None:
    resolved = _alpha_graph(taus=(tau,), weights=(weight,)).resolve()
    model = resolved.models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    assert model.state_names == (
        "v",
        "edge_mode[0].alpha_exc.s",
        "edge_mode[0].alpha_exc.z",
    )
    assert resolved.edges[0].target == 2
    assert resolved.edges[0].weight == pytest.approx(weight)
    assert resolved.edges[0].deposit_scale == pytest.approx(1.0 / tau**2)
    result = resolved.run(
        core,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
        t_end=60.0,
    )
    expected = 1.0 + _first_alpha_crossing((tau,), (weight,))
    assert result.outputs[0].t == pytest.approx(expected, abs=3e-9)


def test_distinct_alpha_blocks_use_bounded_multi_exp_polynomial_crossing(
    core: CoreEvaluator,
) -> None:
    taus = (5.0, 10.0)
    weights = (20.0, 40.0)
    resolved = _alpha_graph(taus=taus, weights=weights).resolve()
    model = resolved.models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    assert len(model.state_names) == 5
    assert [edge.target for edge in resolved.edges] == [2, 4]
    result = resolved.run(
        core,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
        t_end=60.0,
    )
    expected = 1.0 + _first_alpha_crossing(taus, weights)
    assert result.outputs[0].t == pytest.approx(expected, abs=3e-9)


def test_multi_alpha_propagation_has_semigroup_property(core: CoreEvaluator) -> None:
    model = _alpha_graph(taus=(5.0, 10.0), weights=(1.0, 1.0)).resolve().models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    initial = AugmentedState((-63.0, 1.2, -0.4, 0.8, 0.2), 0.0)
    split = core.advance_analytical(model, initial, 3.0)
    split = core.advance_analytical(model, split, 9.0)
    combined = core.advance_analytical(model, initial, 9.0)
    assert split.values == pytest.approx(combined.values, abs=3e-13)


def test_scalar_and_alpha_edge_blocks_share_generic_crossing_without_folding(
    core: CoreEvaluator,
) -> None:
    base = _graph(taus=(5.0,), weights=(1.0,))
    graph = Graph(
        models=base.models,
        nodes=base.nodes,
        edges=(
            GraphEdge(
                0,
                0,
                1,
                10.0,
                synapse="exp",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": 5.0},
            ),
            GraphEdge(
                1,
                0,
                1,
                40.0,
                synapse="alpha",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": 10.0},
            ),
        ),
        input_ports=base.input_ports,
        output_ports=base.output_ports,
        synapses=(
            GraphSynapse("exp", EXP_SYNAPSE),
            GraphSynapse("alpha", ALPHA_SYNAPSE),
        ),
    )
    resolved = graph.resolve()
    model = resolved.models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    assert isinstance(model.root_hint, ExpPolyRootHint)
    assert len(model.state_names) == 4
    assert [edge.target for edge in resolved.edges] == [1, 3]

    mpmath.mp.dps = 80
    a = -mpmath.mpf(1) / 10
    q = -mpmath.mpf(1) / 5

    def voltage(delta):
        scalar = 10 * (mpmath.exp(q * delta) - mpmath.exp(a * delta)) / (q - a)
        alpha = mpmath.mpf("0.2") * delta**2 * mpmath.exp(a * delta)
        return -65 + scalar + alpha

    low = mpmath.mpf(0)
    high = mpmath.mpf(1)
    while voltage(high) < -50:
        low = high
        high *= 2
    for _ in range(260):
        middle = (low + high) / 2
        if voltage(middle) < -50:
            low = middle
        else:
            high = middle
    expected = 1.0 + float((low + high) / 2)
    result = resolved.run(
        core,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
        t_end=30.0,
    )
    assert result.outputs[0].t == pytest.approx(expected, abs=3e-9)


def test_per_edge_state_program_supports_selected_dependency_closure(
    core: CoreEvaluator,
) -> None:
    model = _graph(taus=(5.0, 7.0), weights=(10.0, 10.0)).resolve().models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    initial = AugmentedState((-65.0, 10.0, 10.0), 0.0)
    complete = core.advance_analytical(model, initial, 1.0)
    selected = core.advance_analytical_selected(model, initial, 1.0, (1, 2))
    assert selected == pytest.approx(complete.values[1:], abs=1e-14)
    assert selected == pytest.approx(
        (10.0 * mpmath.exp(-1.0 / 5.0), 10.0 * mpmath.exp(-1.0 / 7.0)),
        abs=1e-14,
    )


def test_per_edge_node_keeps_validated_neuron_drive_bindings(
    core: CoreEvaluator,
) -> None:
    driven_target = ALPHA_LIF.replace(
        "v_rest = -65.0", "v_rest = -65.0\n        drive = 0.0"
    ).replace(
        "-(v - v_rest)/tau_m + i_exc",
        "-(v - v_rest)/tau_m + drive/tau_m + i_exc",
    )
    graph = _graph(taus=(5.0,), weights=(1.0,))
    graph = Graph(
        models=(graph.models[0], GraphModel("target", driven_target)),
        nodes=graph.nodes,
        edges=graph.edges,
        input_ports=(
            *graph.input_ports,
            InputPort("bias", 1, InputMode.DRIVE, "drive"),
        ),
        output_ports=graph.output_ports,
        synapses=graph.synapses,
    )
    result = graph.resolve().run(
        core,
        drive_inputs=(DriveInput(0.0, "bias", 20.0),),
        t_end=20.0,
    )
    assert [item.t for item in result.outputs] == pytest.approx(
        [10.0 * mpmath.log(4.0)], abs=2e-9
    )


def test_shared_and_distinct_paths_preserve_one_shot_incremental_equivalence(
    core: CoreEvaluator,
) -> None:
    stimulus = (SpikeInput(1.0, "stimulus", 20.0),)
    for graph in (
        _graph(taus=(5.0, 5.0), weights=(10.0, 10.0)),
        _graph(taus=(5.0, 7.0), weights=(10.0, 10.0)),
    ):
        resolved = graph.resolve()
        expected = resolved.run(core, spike_inputs=stimulus, t_end=12.0)
        with resolved.compile(core) as compiled:
            with compiled.create_incremental_run(t_end=12.0) as run:
                first = run.advance_until(1.0)
                second = run.advance_until(6.0, spike_inputs=stimulus)
                actual = run.finish()
        outputs = (*first.outputs, *second.outputs, *actual.outputs)
        assert [(item.port, item.node) for item in outputs] == [
            (item.port, item.node) for item in expected.outputs
        ]
        assert [item.t for item in outputs] == pytest.approx(
            [item.t for item in expected.outputs], rel=0.0, abs=2e-12
        )
        for actual_state, expected_state in zip(
            actual.core.states, expected.core.states
        ):
            assert actual_state.values == pytest.approx(
                expected_state.values, abs=2e-12
            )


def test_edge_synapse_schema_round_trips_with_separate_bindings() -> None:
    graph = _graph(taus=(5.0, 7.0), weights=(10.0, 11.0))
    text = graph.to_text()
    assert Graph.from_text(text).to_text() == text
    document = json.loads(text)
    assert document["schema"] == 10
    assert document["edges"][0]["synapse"] == "exp"
    assert document["edges"][1]["synapse_bindings"]["tau_s"] == "0x1.c000000000000p+2"


def test_delta_edge_can_coexist_without_consuming_another_synapse_mode() -> None:
    graph = _graph(taus=(5.0,), weights=(10.0,))
    graph = Graph(
        models=graph.models,
        nodes=graph.nodes,
        edges=(*graph.edges, GraphEdge(1, 0, 1, 1.0)),
        input_ports=graph.input_ports,
        output_ports=graph.output_ports,
        synapses=graph.synapses,
    )
    resolved = graph.resolve()
    model = resolved.models[1]
    assert isinstance(model, ResolvedPerEdgeLIF)
    assert len(model.state_names) == 2
    assert [edge.target for edge in resolved.edges] == [1, 0]


def test_edge_initial_state_is_aggregated_only_for_equal_modes() -> None:
    graph = _graph(taus=(5.0, 5.0), weights=(1.0, 1.0))
    graph = Graph(
        models=graph.models,
        nodes=graph.nodes,
        edges=(
            GraphEdge(**{**graph.edges[0].__dict__, "initial": 2.0}),
            GraphEdge(**{**graph.edges[1].__dict__, "initial": 3.0}),
        ),
        input_ports=graph.input_ports,
        output_ports=graph.output_ports,
        synapses=graph.synapses,
    )
    resolved = graph.resolve()
    assert resolved.initial_values[1] == (-65.0, 5.0)


def test_equal_alpha_edges_aggregate_vector_initial_state_and_round_trip() -> None:
    graph = _alpha_graph(
        taus=(5.0, 5.0),
        weights=(1.0, 1.0),
        initials=((1.0, 2.0), (3.0, 4.0)),
    )
    resolved = graph.resolve()
    assert resolved.initial_values[1] == (-65.0, 4.0, 6.0)
    assert [edge.target for edge in resolved.edges] == [2, 2]
    text = graph.to_text()
    assert Graph.from_text(text).to_text() == text
    assert json.loads(text)["edges"][0]["initial"] == [
        "0x1.0000000000000p+0",
        "0x1.0000000000000p+1",
    ]


def test_per_edge_rejects_unbounded_or_unsupported_regimes() -> None:
    too_many = _graph(
        taus=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
        weights=(1.0,) * 8,
    )
    with pytest.raises(CapabilityError, match="eight-state"):
        too_many.resolve()

    repeated = _graph(taus=(10.0,), weights=(20.0,)).resolve()
    assert isinstance(repeated.models[1], ResolvedPerEdgeLIF)

    alpha_edge = _graph(taus=(5.0,), weights=(1.0,))
    alpha_edge = Graph(
        models=alpha_edge.models,
        nodes=alpha_edge.nodes,
        edges=(
            GraphEdge(
                0,
                0,
                1,
                1.0,
                synapse="alpha",
                receptor="i_exc",
                output="current",
            ),
        ),
        input_ports=alpha_edge.input_ports,
        output_ports=alpha_edge.output_ports,
        synapses=(GraphSynapse("alpha", ALPHA_SYNAPSE),),
    )
    resolved_alpha = alpha_edge.resolve()
    assert isinstance(resolved_alpha.models[1], ResolvedPerEdgeLIF)
    assert resolved_alpha.models[1].state_names == (
        "v",
        "edge_mode[0].alpha_exc.s",
        "edge_mode[0].alpha_exc.z",
    )


def test_synapse_fields_without_a_synapse_are_rejected() -> None:
    graph = _graph(taus=(5.0,), weights=(1.0,))
    graph = Graph(
        models=graph.models,
        nodes=graph.nodes,
        edges=(GraphEdge(0, 0, 1, 1.0, receptor="i_exc"),),
        input_ports=graph.input_ports,
        output_ports=graph.output_ports,
        synapses=graph.synapses,
    )
    with pytest.raises(ResolutionError, match="fields but no synapse"):
        graph.resolve()
