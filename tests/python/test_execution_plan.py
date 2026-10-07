from __future__ import annotations

import struct
from dataclasses import replace

import lacuna.ffi as ffi_module

from lacuna import (
    AdaptiveLIF,
    ArithmeticMethod,
    AlphaCurrent,
    CrossingMethod,
    DeltaEdge,
    DepositOperation,
    Engine,
    EvolutionMethod,
    Graph,
    GraphModel,
    GraphNode,
    InputSpike,
    LIF,
    MixedInputSpike,
    NetworkBuilder,
    PairSTDP,
)
from lacuna.ffi import CoreEvaluator


def test_equation_name_does_not_control_plan_selection() -> None:
    source = LIF().source.replace(
        "neuron StandardLIF", "neuron ThisNameHasNoExecutionMeaning"
    )
    graph = Graph(
        models=(GraphModel("equation", source),),
        nodes=(GraphNode(0, "equation", -65.0, {}),),
    )

    plan = graph.resolve().execution_plan()

    assert plan.programs[0].crossing is CrossingMethod.SCALAR_LOG
    assert plan.compact_scalar_delta_compatible


def test_plan_derives_compact_scalar_execution_from_equations_and_wiring() -> None:
    builder = NetworkBuilder("derived-plan")
    population = builder.population(
        "nodes",
        3,
        LIF(name="arbitrarily_named_equation", drive=0.0),
        parameters={"drive": (16.0, 18.0, 20.0)},
    )
    builder.connect(population[0], population[2], weight=4.0, delay=1.5)

    plan = builder.build().graph.resolve().execution_plan()

    assert len(plan.programs) == 1
    program = plan.programs[0]
    assert program.evolution is EvolutionMethod.EXACT_EXPRESSIONS
    assert program.crossing is CrossingMethod.SCALAR_LOG
    assert program.state_names == ("v",)
    assert [node.state_offset for node in plan.nodes] == [0, 1, 2]
    assert [slot.initial for slot in plan.states] == [-65.0, -65.0, -65.0]
    drive_index = program.propagation.parameters.index("drive")
    assert [node.parameter_values[drive_index] for node in plan.nodes] == [
        16.0,
        18.0,
        20.0,
    ]
    assert plan.connections[0].operation is DepositOperation.ADD_STATE
    assert plan.connections[0].target_state == plan.nodes[2].readout_index
    assert {node.arithmetic for node in plan.nodes} == {
        ArithmeticMethod.SCALAR_AFFINE
    }
    assert plan.node_batches[0].arithmetic is ArithmeticMethod.SCALAR_AFFINE
    assert plan.compact_scalar_delta_compatible


def test_plan_reuses_equation_program_while_retaining_per_node_parameters() -> None:
    builder = NetworkBuilder()
    builder.population(
        "nodes",
        64,
        LIF(drive=0.0),
        parameters={"drive": tuple(16.0 + index / 64.0 for index in range(64))},
    )

    plan = builder.build().graph.resolve().execution_plan()

    assert len(plan.programs) == 1
    assert len(plan.nodes) == 64
    assert len(plan.states) == 64
    assert {node.program for node in plan.nodes} == {0}
    assert len({node.parameter_values for node in plan.nodes}) == 64
    assert len(plan.node_batches) == 1
    assert plan.node_batches[0].nodes == tuple(range(64))


def test_plan_packer_lowers_one_expression_program_per_shared_structure(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder()
    builder.population(
        "nodes",
        128,
        LIF(drive=0.0),
        parameters={"drive": tuple(16.0 + index / 128.0 for index in range(128))},
    )
    plan = builder.build().graph.resolve().execution_plan()

    packed = core._pack_execution_plan(plan)

    assert len(plan.programs) == 1
    assert len(packed[4]) == 1
    assert packed[6] is None
    assert {descriptor.arithmetic_kind for descriptor in packed[0]} == {1}


def test_plan_native_learning_never_constructs_legacy_rule_descriptors(
    core: CoreEvaluator,
    monkeypatch,
) -> None:
    class RejectLegacyDescriptor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("plan-native compilation constructed a legacy rule")

    monkeypatch.setattr(ffi_module, "_CPlasticityRule", RejectLegacyDescriptor)
    builder = NetworkBuilder("plan-native-learning-isolation")
    pre = builder.neuron("pre", LIF(drive=0.0))
    post = builder.neuron("post", LIF(drive=0.0))
    builder.connect(pre, post, weight=0.5, plasticity=PairSTDP())
    resolved = builder.build().graph.resolve()

    with core.compile_execution_plan(resolved.execution_plan()) as compiled:
        result = compiled.run(resolved.initial_values, t_end=1.0)

    assert result.weights == (0.5,)


def test_plan_describes_additional_exact_state_without_model_scenario_checks() -> None:
    builder = NetworkBuilder()
    builder.population("adaptive", 1, AdaptiveLIF())

    plan = builder.build().graph.resolve().execution_plan()

    assert plan.programs[0].evolution is EvolutionMethod.EXACT_EXPRESSIONS
    assert plan.programs[0].crossing is CrossingMethod.TWO_EXPONENTIAL
    assert plan.nodes[0].state_count == 2
    assert plan.nodes[0].arithmetic is ArithmeticMethod.EXPRESSION_DAG
    assert plan.state_count == 2
    assert not plan.compact_scalar_delta_compatible


def test_connection_state_and_learning_are_structural_plan_requirements() -> None:
    builder = NetworkBuilder()
    source = builder.population("source", 1, LIF(name="source_lif"))
    target = builder.population(
        "target", 1, LIF(name="target_lif", synaptic_input=True)
    )
    builder.connect(
        source,
        target,
        synapse=AlphaCurrent(5.0),
        weight=0.5,
        plasticity=PairSTDP(),
    )

    plan = builder.build().graph.resolve().execution_plan()

    connection = plan.connections[0]
    assert connection.operation is DepositOperation.ADD_STATE
    assert connection.target_state != plan.nodes[connection.post].readout_index
    assert connection.has_learning
    assert plan.requirements.learning_connections == 1
    assert not plan.compact_scalar_delta_compatible


def test_compiled_network_exposes_the_plan_used_by_the_shared_scheduler(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder()
    builder.population("nodes", 2, LIF(drive=20.0))

    with Engine(core._lib._name).compile(builder.build()) as compiled:
        assert compiled.execution_plan.compact_scalar_delta_compatible
        assert compiled.preferred_execution_path == "compiled_sparse"
        compiled.run(1.0)
        assert compiled.last_execution_path == "compiled_sparse"


def test_plan_lowering_matches_legacy_scalar_delta_entry_point(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder()
    population = builder.population(
        "nodes",
        2,
        LIF(drive=0.0),
        parameters={"drive": (20.0, 0.0)},
    )
    builder.connect(population[0], population[1], weight=3.0, delay=1.0)
    resolved = builder.build().graph.resolve()
    stimulus = (InputSpike(1.0, 0, 20.0),)

    planned = core.run_affine_delta_plan(
        resolved.execution_plan(), inputs=stimulus, t_end=20.0
    )
    legacy = core.run_delta(
        tuple(resolved.models),
        tuple(float(value) for value in resolved.initial_values),
        edges=(DeltaEdge(0, 1, 3.0, 1.0),),
        polarities=resolved.polarities,
        inputs=stimulus,
        t_end=20.0,
    )

    assert planned == legacy


def test_shared_scheduler_matches_compact_scalar_oracle_bitwise(
    core: CoreEvaluator,
) -> None:
    cases = []

    tonic = NetworkBuilder("tonic-affine")
    tonic.population(
        "nodes",
        3,
        LIF(drive=0.0),
        parameters={
            "drive": (0.0, 18.0, 24.0),
            "tau_m": (7.5, 20.0, 31.0),
        },
        initial=(-64.0, -63.0, -62.0),
    )
    cases.append((tonic.build().graph.resolve(), (), 40.0))

    connected = NetworkBuilder("connected-affine")
    nodes = connected.population(
        "nodes",
        4,
        LIF(drive=0.0, refractory=1.75),
        parameters={"drive": (20.0, 0.0, 17.0, 0.0)},
        polarity=(
            "excitatory",
            "inhibitory",
            "excitatory",
            "excitatory",
        ),
    )
    connected.connect(nodes[0], nodes[2], weight=4.0, delay=1.25)
    connected.connect(nodes[1], nodes[2], weight=2.5, delay=1.25)
    connected.connect(nodes[2], nodes[3], weight=18.0, delay=0.0)
    connected.connect(nodes[3], nodes[0], weight=1.0, delay=0.5)
    cases.append(
        (
            connected.build().graph.resolve(),
            (
                InputSpike(1.0, 0, 20.0),
                InputSpike(1.0, 1, 20.0),
                InputSpike(3.0, 3, 8.0),
                InputSpike(3.0, 3, 9.0),
            ),
            30.0,
        )
    )

    for resolved, compact_inputs, t_end in cases:
        plan = resolved.execution_plan()
        compact = core.run_affine_delta_plan(
            plan, inputs=compact_inputs, t_end=t_end
        )
        mixed_inputs = tuple(
            MixedInputSpike(spike.t, spike.node, spike.value)
            for spike in compact_inputs
        )
        with core.compile_execution_plan(plan) as compiled:
            shared = compiled.run(
                resolved.initial_values,
                inputs=mixed_inputs,
                t_end=t_end,
            )

        assert shared.spikes == compact.spikes
        assert replace(shared.stats, peak_queue_occupancy=0) == replace(
            compact.stats, peak_queue_occupancy=0
        )
        assert shared.weights == compact.weights
        assert len(shared.states) == len(compact.states)
        for shared_state, compact_state in zip(shared.states, compact.states):
            assert len(shared_state.values) == 1
            assert struct.pack(">d", shared_state.values[0]) == struct.pack(
                ">d", compact_state.value
            )
            assert struct.pack(">d", shared_state.t_last) == struct.pack(
                ">d", compact_state.t_last
            )


def test_plan_runtime_sparse_worklist_preserves_canonical_node_order(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("sparse-worklist-order")
    builder.population("nodes", 4, LIF(drive=0.0))
    resolved = builder.build().graph.resolve()
    reverse_inputs = tuple(
        MixedInputSpike(1.0, node, 20.0) for node in reversed(range(4))
    )

    with core.compile_execution_plan(resolved.execution_plan()) as compiled:
        result = compiled.run(
            resolved.initial_values,
            inputs=reverse_inputs,
            t_end=2.0,
        )

    assert [(spike.t, spike.node) for spike in result.spikes] == [
        (1.0, 0),
        (1.0, 1),
        (1.0, 2),
        (1.0, 3),
    ]


def test_normal_fast_execution_does_not_pack_resolved_model_objects(
    core: CoreEvaluator, monkeypatch
) -> None:
    def reject_legacy_model_packing(*args, **kwargs):
        raise AssertionError("normal execution used legacy model-object packing")

    monkeypatch.setattr(CoreEvaluator, "_model", reject_legacy_model_packing)
    builder = NetworkBuilder()
    builder.population("nodes", 1, LIF(drive=20.0))

    with Engine(core._lib._name).compile(builder.build()) as compiled:
        result = compiled.run(30.0)

    assert result.spikes.events


def test_normal_execution_uses_plan_compiler_by_default(
    core: CoreEvaluator, monkeypatch
) -> None:
    def reject_legacy_compiler(*args, **kwargs):
        raise AssertionError("normal execution used the legacy mixed compiler")

    monkeypatch.setattr(core, "compile_mixed", reject_legacy_compiler)
    builder = NetworkBuilder()
    builder.population("adaptive", 1, AdaptiveLIF(drive=25.0))

    with Engine(core._lib._name).compile(builder.build()) as compiled:
        result = compiled.run(30.0)

    assert compiled.preferred_execution_path == "compiled_sparse"
    assert compiled.last_execution_path == "compiled_sparse"
    assert result.spikes.events
