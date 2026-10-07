from __future__ import annotations

import random
import struct
from collections.abc import Callable
from dataclasses import fields, is_dataclass

import pytest

from lacuna import (
    AdEx,
    AdaptiveLIF,
    AlphaCurrent,
    CrossingMethod,
    CoreEvaluator,
    DepositKind,
    DepositOperation,
    EvolutionMethod,
    ExponentialCurrent,
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    GraphSynapse,
    HeldCurrentEncoder,
    InputMode,
    InputPort,
    IntegrateAndFire,
    LIF,
    ModulatedSTDP,
    ModulationEvent,
    ModulatorPort,
    NativeEventEncoder,
    NetworkBuilder,
    NeuronPolarity,
    OutputPort,
    PairSTDP,
    QIF,
    RateDecoder,
    ReactiveMode,
    TTFSEncoder,
    TTFSDecoder,
    TripletSTDP,
    MixedDriveUpdate,
    MixedInputSpike,
    resolve_learning,
)
from lacuna.ir import (
    ResolvedAdaptiveLIF,
    ResolvedAlphaLIF,
    ResolvedPerEdgeLIF,
    ResolvedReactiveIF,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
)


FOLDED_ALPHA_NEURON = """
neuron FoldedAlphaLIF {
    params {
        tau_m : positive = 10.0
        v_rest = -65.0
        drive = 3.0
        v_th = -50.0
        v_reset = -65.0
    }
    state { v : membrane; i_exc : receptor }
    dynamics { dv/dt = -(v-v_rest)/tau_m + drive/tau_m + i_exc }
    threshold { v > v_th }
    reset { v <- v_reset }
    refractory { 2.0 }
}
"""


FOLDED_ALPHA_SYNAPSE = """
synapse folded_alpha {
    params { tau_s : positive = 5.0 }
    state { s; z }
    dynamics { ds/dt = -s/tau_s + z; dz/dt = -z/tau_s }
    on_spike { z <- z + w/tau_s^2 }
    output { current = s }
}
"""


def _scalar_ports_and_polarity() -> Graph:
    builder = NetworkBuilder("scalar-ports")
    nodes = builder.population(
        "lif",
        2,
        LIF(drive=0.0),
        parameters={"drive": (18.0, 0.0), "tau_m": (10.0, 30.0)},
        initial=(-64.0, -63.0),
        polarity=(NeuronPolarity.EXCITATORY, NeuronPolarity.INHIBITORY),
    )
    builder.connect(nodes[0], nodes[1], weight=2.5, delay=1.25)
    builder.input(
        "encoded-spikes",
        nodes[0],
        encoder=TTFSEncoder(1.0, 6.0, amplitude=4.0),
    )
    builder.input(
        "held-drive",
        nodes[1],
        encoder=HeldCurrentEncoder(gain=4.0, offset=17.0, baseline=18.0),
        parameter="drive",
    )
    builder.output("rate", nodes[0], decoder=RateDecoder())
    builder.output("first", nodes[1], decoder=TTFSDecoder())
    return builder.build().graph


def _adaptive() -> Graph:
    model = AdaptiveLIF(
        drive=26.0,
        tau_m=17.0,
        tau_adaptation=83.0,
        adaptation_increment=1.75,
    )
    return Graph(
        models=(model.model,),
        nodes=(model.node(10, initial=(-61.0, 0.5)),),
    )


def _stepped_mixed() -> Graph:
    adex = AdEx(name="adex", drive=480.0)
    qif = QIF(name="qif", drive=1.2)
    return Graph(
        models=(adex.model, qif.model),
        nodes=(adex.node(20), qif.node(21)),
        edges=(GraphEdge(7, 20, 21, 0.4, 0.3),),
    )


def _reactive(*, leak: bool) -> Graph:
    model = IntegrateAndFire(
        name="leaky_batch" if leak else "held_batch",
        threshold=1.25,
        reset=-0.2,
        refractory=0.5,
        leak=leak,
    )
    return Graph(
        models=(model.model,),
        nodes=(model.node(30, initial=0.1),),
        input_ports=(
            InputPort("events", 30, InputMode.SPIKE, encoder=NativeEventEncoder()),
        ),
    )


def _per_edge_exponential(*, equal_membrane_decay: bool) -> Graph:
    builder = NetworkBuilder("per-edge-exp")
    sources = builder.population("sources", 2, LIF(name="source"))
    target_model = LIF(name="target", tau_m=20.0, synaptic_input=True)
    target = builder.neuron("target", target_model)
    first_tau = 20.0 if equal_membrane_decay else 5.0
    builder.connect(
        sources[0],
        target,
        synapse=ExponentialCurrent(first_tau, name="exp_a"),
        weight=1.5,
        delay=0.25,
    )
    builder.connect(
        sources[1],
        target,
        synapse=ExponentialCurrent(7.0, name="exp_b"),
        weight=2.5,
        delay=0.75,
    )
    return builder.build().graph


def _per_edge_alpha() -> Graph:
    builder = NetworkBuilder("per-edge-alpha")
    source = builder.neuron(
        "source",
        LIF(name="source"),
        polarity=NeuronPolarity.INHIBITORY,
    )
    target = builder.neuron(
        "target", LIF(name="target", synaptic_input=True)
    )
    builder.connect(
        source,
        target,
        synapse=AlphaCurrent(4.0),
        weight=3.0,
        delay=1.0,
    )
    return builder.build().graph


def _folded_alpha() -> Graph:
    return Graph(
        models=(GraphModel("folded", FOLDED_ALPHA_NEURON),),
        synapses=(GraphSynapse("alpha", FOLDED_ALPHA_SYNAPSE),),
        nodes=(
            GraphNode(
                40,
                "folded",
                (-64.0, 0.2, 0.1),
                {"drive": 7.0},
                synapse="alpha",
                receptor="i_exc",
                output="current",
                synapse_bindings={"tau_s": 6.0},
            ),
        ),
    )


def _mixed_intrinsic_families() -> Graph:
    lif = LIF(name="lif", drive=19.0)
    adaptive = AdaptiveLIF(name="adaptive", drive=24.0)
    qif = QIF(name="qif")
    reactive = IntegrateAndFire(name="reactive")
    return Graph(
        models=(lif.model, adaptive.model, qif.model, reactive.model),
        nodes=(
            lif.node(50),
            adaptive.node(51),
            qif.node(52),
            reactive.node(53),
        ),
        edges=(
            GraphEdge(60, 50, 51, 0.5, 0.1),
            GraphEdge(61, 51, 52, 0.6, 0.2),
            GraphEdge(62, 52, 53, 0.7, 0.3),
            GraphEdge(63, 53, 50, 0.8, 0.4),
        ),
    )


def _mixed_plasticity() -> Graph:
    model = LIF(name="plastic_lif")
    pair = PairSTDP(
        tau_pre=4.0,
        tau_post=6.0,
        a_plus=0.4,
        a_minus=0.2,
        learning_rate=0.012,
        bounds=(0.1, 0.9),
    )
    triplet = TripletSTDP.visual_cortex(
        learning_rate=0.02, bounds=(0.05, 0.95)
    )
    modulated = ModulatedSTDP(
        tau_pre=3.0,
        tau_post=8.0,
        tau_eligibility_plus=100.0,
        tau_eligibility_minus=120.0,
        learning_rate=0.03,
        bounds=(0.01, 0.99),
        consume_on_modulation=False,
    )
    return Graph(
        models=(model.model,),
        nodes=tuple(model.node(index) for index in range(6)),
        edges=(
            GraphEdge(70, 0, 1, 0.5, 0.0),
            GraphEdge(71, 0, 2, 0.5, 0.1, plasticity=pair, weight_group=11),
            GraphEdge(72, 0, 3, 0.5, 0.2, plasticity=triplet),
            GraphEdge(73, 0, 4, 0.5, 0.3, plasticity=modulated),
            GraphEdge(74, 0, 5, 0.5, 0.1, plasticity=pair, weight_group=11),
        ),
        modulator_ports=(ModulatorPort("reward", (73,)),),
    )


PLAN_CASES: tuple[tuple[str, Callable[[], Graph]], ...] = (
    ("scalar-parameters-polarity-codecs", _scalar_ports_and_polarity),
    ("adaptive-exact", _adaptive),
    ("adex-and-qif-stepped", _stepped_mixed),
    ("reactive-hold", lambda: _reactive(leak=False)),
    ("reactive-reset-before-deposit", lambda: _reactive(leak=True)),
    ("per-edge-exponential-distinct", lambda: _per_edge_exponential(equal_membrane_decay=False)),
    ("per-edge-exponential-equal-decay", lambda: _per_edge_exponential(equal_membrane_decay=True)),
    ("per-edge-alpha-inhibitory", _per_edge_alpha),
    ("folded-alpha-program-deposit", _folded_alpha),
    ("mixed-intrinsic-families", _mixed_intrinsic_families),
    ("mixed-static-pair-triplet-modulated", _mixed_plasticity),
)


def _initial_values(value: float | tuple[float, ...]) -> tuple[float, ...]:
    return value if isinstance(value, tuple) else (value,)


def _assert_bitwise_equal(actual, expected) -> None:
    """Compare runtime records without allowing any float-bit drift."""

    assert type(actual) is type(expected)
    if isinstance(actual, float):
        assert struct.pack(">d", actual) == struct.pack(">d", expected)
        return
    if is_dataclass(actual):
        for field in fields(actual):
            if not field.compare:
                continue
            _assert_bitwise_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
        return
    if isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_bitwise_equal(actual_item, expected_item)
        return
    assert actual == expected


def _assert_plan_equivalent_to_resolved(graph: Graph) -> None:
    resolved = graph.resolve()
    plan = resolved.execution_plan()

    assert len(plan.nodes) == len(resolved.models)
    assert len(plan.connections) == len(resolved.edges)

    expected_state_index = 0
    for index, (source_id, model, initial, polarity) in enumerate(
        zip(
            resolved.node_ids,
            resolved.models,
            resolved.initial_values,
            resolved.polarities,
        )
    ):
        node = plan.nodes[index]
        program = plan.programs[node.program]
        state_names = (
            tuple(model.state_names)
            if hasattr(model, "state_names")
            else (model.state_name,)
        )
        values = _initial_values(initial)

        assert node.index == index
        assert node.source_id == source_id
        assert node.state_offset == expected_state_index
        assert node.state_count == len(state_names)
        assert node.readout_index == model.readout_index
        assert node.threshold == model.threshold
        assert node.refractory == model.refractory
        assert node.polarity is polarity
        assert node.crossing_hint == getattr(model, "root_hint", None)
        assert node.numerical == getattr(model, "numerical", None)
        assert node.reactive_mode == getattr(model, "reactive_mode", None)
        assert node.autonomous_crossing is (
            model.dispatch.value != "REACTIVE"
        )

        assert program.key == model.resolution_key
        assert program.state_names == state_names
        assert program.propagation == model.propagation_dag
        assert program.normal_roots == model.normal_roots
        assert program.clamped_roots == model.clamped_roots
        assert program.reset_roots == model.reset_roots
        assert program.readout_index == model.readout_index
        assert program.drive_parameters == model.drive_parameters
        assert tuple(binding.name for binding in program.drive_bindings) == tuple(
            model.drive_parameters
        )
        assert tuple(binding.parameter_index for binding in program.drive_bindings) == tuple(
            model.propagation_dag.parameters.index(name)
            for name in model.drive_parameters
        )
        assert tuple(binding.domain for binding in program.drive_bindings) == tuple(
            model.parameter_domains[name] for name in model.drive_parameters
        )
        assert node.parameter_values == tuple(
            float(model.bindings[name]) for name in model.propagation_dag.parameters
        )

        deposit_dag = getattr(model, "deposit_dag", None)
        if deposit_dag is None:
            assert program.deposit is None
            assert node.deposit_parameter_values == ()
        else:
            assert program.deposit is not None
            assert program.deposit.expressions == deposit_dag
            assert program.deposit.root == model.deposit_root
            assert program.deposit.target_state == model.deposit_index
            assert node.deposit_parameter_values == tuple(
                float(model.bindings[name]) for name in deposit_dag.parameters
            )

        slots = plan.states[expected_state_index : expected_state_index + len(values)]
        assert tuple(slot.index for slot in slots) == tuple(
            range(expected_state_index, expected_state_index + len(values))
        )
        assert tuple(slot.node for slot in slots) == (index,) * len(values)
        assert tuple(slot.local_index for slot in slots) == tuple(range(len(values)))
        assert tuple(slot.name for slot in slots) == state_names
        assert tuple(slot.initial for slot in slots) == values
        expected_state_index += len(values)

    modulator_by_edge = {
        edge: modulator
        for modulator, port in enumerate(resolved.graph.modulator_ports)
        for edge in port.edges
    }
    for connection, authored, edge in zip(
        plan.connections, resolved.graph.edges, resolved.edges
    ):
        assert connection.source_id == authored.id
        assert (connection.pre, connection.post) == (edge.pre, edge.post)
        assert connection.weight == edge.weight
        assert connection.delay == edge.delay
        assert connection.target_state == edge.target
        assert connection.scale == edge.deposit_scale
        assert connection.operation is (
            DepositOperation.ADD_STATE
            if edge.deposit_kind.name == "STATE_ADD"
            else DepositOperation.EVALUATE_PROGRAM
        )
        if authored.plasticity is None:
            assert connection.learning_program is None
            assert connection.learning_parameter_values == ()
            assert connection.learning_weight_bounds is None
        else:
            learned = resolve_learning(authored.plasticity)
            assert connection.learning_program is not None
            assert plan.learning_programs[connection.learning_program].key == learned.program.key
            assert connection.learning_parameter_values == learned.parameter_values
            assert connection.learning_weight_bounds == learned.weight_bounds
        assert connection.weight_group == authored.weight_group
        assert connection.modulator == modulator_by_edge.get(authored.id)

    node_index = {source_id: index for index, source_id in enumerate(resolved.node_ids)}
    edge_index = {
        connection.source_id: connection.index for connection in plan.connections
    }
    assert tuple(
        (item.id, item.node, item.mode, item.parameter, item.encoder)
        for item in plan.inputs
    ) == tuple(
        (port.id, node_index[port.node], port.mode, port.parameter, port.encoder)
        for port in resolved.graph.input_ports
    )
    assert tuple((item.id, item.node, item.decoder) for item in plan.outputs) == tuple(
        (port.id, node_index[port.node], port.decoder)
        for port in resolved.graph.output_ports
    )
    assert tuple((item.id, item.connections) for item in plan.modulators) == tuple(
        (port.id, tuple(edge_index[edge] for edge in port.edges))
        for port in resolved.graph.modulator_ports
    )
    assert plan.requirements.learning_connections == sum(
        edge.plasticity is not None for edge in resolved.graph.edges
    )
    assert plan.requirements.modulators == len(resolved.graph.modulator_ports)
    assert plan.requirements.decoders == sum(
        port.decoder is not None for port in resolved.graph.output_ports
    )


@pytest.mark.parametrize(
    ("case_name", "factory"),
    PLAN_CASES,
    ids=tuple(name for name, _ in PLAN_CASES),
)
def test_execution_plan_preserves_every_resolved_runtime_input(
    case_name: str, factory: Callable[[], Graph]
) -> None:
    del case_name
    _assert_plan_equivalent_to_resolved(factory())


def test_matrix_exercises_every_current_neuron_execution_family() -> None:
    model_types = {
        type(model)
        for _, factory in PLAN_CASES
        for model in factory().resolve().models
    }
    assert {
        ResolvedScalarLIF,
        ResolvedAdaptiveLIF,
        ResolvedSteppedNeuron,
        ResolvedReactiveIF,
        ResolvedPerEdgeLIF,
        ResolvedAlphaLIF,
    }.issubset(model_types)


def test_matrix_exercises_every_crossing_and_evolution_strategy() -> None:
    programs = tuple(
        program
        for _, factory in PLAN_CASES
        for program in factory().resolve().execution_plan().programs
    )
    assert {program.evolution for program in programs} == {
        EvolutionMethod.EXACT_EXPRESSIONS,
        EvolutionMethod.EVENT_BATCHED,
        EvolutionMethod.NUMERICAL_ODE,
    }
    assert {
        CrossingMethod.NONE,
        CrossingMethod.SCALAR_LOG,
        CrossingMethod.TWO_EXPONENTIAL,
        CrossingMethod.MULTI_EXPONENTIAL,
        CrossingMethod.EXPONENTIAL_POLYNOMIAL,
        CrossingMethod.NUMERICAL_EVENT,
    }.issubset({program.crossing for program in programs})


def test_reactive_modes_and_folded_deposit_are_not_implicit() -> None:
    held = _reactive(leak=False).resolve().execution_plan().nodes[0]
    reset = _reactive(leak=True).resolve().execution_plan().nodes[0]
    folded = _folded_alpha().resolve().execution_plan()

    assert held.reactive_mode is ReactiveMode.HOLD
    assert reset.reactive_mode is ReactiveMode.RESET_BEFORE_DEPOSIT
    assert folded.programs[folded.nodes[0].program].deposit is not None


def test_batches_separate_parameter_dependent_dispatch_without_duplicating_programs() -> None:
    plan = _scalar_ports_and_polarity().resolve().execution_plan()

    assert len(plan.programs) == 1
    assert tuple(batch.nodes for batch in plan.node_batches) == ((0,), (1,))
    assert tuple(batch.autonomous_crossing for batch in plan.node_batches) == (
        True,
        False,
    )


def test_all_plasticity_parameters_and_modulator_scope_survive_lowering() -> None:
    resolved = _mixed_plasticity().resolve()
    plan = resolved.execution_plan()

    assert tuple(connection.weight_group for connection in plan.connections) == (
        None,
        11,
        None,
        None,
        11,
    )
    assert plan.modulators[0].id == "reward"
    assert plan.modulators[0].connections == (3,)
    assert tuple(
        (
            batch.learning_program,
            batch.shared_weight,
            batch.modulated,
            batch.connections,
        )
        for batch in plan.connection_batches
    ) == (
        (None, False, False, (0,)),
        (0, True, False, (1, 4)),
        (1, False, False, (2,)),
        (2, False, True, (3,)),
    )


def _legacy_compile(core: CoreEvaluator, graph: Graph):
    resolved = graph.resolve()
    modulator_by_edge = {
        edge: index
        for index, port in enumerate(resolved.graph.modulator_ports)
        for edge in port.edges
    }
    return resolved, core.compile_mixed(
        resolved.models,
        edges=resolved.edges,
        polarities=resolved.polarities,
        plasticity=tuple(edge.plasticity for edge in resolved.graph.edges),
        modulators=tuple(
            modulator_by_edge.get(edge.id) for edge in resolved.graph.edges
        ),
        weight_groups=tuple(edge.weight_group for edge in resolved.graph.edges),
    )


def _matrix_inputs(resolved) -> tuple[MixedInputSpike, ...]:
    plan = resolved.execution_plan()
    events = []
    for node in plan.nodes:
        program = plan.programs[node.program]
        if program.deposit is None:
            kind = DepositKind.STATE_ADD
            target = node.readout_index
        else:
            kind = DepositKind.PROGRAM
            target = program.deposit.target_state
        events.append(MixedInputSpike(0.5, node.index, 20.0, kind, target))
        events.append(MixedInputSpike(2.75, node.index, 4.0, kind, target))
    return tuple(events)


@pytest.mark.parametrize(
    ("case_name", "factory"),
    PLAN_CASES,
    ids=tuple(name for name, _ in PLAN_CASES),
)
def test_plan_compiler_is_behaviorally_identical_to_legacy_compiler(
    core: CoreEvaluator,
    case_name: str,
    factory: Callable[[], Graph],
) -> None:
    del case_name
    resolved, legacy = _legacy_compile(core, factory())
    plan = resolved.execution_plan()
    planned = core.compile_execution_plan(plan)
    inputs = _matrix_inputs(resolved)
    modulations = tuple(
        ModulationEvent(1.5, index, 0.75)
        for index in range(len(plan.modulators))
    )
    try:
        expected = legacy.run(
            resolved.initial_values,
            inputs=inputs,
            modulations=modulations,
            t_end=5.0,
        )
        actual = planned.run(
            resolved.initial_values,
            inputs=inputs,
            modulations=modulations,
            t_end=5.0,
        )
    finally:
        legacy.close()
        planned.close()

    _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize(
    ("case_name", "factory"),
    PLAN_CASES,
    ids=tuple(name for name, _ in PLAN_CASES),
)
def test_compiled_graph_image_preserves_execution_plan_matrix(
    core: CoreEvaluator,
    case_name: str,
    factory: Callable[[], Graph],
) -> None:
    del case_name
    resolved = factory().resolve()
    plan = resolved.execution_plan()
    compiled = core.compile_execution_plan(plan)
    loaded = core.load_compiled_graph_image(
        compiled.to_bytes(), execution_plan=plan
    )
    inputs = _matrix_inputs(resolved)
    modulations = tuple(
        ModulationEvent(1.5, index, 0.75)
        for index in range(len(plan.modulators))
    )
    try:
        expected = compiled.run(
            resolved.initial_values,
            inputs=inputs,
            modulations=modulations,
            t_end=5.0,
        )
        actual = loaded.run(
            resolved.initial_values,
            inputs=inputs,
            modulations=modulations,
            t_end=5.0,
        )
    finally:
        compiled.close()
        loaded.close()

    _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize("consume", (False, True))
@pytest.mark.parametrize("seed", range(5))
def test_generic_learning_executor_matches_legacy_across_event_streams(
    core: CoreEvaluator,
    consume: bool,
    seed: int,
) -> None:
    builder = NetworkBuilder(f"learning-differential-{consume}-{seed}")
    pre = builder.neuron("pre", LIF(drive=0.0))
    posts = builder.population("post", 3, LIF(drive=0.0))
    builder.connect(
        pre,
        posts[0],
        weight=0.5,
        delay=0.0,
        plasticity=PairSTDP(
            tau_pre=3.5,
            tau_post=7.25,
            a_plus=0.23,
            a_minus=0.17,
            learning_rate=0.031,
        ),
    )
    builder.connect(
        pre,
        posts[1],
        weight=0.5,
        delay=0.07,
        plasticity=TripletSTDP(
            tau_plus=4.0,
            tau_minus=6.0,
            tau_x=13.0,
            tau_y=17.0,
            a2_plus=0.11,
            a2_minus=0.09,
            a3_plus=0.07,
            a3_minus=0.05,
            learning_rate=0.019,
        ),
    )
    modulated_edge = builder.connect(
        pre,
        posts[2],
        weight=0.5,
        delay=0.13,
        plasticity=ModulatedSTDP(
            tau_pre=5.0,
            tau_post=8.0,
            tau_eligibility_plus=19.0,
            tau_eligibility_minus=23.0,
            positive_plus=0.8,
            positive_minus=-0.4,
            negative_plus=-0.6,
            negative_minus=0.3,
            learning_rate=0.027,
            consume_on_modulation=consume,
        ),
    )[0]
    builder.modulator("reward", targets=(modulated_edge,))
    graph = builder.build().graph
    resolved, legacy = _legacy_compile(core, graph)
    planned = core.compile_execution_plan(resolved.execution_plan())
    rng = random.Random(seed)
    inputs = [
        MixedInputSpike(1.5, node, 20.0, DepositKind.STATE_ADD, 0)
        for node in range(4)
    ]
    for node in range(4):
        inputs.extend(
            MixedInputSpike(
                round(rng.uniform(0.1, 3.4), 6),
                node,
                20.0,
                DepositKind.STATE_ADD,
                0,
            )
            for _ in range(8)
        )
    modulations = (
        ModulationEvent(0.75, 0, 0.4),
        ModulationEvent(1.5, 0, -0.6),
        ModulationEvent(2.25, 0, 0.9),
        ModulationEvent(3.0, 0, -0.2),
    )
    try:
        expected = legacy.run(
            resolved.initial_values,
            inputs=tuple(inputs),
            modulations=modulations,
            t_end=4.0,
        )
        actual = planned.run(
            resolved.initial_values,
            inputs=tuple(inputs),
            modulations=modulations,
            t_end=4.0,
        )
    finally:
        legacy.close()
        planned.close()

    _assert_bitwise_equal(actual, expected)


def test_program_batches_preserve_mixed_fan_in_and_modulator_semantics(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("mixed-learning-batches")
    pres = builder.population("pre", 4, LIF(drive=0.0))
    post = builder.neuron("post", LIF(drive=0.0))
    builder.connect(pres[0], post, weight=0.5, plasticity=PairSTDP())
    builder.connect(
        pres[1], post, weight=0.5, plasticity=TripletSTDP.hippocampus()
    )
    retained = builder.connect(
        pres[2],
        post,
        weight=0.5,
        plasticity=ModulatedSTDP(consume_on_modulation=False),
    )[0]
    consumed = builder.connect(
        pres[3],
        post,
        weight=0.5,
        plasticity=ModulatedSTDP(consume_on_modulation=True),
    )[0]
    builder.modulator("reward", targets=(retained, consumed))
    resolved, legacy = _legacy_compile(core, builder.build().graph)
    planned = core.compile_execution_plan(resolved.execution_plan())
    inputs = tuple(
        MixedInputSpike(t, node, 20.0, DepositKind.STATE_ADD, 0)
        for t, node in (
            (0.5, 0),
            (0.6, 1),
            (0.7, 2),
            (0.8, 3),
            (1.5, 4),
            (2.0, 0),
            (2.1, 1),
            (2.2, 2),
            (2.3, 3),
            (3.0, 4),
        )
    )
    modulations = (
        ModulationEvent(1.75, 0, 0.8),
        ModulationEvent(3.25, 0, -0.4),
    )
    try:
        expected = legacy.run(
            resolved.initial_values,
            inputs=inputs,
            modulations=modulations,
            t_end=4.0,
        )
        actual = planned.run(
            resolved.initial_values,
            inputs=inputs,
            modulations=modulations,
            t_end=4.0,
        )
    finally:
        legacy.close()
        planned.close()

    _assert_bitwise_equal(actual, expected)


def test_plan_compiler_matches_runtime_drive_parameter_updates(
    core: CoreEvaluator,
) -> None:
    resolved, legacy = _legacy_compile(core, _scalar_ports_and_polarity())
    plan = resolved.execution_plan()
    planned = core.compile_execution_plan(plan)
    updates = (
        MixedDriveUpdate(0.75, 0, 22.0, "drive"),
        MixedDriveUpdate(2.5, 1, 16.0, "drive"),
    )
    try:
        expected = legacy.run(
            resolved.initial_values,
            drive_updates=updates,
            t_end=8.0,
        )
        actual = planned.run(
            resolved.initial_values,
            drive_updates=updates,
            t_end=8.0,
        )
    finally:
        legacy.close()
        planned.close()

    _assert_bitwise_equal(actual, expected)


def test_plan_compiler_matches_prefixed_filtered_node_drive_updates(
    core: CoreEvaluator,
) -> None:
    resolved, legacy = _legacy_compile(
        core, _per_edge_exponential(equal_membrane_decay=False)
    )
    plan = resolved.execution_plan()
    planned = core.compile_execution_plan(plan)
    updates = (MixedDriveUpdate(0.75, 2, 24.0, "drive"),)
    try:
        expected = legacy.run(
            resolved.initial_values,
            drive_updates=updates,
            t_end=8.0,
        )
        actual = planned.run(
            resolved.initial_values,
            drive_updates=updates,
            t_end=8.0,
        )
    finally:
        legacy.close()
        planned.close()

    _assert_bitwise_equal(actual, expected)


def test_plan_compiled_graph_retains_no_resolved_model_records(
    core: CoreEvaluator,
) -> None:
    resolved = _adaptive().resolve()
    compiled = core.compile_execution_plan(resolved.execution_plan())
    try:
        assert compiled.models == ()
        assert compiled.node_count == 1
        assert compiled.runtime_layouts[0].state_count == 2
        with pytest.raises(ValueError, match="must contain 2 value"):
            compiled.run(((-65.0,),), t_end=1.0)
    finally:
        compiled.close()
