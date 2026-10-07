from __future__ import annotations

from dataclasses import replace

import pytest

from lacuna import (
    DepositKind,
    MixedDriveUpdate,
    MixedEdge,
    MixedInputSpike,
    RecordingConfig,
    SpikeInput,
    StateInspectionRequest,
    TraceKind,
    TracePhase,
    parse_neuron,
    resolve_scalar_lif,
)
from lacuna.ffi import CoreEvaluator
from lacuna.errors import CoreError

from .test_alpha import _driven_model, _model
from .test_dsl_resolver import LIF
from .test_mixed_graph import _mixed_graph


def _reactive():
    return resolve_scalar_lif(parse_neuron(LIF), {"drive": 0.0})


def _network():
    models = (_reactive(), _model(), _reactive())
    initial = (-65.0, (-65.0, 0.0, 0.0), -65.0)
    edges = (
        MixedEdge(0, 1, 40.0, 1.0, DepositKind.FOLDED_ALPHA, 2),
        MixedEdge(1, 2, 20.0, 1.0),
    )
    inputs = (MixedInputSpike(1.0, 0, 20.0),)
    return models, initial, edges, inputs


def test_compiled_graph_matches_temporary_compatibility_path(
    core: CoreEvaluator,
) -> None:
    models, initial, edges, inputs = _network()
    expected = core.run_mixed(
        models, initial, edges=edges, inputs=inputs, t_end=14.0
    )
    with core.compile_mixed(models, edges=edges) as compiled:
        actual = compiled.run(initial, inputs=inputs, t_end=14.0)
    assert actual == expected


def test_equal_delay_groups_preserve_per_edge_results_stats_and_trace_order(
    core: CoreEvaluator,
) -> None:
    models = tuple(_reactive() for _ in range(5))
    initial = (-65.0,) * len(models)
    edges = (
        MixedEdge(0, 1, 20.0, 1.0),
        MixedEdge(0, 2, 20.0, 2.0),
        MixedEdge(0, 3, 20.0, 1.0),
        MixedEdge(0, 4, 20.0, 1.0),
        MixedEdge(0, 1, 1.0, 2.0),
    )
    inputs = (MixedInputSpike(1.0, 0, 20.0),)
    recording = RecordingConfig(
        kinds=frozenset({TraceKind.DELIVERY}),
        capture_state=False,
        capacity=len(edges),
    )
    expected = core.run_mixed(
        models,
        initial,
        edges=edges,
        inputs=inputs,
        t_end=4.0,
        queue_capacity=16,
        recording=recording,
    )
    with core.compile_mixed(models, edges=edges) as compiled:
        actual = compiled.run(
            initial,
            inputs=inputs,
            t_end=4.0,
            queue_capacity=16,
            recording=recording,
        )

    assert actual == expected
    assert [(record.t, record.subject) for record in actual.trace] == [
        (2.0, 0),
        (2.0, 2),
        (2.0, 3),
        (3.0, 1),
        (3.0, 4),
    ]


def test_equal_delay_group_overflow_reports_first_logically_rejected_edge(
    core: CoreEvaluator,
) -> None:
    models = tuple(_reactive() for _ in range(5))
    edges = tuple(MixedEdge(0, post, 20.0, 1.0) for post in range(1, 5))
    with core.compile_mixed(models, edges=edges) as compiled:
        with pytest.raises(
            CoreError, match="event queue capacity exceeded"
        ) as captured:
            compiled.run(
                (-65.0,) * len(models),
                inputs=(MixedInputSpike(1.0, 0, 20.0),),
                t_end=3.0,
                queue_capacity=2,
            )

    diagnostic = captured.value.diagnostic
    assert captured.value.status == 6
    assert diagnostic is not None
    assert diagnostic.resource == "queue"
    assert (diagnostic.capacity, diagnostic.occupancy, diagnostic.peak) == (2, 2, 2)
    assert diagnostic.event_kind == "DELIVERY"
    assert diagnostic.event_phase == "DEPOSIT"
    assert diagnostic.event_index == 2
    assert diagnostic.node == 3
    assert diagnostic.t == 2.0


def test_run_session_requires_reset_and_replays_exactly(core: CoreEvaluator) -> None:
    models, initial, edges, inputs = _network()
    with core.compile_mixed(models, edges=edges) as compiled:
        with compiled.create_run(initial) as run:
            first = run.execute(inputs=inputs, t_end=14.0)
            with pytest.raises(RuntimeError, match="must be reset"):
                run.execute(inputs=inputs, t_end=14.0)
            run.reset(initial)
            second = run.execute(inputs=inputs, t_end=14.0)
    assert second == first


def test_incremental_run_matches_one_shot_across_open_boundaries(
    core: CoreEvaluator,
) -> None:
    models, initial, edges, inputs = _network()
    recording = RecordingConfig(capacity=128)
    with core.compile_mixed(models, edges=edges) as compiled:
        expected = compiled.run(
            initial,
            inputs=inputs,
            t_end=14.0,
            recording=recording,
        )
        with compiled.create_incremental_run(
            initial, t_end=14.0
        ) as run:
            first = run.advance_until(1.0, recording=recording)
            second = run.advance_until(
                2.0, inputs=inputs, recording=recording
            )
            third = run.advance_until(8.0, recording=recording)
            final = run.finish(recording=recording)
            cumulative_stats = run.cumulative_stats

    assert first.spikes == ()
    assert second.spikes == (expected.spikes[0],)
    assert second.states[0].t_last == 2.0
    for actual_state, expected_state in zip(final.states, expected.states):
        assert actual_state.t_last == expected_state.t_last
        assert actual_state.values == pytest.approx(expected_state.values, abs=1e-14)
    assert first.spikes + second.spikes + third.spikes + final.spikes == expected.spikes
    split_trace = first.trace + second.trace + third.trace + final.trace
    assert len(split_trace) == len(expected.trace)
    for actual_record, expected_record in zip(split_trace, expected.trace):
        assert replace(actual_record, before=(), after=()) == replace(
            expected_record, before=(), after=()
        )
        assert actual_record.before == pytest.approx(expected_record.before, abs=1e-14)
        assert actual_record.after == pytest.approx(expected_record.after, abs=1e-14)
    assert [record.sequence for record in expected.trace] == list(
        range(len(expected.trace))
    )
    assert replace(cumulative_stats, peak_queue_occupancy=0) == replace(
        expected.stats, peak_queue_occupancy=0
    )


def test_incremental_boundary_remains_open_for_same_time_input_and_cascade(
    core: CoreEvaluator,
) -> None:
    models = (_reactive(), _reactive(), _reactive())
    edges = (
        MixedEdge(0, 1, 20.0),
        MixedEdge(1, 2, 20.0),
    )
    boundary_input = (MixedInputSpike(1.0, 0, 20.0),)
    with core.compile_mixed(models, edges=edges) as compiled:
        with compiled.create_incremental_run(
            (-65.0, -65.0, -65.0), t_end=2.0
        ) as run:
            before = run.advance_until(1.0)
            after = run.advance_until(
                2.0,
                inputs=boundary_input,
                inspections=(StateInspectionRequest(1.0, 2),),
            )
            final = run.finish()
            with pytest.raises(RuntimeError, match="finished"):
                run.advance_until(2.0)

    assert before.spikes == ()
    assert [(spike.t, spike.node) for spike in after.spikes] == [
        (1.0, 0),
        (1.0, 1),
        (1.0, 2),
    ]
    assert final.spikes == ()
    assert after.inspections[0].t == 1.0
    assert after.inspections[0].clamped


def test_incremental_run_rejects_late_raw_input(core: CoreEvaluator) -> None:
    with core.compile_mixed((_reactive(),)) as compiled:
        with compiled.create_incremental_run((-65.0,), t_end=3.0) as run:
            run.advance_until(2.0)
            with pytest.raises(ValueError, match="outside this open interval"):
                run.advance_until(
                    3.0, inputs=(MixedInputSpike(1.5, 0, 20.0),)
                )


def test_incremental_same_time_drive_precedes_retained_refractory_release(
    core: CoreEvaluator,
) -> None:
    model = _reactive()
    inputs = (MixedInputSpike(1.0, 0, 20.0),)
    drives = (MixedDriveUpdate(3.0, 0, 1.0, "drive"),)
    recording = RecordingConfig(
        kinds=frozenset(
            {TraceKind.DRIVE_UPDATE, TraceKind.REFRACTORY_RELEASE}
        ),
        capture_state=False,
        capacity=4,
    )
    with core.compile_mixed((model,)) as compiled:
        expected = compiled.run(
            (-65.0,),
            inputs=inputs,
            drive_updates=drives,
            t_end=4.0,
            recording=recording,
        )
        with compiled.create_incremental_run((-65.0,), t_end=4.0) as run:
            first = run.advance_until(
                3.0, inputs=inputs, recording=recording
            )
            final = run.finish(drive_updates=drives, recording=recording)

    assert first.trace + final.trace == expected.trace
    assert [record.kind for record in final.trace] == [
        TraceKind.DRIVE_UPDATE,
        TraceKind.REFRACTORY_RELEASE,
    ]


def test_incremental_execution_failure_poisons_the_session(
    core: CoreEvaluator,
) -> None:
    models = (_reactive(), _reactive(), _reactive())
    edges = (MixedEdge(0, 1, 20.0), MixedEdge(1, 2, 20.0))
    with core.compile_mixed(models, edges=edges) as compiled:
        with compiled.create_incremental_run(
            (-65.0, -65.0, -65.0),
            t_end=2.0,
            output_capacity=1,
        ) as run:
            with pytest.raises(
                CoreError, match="output spike capacity exceeded"
            ) as captured:
                run.advance_until(
                    2.0, inputs=(MixedInputSpike(1.0, 0, 20.0),)
                )
            diagnostic = captured.value.diagnostic
            assert captured.value.status == 7
            assert diagnostic is not None
            assert diagnostic.resource == "output"
            assert (diagnostic.capacity, diagnostic.occupancy, diagnostic.peak) == (
                1,
                1,
                1,
            )
            assert diagnostic.event_kind == "OUTPUT_SPIKE"
            assert diagnostic.event_phase == "PREDICTION"
            assert diagnostic.event_index == 1
            assert diagnostic.node == 1
            assert diagnostic.t == 1.0
            with pytest.raises(RuntimeError, match="failed and cannot resume"):
                run.finish()


def test_ordered_inputs_do_not_consume_compiled_heap_capacity(
    core: CoreEvaluator,
) -> None:
    with core.compile_mixed((_reactive(),)) as compiled:
        result = compiled.run(
            (-65.0,),
            inputs=(
                MixedInputSpike(1.0, 0, -1.0),
                MixedInputSpike(1.5, 0, -1.0),
            ),
            t_end=2.0,
            queue_capacity=1,
        )

    assert result.stats.input_spikes_processed == 2
    assert result.stats.peak_queue_occupancy == 0


def test_ordered_incremental_inputs_do_not_consume_heap_capacity(
    core: CoreEvaluator,
) -> None:
    with core.compile_mixed((_reactive(),)) as compiled:
        with compiled.create_incremental_run(
            (-65.0,), t_end=2.0, queue_capacity=1
        ) as run:
            part = run.advance_until(
                2.0,
                inputs=(
                    MixedInputSpike(1.0, 0, -1.0),
                    MixedInputSpike(1.5, 0, -1.0),
                ),
            )
            run.finish()

    assert part.stats.input_spikes_processed == 2
    assert part.stats.peak_queue_occupancy == 0


def test_unsorted_low_level_inputs_retain_the_heap_fallback(
    core: CoreEvaluator,
) -> None:
    with core.compile_mixed((_reactive(),)) as compiled:
        with pytest.raises(CoreError, match="event queue capacity exceeded"):
            compiled.run(
                (-65.0,),
                inputs=(
                    MixedInputSpike(1.5, 0, -1.0),
                    MixedInputSpike(1.0, 0, -1.0),
                ),
                t_end=2.0,
                queue_capacity=1,
            )


def test_reset_restores_compiled_default_drive_parameters(core: CoreEvaluator) -> None:
    model = _driven_model()
    initial = ((-65.0, 0.0, 0.0),)
    with core.compile_mixed((model,)) as compiled:
        with compiled.create_run(initial) as run:
            suppressed = run.execute(
                drive_updates=(MixedDriveUpdate(5.0, 0, 0.0, "drive"),),
                t_end=20.0,
            )
            run.reset(initial)
            restored = run.execute(t_end=20.0)
    assert suppressed.spikes == ()
    assert restored.spikes


def test_independent_sessions_outlive_the_public_graph_owner(
    core: CoreEvaluator,
) -> None:
    models, initial, edges, inputs = _network()
    compiled = core.compile_mixed(models, edges=edges)
    first_run = compiled.create_run(initial)
    second_run = compiled.create_run(initial)
    compiled.close()
    try:
        first = first_run.execute(inputs=inputs, t_end=14.0)
        second = second_run.execute(inputs=inputs, t_end=14.0)
    finally:
        first_run.close()
        second_run.close()
    assert first == second


def test_resolved_graph_compilation_preserves_named_ports(core: CoreEvaluator) -> None:
    resolved = _mixed_graph().resolve()
    inputs = [
        # The graph-level wrapper continues to accept port names, not node indices.
        SpikeInput(1.0, "stimulus", 20.0)
    ]
    expected = resolved.run(core, spike_inputs=inputs, t_end=12.0)
    with resolved.compile(core) as compiled:
        actual = compiled.run(spike_inputs=inputs, t_end=12.0)
        replay = compiled.run(spike_inputs=inputs, t_end=12.0)
    assert actual == expected
    assert replay == actual


def test_compiled_resolved_graph_exposes_named_incremental_raw_events(
    core: CoreEvaluator,
) -> None:
    resolved = _mixed_graph().resolve()
    stimulus = (SpikeInput(1.0, "stimulus", 20.0),)
    expected = resolved.run(core, spike_inputs=stimulus, t_end=12.0)
    with resolved.compile(core) as compiled:
        with compiled.create_incremental_run(t_end=12.0) as run:
            first = run.advance_until(1.0)
            second = run.advance_until(8.0, spike_inputs=stimulus)
            final = run.finish()

    assert first.outputs == second.outputs == ()
    assert final.outputs == expected.outputs
    for actual_state, expected_state in zip(
        final.core.states, expected.core.states
    ):
        assert actual_state.values == pytest.approx(expected_state.values, abs=1e-14)
        assert actual_state.t_last == expected_state.t_last


def test_buffered_trace_filters_nodes_and_captures_atomic_reset(
    core: CoreEvaluator,
) -> None:
    models, initial, edges, inputs = _network()
    recording = RecordingConfig(
        kinds=frozenset(
            {
                TraceKind.DELIVERY,
                TraceKind.DEPOSIT_APPLY,
                TraceKind.SPIKE,
                TraceKind.RESET,
                TraceKind.FINAL_STATE,
            }
        ),
        nodes=(1,),
        state_indices=(0, 2),
        capacity=8,
    )
    result = core.run_mixed(
        models,
        initial,
        edges=edges,
        inputs=inputs,
        t_end=14.0,
        recording=recording,
    )
    assert [record.sequence for record in result.trace] == list(range(len(result.trace)))
    assert {record.node for record in result.trace} == {1}
    assert [record.kind for record in result.trace] == [
        TraceKind.DELIVERY,
        TraceKind.DEPOSIT_APPLY,
        TraceKind.SPIKE,
        TraceKind.RESET,
        TraceKind.FINAL_STATE,
    ]
    reset = result.trace[3]
    assert reset.phase is TracePhase.FIRE
    assert reset.state_indices == (0, 2)
    assert len(reset.before) == len(reset.after) == 2
    assert reset.before[0] >= models[1].threshold
    assert reset.after[0] == models[1].reset
    assert reset.after[1] == pytest.approx(reset.before[1])


def test_trace_streams_without_retaining_records(core: CoreEvaluator) -> None:
    models, initial, edges, inputs = _network()
    streamed = []
    result = core.run_mixed(
        models,
        initial,
        edges=edges,
        inputs=inputs,
        t_end=14.0,
        recording=RecordingConfig(
            kinds=frozenset({TraceKind.SPIKE}),
            capture_state=False,
            consumer=streamed.append,
        ),
    )
    assert result.trace == ()
    assert [(record.node, record.t) for record in streamed] == [
        (spike.node, spike.t) for spike in result.spikes
    ]
    assert all(record.before == record.after == () for record in streamed)


def test_trace_capacity_overflow_is_explicit(core: CoreEvaluator) -> None:
    models, initial, edges, inputs = _network()
    with pytest.raises(CoreError, match="trace record capacity exceeded") as captured:
        core.run_mixed(
            models,
            initial,
            edges=edges,
            inputs=inputs,
            t_end=14.0,
            recording=RecordingConfig(
                kinds=frozenset({TraceKind.FINAL_STATE}),
                capacity=2,
            ),
        )
    diagnostic = captured.value.diagnostic
    assert captured.value.status == 12
    assert diagnostic is not None
    assert diagnostic.resource == "trace"
    assert (diagnostic.capacity, diagnostic.occupancy, diagnostic.peak) == (2, 2, 2)
    assert diagnostic.event_kind == "TRACE_FINAL_STATE"
    assert diagnostic.event_phase == "FINAL"
    assert diagnostic.event_index == TraceKind.FINAL_STATE
    assert diagnostic.node == 2
    assert diagnostic.t == 14.0


def test_trace_retains_a_zero_net_deposit_batch(core: CoreEvaluator) -> None:
    result = core.run_mixed(
        (_reactive(),),
        (-65.0,),
        inputs=(
            MixedInputSpike(1.0, 0, 5.0),
            MixedInputSpike(1.0, 0, -5.0),
        ),
        t_end=2.0,
        recording=RecordingConfig(
            kinds=frozenset(
                {TraceKind.INPUT_SPIKE, TraceKind.DEPOSIT_APPLY}
            ),
            capacity=3,
        ),
    )
    assert [record.kind for record in result.trace] == [
        TraceKind.INPUT_SPIKE,
        TraceKind.INPUT_SPIKE,
        TraceKind.DEPOSIT_APPLY,
    ]
    assert result.trace[-1].before == result.trace[-1].after == (-65.0,)


def test_explicit_time_inspection_is_post_event_and_read_only(
    core: CoreEvaluator,
) -> None:
    model = _reactive()
    inputs = (MixedInputSpike(1.0, 0, 20.0),)
    requests = (
        StateInspectionRequest(3.0, 0),
        StateInspectionRequest(0.5, 0),
        StateInspectionRequest(1.0, 0),
        StateInspectionRequest(2.0, 0),
    )
    with core.compile_mixed((model,)) as compiled:
        baseline = compiled.run((-65.0,), inputs=inputs, t_end=3.0)
        observed = compiled.run(
            (-65.0,), inputs=inputs, t_end=3.0, inspections=requests
        )

    assert [item.t for item in observed.inspections] == [3.0, 0.5, 1.0, 2.0]
    assert [item.values for item in observed.inspections] == [
        (-65.0,),
        (-65.0,),
        (-65.0,),
        (-65.0,),
    ]
    assert [item.clamped for item in observed.inspections] == [
        False,
        False,
        True,
        True,
    ]
    assert replace(observed, inspections=()) == baseline


def test_vector_inspection_selects_local_states_and_coexists_with_trace(
    core: CoreEvaluator,
) -> None:
    models, initial, edges, inputs = _network()
    result = core.run_mixed(
        models,
        initial,
        edges=edges,
        inputs=inputs,
        t_end=3.0,
        inspections=(
            StateInspectionRequest(2.0, 1, (2, 0)),
            StateInspectionRequest(3.0, 1),
        ),
        recording=RecordingConfig(
            kinds=frozenset({TraceKind.DELIVERY, TraceKind.FINAL_STATE}),
            capacity=4,
        ),
    )

    deposit = result.inspections[0]
    assert deposit.state_indices == (2, 0)
    assert deposit.values == pytest.approx((1.6, -65.0), abs=1e-15)
    assert result.inspections[1].values == pytest.approx(
        result.states[1].values, abs=1e-15
    )
    assert result.trace


def test_same_time_inspection_waits_for_zero_delay_cascade(
    core: CoreEvaluator,
) -> None:
    models = (_reactive(), _reactive(), _reactive())
    result = core.run_mixed(
        models,
        (-65.0, -65.0, -65.0),
        edges=(
            MixedEdge(0, 1, 20.0),
            MixedEdge(1, 2, 20.0),
        ),
        inputs=(MixedInputSpike(1.0, 0, 20.0),),
        t_end=1.0,
        inspections=tuple(
            StateInspectionRequest(1.0, node) for node in (2, 0, 1)
        ),
    )
    assert [(spike.t, spike.node) for spike in result.spikes] == [
        (1.0, 0),
        (1.0, 1),
        (1.0, 2),
    ]
    assert [item.node for item in result.inspections] == [2, 0, 1]
    assert all(item.values == (-65.0,) for item in result.inspections)
    assert all(item.clamped for item in result.inspections)


@pytest.mark.parametrize(
    ("inspection_request", "message"),
    (
        (StateInspectionRequest(4.0, 0), "inside the run"),
        (StateInspectionRequest(1.0, 99), "invalid node"),
        (StateInspectionRequest(1.0, 0, (0, 0)), "unique valid"),
    ),
)
def test_invalid_state_inspection_request_is_rejected(
    core: CoreEvaluator,
    inspection_request: StateInspectionRequest,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        core.run_mixed(
            (_reactive(),),
            (-65.0,),
            t_end=3.0,
            inspections=(inspection_request,),
        )


def test_compiled_graph_image_round_trip_matches_original(
    core: CoreEvaluator,
) -> None:
    models, initial, edges, inputs = _network()
    with core.compile_mixed(models, edges=edges) as compiled:
        expected = compiled.run(initial, inputs=inputs, t_end=14.0)
        first = compiled.to_bytes()
        second = compiled.to_bytes()

    assert first == second
    with core.load_compiled_graph_image(first) as loaded:
        assert loaded.node_count == 3
        assert loaded.state_count == 5
        assert loaded.edge_count == 2
        actual = loaded.run(initial, inputs=inputs, t_end=14.0)
    assert actual == expected


def test_compiled_graph_image_file_is_independent_of_source_buffer(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    model = _reactive()
    destination = tmp_path / "reactive.lcg"
    with core.compile_mixed((model,)) as compiled:
        assert compiled.save_image(destination) == destination

    with core.load_compiled_graph_file(destination) as loaded:
        destination.unlink()
        result = loaded.run(
            (-65.0,),
            inputs=(MixedInputSpike(1.0, 0, 20.0),),
            t_end=2.0,
        )
    assert [(spike.t, spike.node) for spike in result.spikes] == [(1.0, 0)]


def test_compiled_graph_image_rejects_corruption(core: CoreEvaluator) -> None:
    with core.compile_mixed((_reactive(),)) as compiled:
        damaged = bytearray(compiled.to_bytes())
    damaged[-1] ^= 1

    with pytest.raises(CoreError, match="checksum mismatch"):
        core.load_compiled_graph_image(damaged)


def test_equation_learning_graph_image_preserves_plastic_execution(
    core: CoreEvaluator,
) -> None:
    from lacuna import LIF, NetworkBuilder, PairSTDP

    builder = NetworkBuilder("learning-image")
    pre = builder.neuron("pre", LIF(name="pre"))
    post = builder.neuron("post", LIF(name="post"))
    builder.connect(pre, post, weight=0.5, plasticity=PairSTDP())
    plan = builder.build().graph.resolve().execution_plan()
    initial = (-65.0, -65.0)
    inputs = (
        MixedInputSpike(1.0, 0, 20.0),
        MixedInputSpike(2.0, 1, 20.0),
    )
    with core.compile_execution_plan(plan) as compiled:
        expected = compiled.run(initial, inputs=inputs, t_end=3.0)
        image = compiled.to_bytes()
    with core.load_compiled_graph_image(
        image, execution_plan=plan
    ) as loaded:
        actual = loaded.run(initial, inputs=inputs, t_end=3.0)

    assert actual == expected
