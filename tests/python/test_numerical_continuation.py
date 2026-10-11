"""Numerical continuation is a solver event, never a neuronal spike."""
import math

import pytest

from lacuna import (AugmentedState, RecordingConfig, StateInspectionRequest,
                    TraceKind, audit_causal_trace, parse_neuron, resolve_stepped_neuron)
from lacuna.ffi import MixedInputSpike
from lacuna.errors import CoreError
from lacuna.ir import NumericalConfig
from .test_stepped import QIF, ADEX


def stepped(source=QIF, **settings):
    return resolve_stepped_neuron(parse_neuron(source), numerical=NumericalConfig(**settings))


def test_silent_continuations_are_accounted_but_not_spikes(core):
    source = QIF.replace('v*v + drive', '-v').replace('    drive = 1\n', '')
    model = stepped(source.replace('threshold = 1', 'threshold = 10'))
    with core.compile_mixed([model]) as graph:
        result = graph.run([(0.,)], t_end=100., recording=RecordingConfig(capacity=1000))
    assert not result.spikes
    assert result.stats.autonomous_spikes_confirmed == 0
    wakes = [r for r in result.trace if r.kind == TraceKind.NUMERICAL_CONTINUATION]
    assert len(wakes) >= 399
    assert all(b.t > a.t for a, b in zip(wakes, wakes[1:]))
    audit_causal_trace(result, models=[model], edges=[], t_end=100.)


def test_autonomous_crossing_is_not_lost_after_many_empty_windows(core):
    model = stepped(QIF.replace('drive = 1', 'drive = 0.01'))
    expected = math.atan(10.) / .1
    result = core.run_mixed([model], [(0.,)], t_end=30.)
    assert [s.t for s in result.spikes] == pytest.approx([expected, 2*expected], abs=1e-6)


@pytest.mark.parametrize('when', [.013, .249, .25, .251, .5])
def test_inputs_invalidate_cached_future_including_window_boundary(core, when):
    model = stepped()
    inputs = [MixedInputSpike(when, 0, -.1), MixedInputSpike(when, 0, .04)]
    # Preserve the actual binary64 sum: -.1 + .04 is not bit-identical to -.06.
    total = -.1 + .04
    before, _ = core.advance_stepped(model, AugmentedState((0.,), 0.), when)
    state = AugmentedState((before.values[0] + total,), when)
    expected = core.predict_stepped(model, state, 1.).t_spike
    with core.compile_mixed([model]) as graph:
        combined = graph.run([(0.,)], inputs=inputs, t_end=1.)
        aggregated = graph.run([(0.,)], inputs=[MixedInputSpike(when, 0, total)], t_end=1.)
    assert len(combined.spikes) == 1
    assert combined.spikes[0].t == pytest.approx(expected, abs=5e-8)
    assert combined.spikes == aggregated.spikes


def test_cached_observations_and_incremental_frontiers_do_not_change_trajectory(core):
    model = stepped(ADEX)
    inputs = [MixedInputSpike(t, 0, v) for t,v in ((.25,2.),(1.013,-1.),(14.,3.),(16.,-2.))]
    requests = [StateInspectionRequest(t,0) for t in (.013,.25,.333,1.013,14.,16.,19.999)]
    with core.compile_mixed([model]) as graph:
        expected = graph.run([(-70.,0.)], inputs=inputs, t_end=20.)
        observed = graph.run([(-70.,0.)], inputs=inputs, t_end=20., inspections=requests)
        with graph.create_incremental_run([(-70.,0.)], t_end=20.) as run:
            pieces=[]
            left=0.
            for right in (.013,.25,.333,1.013,14.,16.,19.999):
                pieces.append(run.advance_until(right, inputs=[i for i in inputs if left <= i.t < right]))
                left=right
            pieces.append(run.finish(inputs=[i for i in inputs if i.t >= left]))
    assert observed.spikes == expected.spikes
    assert observed.states == expected.states
    assert tuple(s for part in pieces for s in part.spikes) == expected.spikes
    assert pieces[-1].states == expected.states


def test_reusing_run_discards_old_cache_and_generations(core):
    model = stepped()
    with core.compile_mixed([model]) as graph:
        with graph.create_run([(0.,)]) as run:
            first = run.execute(t_end=2.)
            for _ in range(10):
                run.reset([(0.,)])
                repeat = run.execute(t_end=2.)
                assert repeat.spikes == first.spikes
                assert repeat.states == first.states


def test_final_time_before_initial_step_is_not_skipped(core):
    model = stepped()
    with core.compile_mixed([model]) as graph:
        result = graph.run([(0.,)], t_end=1e-5)
    assert not result.spikes
    assert result.states[0].values[0] == pytest.approx(math.tan(1e-5), abs=1e-14)


def test_dense_replay_matches_inspections_across_empty_windows(core):
    model = stepped(QIF.replace('drive = 1', 'drive = 0.01'))
    times = (.013, .2, .25, .5, .6, 2.113, 10., 11.999)
    with core.compile_mixed([model]) as graph:
        observed = graph.run([(0.,)], t_end=12.,
            inspections=[StateInspectionRequest(t,0) for t in times])
    for sample in observed.inspections:
        replay, _ = core.advance_stepped(model, AugmentedState((0.,),0.), sample.t,
                                        prediction_horizon=12.)
        assert replay.values == sample.values


def test_dense_replay_rejects_missing_reset_and_short_horizon(core):
    model = stepped()
    with pytest.raises(CoreError, match='invalid argument'):
        core.advance_stepped(model, AugmentedState((0.,),0.), 1., prediction_horizon=2.)
    with pytest.raises(CoreError, match='invalid argument'):
        core.advance_stepped(model, AugmentedState((0.,),0.), .5, prediction_horizon=.1)


@pytest.mark.parametrize('end', [.25, 1., 10.])
def test_persistent_step_and_fsal_reduce_work_with_unchanged_error_control(core, end):
    source = QIF.replace('v*v + drive', '-v').replace('    drive = 1\n', '')
    model = stepped(source.replace('threshold = 1', 'threshold = 10'))
    old, old_cost = core.advance_stepped(model, AugmentedState((1.,),0.), end,
        prediction_horizon=end, prediction_reuse=False)
    new, new_cost = core.advance_stepped(model, AugmentedState((1.,),0.), end,
        prediction_horizon=end)
    assert old.values[0] == pytest.approx(math.exp(-end), abs=1e-8)
    assert new.values[0] == pytest.approx(math.exp(-end), abs=1e-8)
    assert old_cost.rhs_evaluations == 7*(old_cost.accepted_steps + old_cost.rejected_steps)
    assert new_cost.rhs_evaluations == 1+6*(new_cost.accepted_steps + new_cost.rejected_steps)
    assert new_cost.rhs_evaluations < old_cost.rhs_evaluations
    if end > .25:
        assert new_cost.accepted_steps < old_cost.accepted_steps


def test_v1_replay_preserves_old_restart_policy(core):
    source = QIF.replace('v*v + drive', '-v').replace('    drive = 1\n', '')
    model = stepped(source.replace('threshold = 1', 'threshold = 10'))
    manual = AugmentedState((1.,),0.)
    for t in (.25,.5,.75,1.):
        manual, _ = core.advance_stepped(model, manual, t)
    replay, _ = core.advance_stepped(model, AugmentedState((1.,),0.), 1.,
        prediction_horizon=1., prediction_reuse=False)
    assert manual == replay


@pytest.mark.parametrize('when', [.249, .25, .251, .5])
def test_drive_change_invalidates_fsal_at_continuation_endpoint(core, when):
    from lacuna import Graph, GraphModel, GraphNode, InputPort, InputMode, DriveInput
    graph = Graph(models=(GraphModel('qif',QIF),), nodes=(GraphNode(0,'qif',0.,{}),),
                  input_ports=(InputPort('drive',0,InputMode.DRIVE,'drive'),)).resolve()
    result = graph.run(core, t_end=4., drive_inputs=(DriveInput(when,'drive',0.),))
    # Before the change v=tan(t), afterwards v'=v^2; after reset it stays zero.
    expected = when + 1/math.tan(when) - 1
    assert [s.t for s in result.core.spikes] == pytest.approx([expected], abs=3e-7)


def test_rejected_trial_reuses_only_the_unchanged_start_derivative(core):
    model = stepped(initial_step=.25, maximum_step=.25, relative_tolerance=1e-12,
                    absolute_tolerance=1e-14)
    value, cost = core.advance_stepped(model, AugmentedState((0.,),0.), .5,
                                     prediction_horizon=.5)
    assert cost.rejected_steps > 0
    assert cost.rhs_evaluations == 1+6*(cost.accepted_steps+cost.rejected_steps)
    assert value.values[0] == pytest.approx(math.tan(.5), abs=1e-11)


@pytest.mark.parametrize('state_count', [1, 2, 3, 8])
def test_custom_equations_and_compact_reset_match_fresh_runs(core, state_count):
    extra = [f'x{i}' for i in range(1, state_count)]
    states = 'v: membrane\n' + '\n'.join(f'{x}: aux' for x in extra)
    rhs = 'dv/dt = drive - v*v' + (' + 0.01*x1' if extra else '')
    rhs += '\n' + '\n'.join(f'd{x}/dt = -{x} + v*{x}/(2 + {x}*{x})' for x in extra)
    resets = 'v <- 0\n' + '\n'.join(f'{x} <- 0.1' for x in extra)
    model = stepped(f'''neuron custom {{
        params {{ drive = 2 }}
        state {{ {states} }}
        dynamics {{ {rhs} }}
        threshold {{ v > 1 }}
        reset {{ {resets} }}
    }}''')
    requests = [StateInspectionRequest(t, 0) for t in (.017, .25, .713, 1.999)]
    with core.compile_mixed([model]) as graph:
        with graph.create_run([tuple(.1 for _ in range(state_count))]) as run:
            for offset in (.1, .3, -.2):
                initial = [tuple(offset + .02*i for i in range(state_count))]
                run.reset(initial)
                reused = run.execute(t_end=2., inspections=requests)
                fresh = graph.run(initial, t_end=2., inspections=requests)
                assert reused.spikes == fresh.spikes
                assert reused.states == fresh.states
                assert reused.inspections == fresh.inspections


def test_numerical_reset_restores_folded_parameters_after_drive_change(core):
    from lacuna.ffi import MixedDriveUpdate
    model = stepped()
    with core.compile_mixed([model]) as graph:
        expected = graph.run([(0.,)], t_end=2.)
        with graph.create_run([(0.,)]) as run:
            for _ in range(3):
                run.execute(t_end=2., drive_updates=[MixedDriveUpdate(.25, 0, .01, 'drive')])
                run.reset([(0.,)])
                restored = run.execute(t_end=2.)
                assert restored.spikes == expected.spikes
                assert restored.states == expected.states
                run.reset([(0.,)])


def test_numerical_episode_reset_invalidates_retained_trajectory(core):
    model = stepped(ADEX)
    initial = (-70., 0.)
    with core.compile_mixed([model]) as graph:
        with graph.create_incremental_run([initial], t_end=4.) as run:
            run.advance_until(2., inputs=[MixedInputSpike(.3,0,5.)])
            run.reset_episode([initial])
            actual = run.finish()
        # Compare at identical absolute times but with a different pre-reset
        # history. Mixed-run initial states do not accept AugmentedState.
        with graph.create_incremental_run([initial], t_end=4.) as reference:
            reference.advance_until(2.)
            reference.reset_episode([initial])
            fresh = reference.finish()
    assert actual.spikes == fresh.spikes
    assert actual.states == fresh.states
