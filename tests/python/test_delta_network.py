from __future__ import annotations

import math

import pytest

from lacuna import (
    DeltaEdge,
    DriveUpdate,
    InputSpike,
    NeuronPolarity,
    ScalarState,
    parse_neuron,
    resolve_scalar_lif,
)
from lacuna.errors import CoreError
from lacuna.ffi import CoreEvaluator

from .test_dsl_resolver import LIF


def _reactive():
    return resolve_scalar_lif(parse_neuron(LIF), {"drive": 0.0})


def test_autonomous_spike_drives_delayed_reactive_target(core: CoreEvaluator) -> None:
    source = resolve_scalar_lif(parse_neuron(LIF))
    target = _reactive()
    result = core.run_delta(
        [source, target],
        [-65.0, -65.0],
        edges=[DeltaEdge(0, 1, 20.0, 1.0)],
        t_end=16.0,
    )
    expected = 10.0 * math.log(4.0)
    assert [spike.node for spike in result.spikes] == [0, 1]
    assert [spike.t for spike in result.spikes] == pytest.approx(
        [expected, expected + 1.0], abs=1e-12
    )


def test_same_time_excitation_and_inhibition_are_aggregated(core: CoreEvaluator) -> None:
    result = core.run_delta(
        [_reactive()],
        [-65.0],
        inputs=[InputSpike(1.0, 0, 20.0), InputSpike(1.0, 0, -10.0)],
        t_end=2.0,
    )
    assert result.spikes == ()
    assert result.states[0].value == pytest.approx(-65.0 + 10.0 * math.exp(-0.1), abs=1e-12)


def test_core_derives_outgoing_sign_from_intrinsic_neuron_polarity(
    core: CoreEvaluator,
) -> None:
    model = _reactive()
    result = core.run_delta(
        [model, model],
        [-65.0, -65.0],
        edges=[DeltaEdge(0, 1, 20.0)],
        polarities=(NeuronPolarity.INHIBITORY, NeuronPolarity.EXCITATORY),
        inputs=[InputSpike(1.0, 0, 20.0)],
        t_end=2.0,
    )
    assert [(spike.node, spike.t) for spike in result.spikes] == [(0, 1.0)]
    assert result.states[1].value == pytest.approx(
        -65.0 - 20.0 * math.exp(-0.1), abs=1e-12
    )

    with pytest.raises(ValueError, match="nonnegative magnitudes"):
        core.run_delta(
            [model],
            [-65.0],
            edges=[DeltaEdge(0, 0, -1.0)],
            t_end=1.0,
        )


def test_zero_delay_cascade_is_processed_to_quiescence(core: CoreEvaluator) -> None:
    model = _reactive()
    result = core.run_delta(
        [model, model, model],
        [-65.0, -65.0, -65.0],
        edges=[DeltaEdge(0, 1, 20.0), DeltaEdge(1, 2, 20.0)],
        inputs=[InputSpike(1.0, 0, 20.0)],
        t_end=2.0,
    )
    assert [(spike.node, spike.t) for spike in result.spikes] == [
        (0, 1.0),
        (1, 1.0),
        (2, 1.0),
    ]
    assert result.stats.max_same_time_cascade_depth == 3


def test_delivery_lazily_invalidates_old_prediction(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    result = core.run_delta(
        [model],
        [-65.0],
        inputs=[InputSpike(5.0, 0, -5.0)],
        t_end=15.0,
    )
    assert result.spikes == ()
    assert result.stats.stale_predictions == 1


def test_fixed_clamp_release_wakes_node_without_input(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    result = core.run_delta([model], [-65.0], t_end=30.0)
    period = 10.0 * math.log(4.0)
    assert [spike.t for spike in result.spikes] == pytest.approx(
        [period, period * 2.0 + 2.0], abs=1e-12
    )


def test_equal_time_inhibition_invalidates_autonomous_prediction(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    predicted = core.predict(model, ScalarState(-65.0, 0.0)).t_spike
    assert predicted is not None
    result = core.run_delta(
        [model],
        [-65.0],
        inputs=[InputSpike(predicted, 0, -20.0)],
        t_end=predicted + 1.0,
    )
    assert result.spikes == ()
    assert result.stats.stale_predictions == 1


def test_drive_boundary_at_crossing_is_evaluated_in_timestamp_batch(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    predicted = core.predict(model, ScalarState(-65.0, 0.0)).t_spike
    assert predicted is not None
    result = core.run_delta(
        [model],
        [-65.0],
        drive_updates=[DriveUpdate(predicted, 0, -6.5)],
        t_end=predicted + 0.5,
    )
    assert [spike.t for spike in result.spikes] == pytest.approx([predicted], abs=1e-12)


def test_zero_delay_self_loop_hits_cascade_limit(core: CoreEvaluator) -> None:
    no_refractory = LIF.replace("    refractory { 2.0 }\n", "")
    model = resolve_scalar_lif(parse_neuron(no_refractory), {"drive": 0.0})
    with pytest.raises(CoreError, match="cascade limit"):
        core.run_delta(
            [model],
            [-65.0],
            edges=[DeltaEdge(0, 0, 20.0)],
            inputs=[InputSpike(1.0, 0, 20.0)],
            t_end=2.0,
            queue_capacity=32,
            output_capacity=32,
            same_time_cascade_limit=3,
        )


def test_queue_overflow_is_a_hard_error(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    with pytest.raises(CoreError, match="queue capacity"):
        core.run_delta(
            [model],
            [-65.0],
            inputs=[InputSpike(1.0, 0, -1.0), InputSpike(1.5, 0, -1.0)],
            t_end=2.0,
            queue_capacity=1,
        )


def test_csr_handles_canonically_ordered_edges_interleaved_by_source(
    core: CoreEvaluator,
) -> None:
    model = _reactive()
    result = core.run_delta(
        [model, model, model, model],
        [-65.0, -65.0, -65.0, -65.0],
        edges=[
            DeltaEdge(1, 3, 20.0),
            DeltaEdge(0, 2, 20.0),
            DeltaEdge(2, 3, 20.0),
            DeltaEdge(0, 1, 20.0),
        ],
        inputs=[InputSpike(1.0, 0, 20.0)],
        t_end=2.0,
    )
    assert [spike.node for spike in result.spikes] == [0, 1, 2, 3]
    assert result.stats.deliveries_scheduled == 4
    assert result.stats.deliveries_processed == 4
