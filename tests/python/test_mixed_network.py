from __future__ import annotations

import pytest

from lacuna import (
    AugmentedState,
    DepositKind,
    MixedDriveUpdate,
    MixedEdge,
    MixedInputSpike,
    parse_neuron,
    resolve_scalar_lif,
)
from lacuna.errors import CoreError
from lacuna.ffi import CoreEvaluator

from .test_alpha import _driven_model, _model
from .test_dsl_resolver import LIF


def _reactive():
    return resolve_scalar_lif(parse_neuron(LIF), {"drive": 0.0})


def _alpha_input(t: float, node: int, weight: float) -> MixedInputSpike:
    return MixedInputSpike(
        t,
        node,
        weight,
        DepositKind.FOLDED_ALPHA,
        target=2,
    )


def test_single_alpha_network_matches_standalone_prediction(core: CoreEvaluator) -> None:
    model = _model()
    deposited = core.deposit_alpha(
        model, AugmentedState((-65.0, 0.0, 0.0), 1.0), 40.0
    )
    expected = core.predict_alpha(model, deposited).t_spike
    assert expected is not None

    result = core.run_mixed(
        [model],
        [(-65.0, 0.0, 0.0)],
        inputs=[_alpha_input(1.0, 0, 40.0)],
        t_end=expected + 0.25,
    )

    assert [(spike.node, spike.t) for spike in result.spikes] == pytest.approx(
        [(0, expected)], abs=5e-10
    )
    assert result.stats.autonomous_spikes_confirmed == 1


def test_mixed_scalar_alpha_scalar_chain_uses_csr_deliveries(
    core: CoreEvaluator,
) -> None:
    models = [_reactive(), _model(), _reactive()]
    result = core.run_mixed(
        models,
        [-65.0, (-65.0, 0.0, 0.0), -65.0],
        edges=[
            MixedEdge(
                0,
                1,
                40.0,
                1.0,
                DepositKind.FOLDED_ALPHA,
                target=2,
            ),
            MixedEdge(1, 2, 20.0, 1.0),
        ],
        inputs=[MixedInputSpike(1.0, 0, 20.0)],
        t_end=14.0,
    )

    assert [spike.node for spike in result.spikes] == [0, 1, 2]
    assert [spike.t for spike in result.spikes] == pytest.approx(
        [1.0, 11.230709364362863, 12.230709364362863], abs=5e-10
    )
    assert result.stats.deliveries_scheduled == 2
    assert result.stats.deliveries_processed == 2


def test_alpha_deposit_invalidates_a_stale_prediction(core: CoreEvaluator) -> None:
    result = core.run_mixed(
        [_model()],
        [(-65.0, 0.0, 0.0)],
        inputs=[_alpha_input(1.0, 0, 40.0), _alpha_input(5.0, 0, -10.0)],
        t_end=15.0,
    )
    assert result.spikes == ()
    assert result.stats.stale_predictions == 1


def test_same_time_alpha_inputs_are_aggregated_before_one_deposit(
    core: CoreEvaluator,
) -> None:
    model = _model()
    separate = core.run_mixed(
        [model],
        [(-65.0, 0.0, 0.0)],
        inputs=[_alpha_input(1.0, 0, 17.0), _alpha_input(1.0, 0, 23.0)],
        t_end=12.0,
    )
    aggregate = core.run_mixed(
        [model],
        [(-65.0, 0.0, 0.0)],
        inputs=[_alpha_input(1.0, 0, 40.0)],
        t_end=12.0,
    )
    assert separate.states == pytest.approx(aggregate.states)
    assert separate.spikes == pytest.approx(aggregate.spikes)


def test_alpha_state_accumulates_input_during_fixed_clamp(core: CoreEvaluator) -> None:
    model = _model()
    baseline = core.run_mixed(
        [model],
        [(-65.0, 0.0, 0.0)],
        inputs=[_alpha_input(1.0, 0, 40.0)],
        t_end=19.0,
    )
    accumulated = core.run_mixed(
        [model],
        [(-65.0, 0.0, 0.0)],
        inputs=[_alpha_input(1.0, 0, 40.0), _alpha_input(11.0, 0, 40.0)],
        t_end=19.0,
    )

    assert len(baseline.spikes) == 1
    assert [spike.t for spike in accumulated.spikes] == pytest.approx(
        [10.230709364362863, 16.81814869285991], abs=5e-9
    )
    assert accumulated.stats.refractory_releases_processed == 2


def test_alpha_drive_boundary_invalidates_prediction(core: CoreEvaluator) -> None:
    result = core.run_mixed(
        [_driven_model()],
        [(-65.0, 0.0, 0.0)],
        drive_updates=[MixedDriveUpdate(5.0, 0, 0.0, "drive")],
        t_end=20.0,
    )
    assert result.spikes == ()
    assert result.stats.drive_updates_processed == 1
    assert result.stats.stale_predictions == 1


def test_all_scalar_mixed_runner_matches_scalar_runner(core: CoreEvaluator) -> None:
    models = [_reactive(), _reactive()]
    scalar = core.run_delta(
        models,
        [-65.0, -65.0],
        edges=[],
        inputs=[],
        t_end=3.0,
    )
    mixed = core.run_mixed(models, [-65.0, -65.0], t_end=3.0)
    assert mixed.spikes == scalar.spikes
    assert [state.values[0] for state in mixed.states] == pytest.approx(
        [state.value for state in scalar.states], abs=1e-12
    )


def test_folded_alpha_deposit_cannot_target_scalar_node(core: CoreEvaluator) -> None:
    with pytest.raises(CoreError, match="invalid argument"):
        core.run_mixed(
            [_reactive()],
            [-65.0],
            inputs=[_alpha_input(1.0, 0, 20.0)],
            t_end=2.0,
        )
