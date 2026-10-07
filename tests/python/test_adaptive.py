from __future__ import annotations

import math

import mpmath
import pytest

from lacuna import (
    AugmentedState,
    DepositKind,
    Graph,
    GraphModel,
    GraphNode,
    MixedDriveUpdate,
    MixedEdge,
    MixedInputSpike,
    StateInspectionRequest,
    parse_neuron,
    resolve_adaptive_lif,
)
from lacuna.errors import CapabilityError
from lacuna.ffi import CoreEvaluator
from lacuna.ir import DispatchForm, ResolvedAdaptiveLIF

from .test_dsl_resolver import LIF
from lacuna import resolve_scalar_lif


ADAPTIVE_LIF = """
neuron AdaptiveLIF {
    params {
        tau_m : positive = 10.0
        tau_w : positive = 50.0
        v_rest = -65.0
        drive = 25.0
        v_th = -50.0
        v_reset = -65.0
        beta = 2.0
    }
    state {
        v : membrane
        w : adaptation
    }
    dynamics {
        dv/dt = -(v - v_rest)/tau_m - w + drive/tau_m
        dw/dt = -w/tau_w
    }
    threshold { v > v_th }
    reset {
        v <- v_reset
        w <- w + beta
    }
    refractory { 2.0 }
}
"""


def _model(**bindings: float) -> ResolvedAdaptiveLIF:
    return resolve_adaptive_lif(parse_neuron(ADAPTIVE_LIF), bindings)


def _reactive_scalar():
    return resolve_scalar_lif(parse_neuron(LIF), {"drive": 0.0})


def test_adaptive_structure_is_resolved_from_equations() -> None:
    model = _model()
    assert model.state_names == ("v", "w")
    assert model.dispatch is DispatchForm.ROOT_FIND
    assert model.a == pytest.approx(-0.1)
    assert model.adaptation_decay == pytest.approx(-0.02)
    assert model.coupling == -1.0
    assert model.adaptation_jump == 2.0
    assert model.tau_m == 10.0
    assert model.tau_w == 50.0
    assert model.normal_roots == ("next_v", "next_w")
    assert model.clamped_roots == ("clamped_v", "clamped_w")
    assert model.reset_roots == ("reset_v", "reset_w")


def test_adaptive_propagation_matches_hand_solution(core: CoreEvaluator) -> None:
    model = _model()
    initial = AugmentedState((-61.0, 1.25), 0.0)
    duration = 7.0
    advanced = core.advance_adaptive(model, initial, duration)
    a = model.a
    q = model.adaptation_decay
    ea = math.exp(a * duration)
    eq = math.exp(q * duration)
    expected_w = initial.values[1] * eq
    expected_v = (
        -model.b / model.a
        + (initial.values[0] + model.b / model.a) * ea
        + model.coupling * initial.values[1] * (eq - ea) / (q - a)
    )
    assert advanced.values == pytest.approx((expected_v, expected_w), abs=2e-13)
    assert advanced.t_last == duration


def test_adaptive_propagation_has_semigroup_property(core: CoreEvaluator) -> None:
    model = _model()
    initial = AugmentedState((-61.0, 1.25), 0.0)
    split = core.advance_adaptive(model, initial, 2.25)
    split = core.advance_adaptive(model, split, 9.0)
    combined = core.advance_adaptive(model, initial, 9.0)
    assert split.values == pytest.approx(combined.values, abs=5e-13)


def test_near_equal_adaptive_rates_use_stable_propagation(core: CoreEvaluator) -> None:
    model = _model(tau_w=10.00000001)
    initial = AugmentedState((-61.0, 1.25), 0.0)
    split = core.advance_adaptive(model, initial, 2.25)
    split = core.advance_adaptive(model, split, 9.0)
    combined = core.advance_adaptive(model, initial, 9.0)
    assert split.values == pytest.approx(combined.values, abs=2e-12)
    assert core.predict_adaptive(model, initial).t_spike is not None


def test_adaptive_reset_is_atomic_and_increments_current(core: CoreEvaluator) -> None:
    model = _model()
    reset = core.reset_adaptive(model, AugmentedState((-50.0, 1.25), 3.0))
    assert reset.values == pytest.approx((-65.0, 3.25))
    assert reset.t_last == 3.0


def test_adaptive_crossing_matches_high_precision_reference(core: CoreEvaluator) -> None:
    model = _model()
    state = AugmentedState((-65.0, 3.0), 0.0)
    prediction = core.predict_adaptive(model, state)
    assert prediction.t_spike is not None
    a = mpmath.mpf(str(model.a))
    q = mpmath.mpf(str(model.adaptation_decay))
    coupling = mpmath.mpf(str(model.coupling))
    v0 = mpmath.mpf(str(state.values[0]))
    w0 = mpmath.mpf(str(state.values[1]))
    asymptote = -mpmath.mpf(str(model.b)) / a
    c_w = coupling * w0 / (q - a)
    c_m = v0 - asymptote - c_w

    def crossing(delta):
        voltage = asymptote + c_m * mpmath.exp(a * delta) + c_w * mpmath.exp(q * delta)
        return mpmath.mpf(str(model.threshold)) - voltage

    expected = mpmath.findroot(crossing, (50, 80))
    assert prediction.t_spike == pytest.approx(float(expected), abs=2e-9)
    assert prediction.diagnostics.extrema_count == 1
    assert prediction.diagnostics.iterations <= model.root_hint.iteration_cap


def test_adaptive_subthreshold_asymptote_has_no_crossing(core: CoreEvaluator) -> None:
    model = _model(drive=10.0)
    prediction = core.predict_adaptive(model, AugmentedState((-65.0, 0.0), 0.0))
    assert prediction.t_spike is None
    assert prediction.diagnostics.horizon > 0.0


def test_adaptive_threshold_asymptote_is_not_a_finite_crossing(
    core: CoreEvaluator,
) -> None:
    model = _model(drive=15.0)
    prediction = core.predict_adaptive(model, AugmentedState((-65.0, 0.0), 0.0))
    assert prediction.t_spike is None


def test_adaptation_decays_while_membrane_is_refractory_clamped(
    core: CoreEvaluator,
) -> None:
    model = _model()
    first = core.predict_adaptive(model, AugmentedState((-65.0, 0.0), 0.0))
    assert first.t_spike is not None
    end = first.t_spike + 1.0
    result = core.run_mixed(
        [model],
        [(-65.0, 0.0)],
        t_end=end,
        inspections=(
            StateInspectionRequest(first.t_spike, 0),
            StateInspectionRequest(end, 0),
        ),
    )
    assert len(result.spikes) == 1
    assert result.inspections[0].clamped is True
    assert result.inspections[0].values == pytest.approx((-65.0, 2.0))
    assert result.inspections[1].clamped is True
    assert result.states[0].values[0] == -65.0
    assert result.states[0].values[1] == pytest.approx(2.0 * math.exp(-1.0 / 50.0))
    assert result.inspections[1].values == pytest.approx(result.states[0].values)


def test_constant_drive_produces_firing_rate_adaptation(core: CoreEvaluator) -> None:
    result = core.run_mixed([_model()], [(-65.0, 0.0)], t_end=300.0)
    times = [spike.t for spike in result.spikes]
    intervals = [right - left for left, right in zip(times, times[1:])]
    assert len(intervals) >= 3
    assert intervals[0] < intervals[1] < intervals[2]
    assert intervals[-1] == pytest.approx(intervals[-2], rel=5e-3)


def test_adaptive_drive_update_invalidates_prediction(core: CoreEvaluator) -> None:
    result = core.run_mixed(
        [_model()],
        [(-65.0, 0.0)],
        drive_updates=[MixedDriveUpdate(5.0, 0, 0.0, "drive")],
        t_end=20.0,
    )
    assert result.spikes == ()
    assert result.stats.drive_updates_processed == 1
    assert result.stats.stale_predictions == 1


def test_scalar_adaptive_scalar_chain_runs_through_csr(core: CoreEvaluator) -> None:
    adaptive = _model(drive=0.0)
    models = [_reactive_scalar(), adaptive, _reactive_scalar()]
    result = core.run_mixed(
        models,
        [-65.0, (-65.0, 0.0), -65.0],
        edges=[
            MixedEdge(0, 1, 20.0, 1.0, DepositKind.STATE_ADD, 0),
            MixedEdge(1, 2, 20.0, 1.0, DepositKind.STATE_ADD, 0),
        ],
        inputs=[MixedInputSpike(1.0, 0, 20.0)],
        t_end=4.0,
    )
    assert [spike.node for spike in result.spikes] == [0, 1, 2]
    assert [spike.t for spike in result.spikes] == [1.0, 2.0, 3.0]
    assert result.states[1].values[1] > 0.0
    assert result.stats.deliveries_processed == 2


def test_adaptive_graph_round_trip_and_execution(core: CoreEvaluator) -> None:
    graph = Graph(
        models=(GraphModel("adaptive", ADAPTIVE_LIF),),
        nodes=(GraphNode(0, "adaptive", (-65.0, 0.0), {}),),
    )
    restored = Graph.from_text(graph.to_text())
    resolved = restored.resolve()
    assert isinstance(resolved.models[0], ResolvedAdaptiveLIF)
    assert resolved.initial_values == ((-65.0, 0.0),)
    result = resolved.run(core, t_end=10.0)
    assert len(result.core.spikes) == 1


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (ADAPTIVE_LIF.replace("tau_w : positive = 50.0", "tau_w : positive = 10.0"), "equal"),
        (ADAPTIVE_LIF.replace("- w + drive/tau_m", "+ w + drive/tau_m"), "inhibitively"),
        (ADAPTIVE_LIF.replace("dw/dt = -w/tau_w", "dw/dt = -w/tau_w + v"), "independent"),
    ],
)
def test_unsupported_adaptive_regimes_fail_with_evidence(source: str, message: str) -> None:
    with pytest.raises(CapabilityError, match=message):
        resolve_adaptive_lif(parse_neuron(source))


def test_negative_adaptation_jump_is_rejected() -> None:
    with pytest.raises(CapabilityError, match="nonnegative"):
        _model(beta=-1.0)
