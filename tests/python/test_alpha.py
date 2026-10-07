from __future__ import annotations

import math

import mpmath
import pytest

from lacuna import (
    AugmentedState,
    parse_neuron,
    parse_synapse,
    resolve_folded_alpha_lif,
)
from lacuna.errors import CapabilityError, CoreError
from lacuna.ffi import CoreEvaluator
from lacuna.ir import DispatchForm, SynapseTier


ALPHA_LIF = """
neuron AlphaLIF {
    params {
        tau_m : positive = 10.0
        v_rest = -65.0
        v_th = -50.0
        v_reset = -65.0
    }
    state {
        v : membrane
        i_exc : receptor
    }
    dynamics {
        dv/dt = -(v - v_rest)/tau_m + i_exc
    }
    threshold { v > v_th }
    reset { v <- v_reset }
    refractory { 2.0 }
}
"""


ALPHA_SYNAPSE = """
synapse alpha_exc {
    params { tau_s : positive = 5.0 }
    state { s; z }
    dynamics {
        ds/dt = -s/tau_s + z
        dz/dt = -z/tau_s
    }
    on_spike { z <- z + w/tau_s^2 }
    output { current = s }
}
"""


def _model(*, tau_s: float = 5.0):
    return resolve_folded_alpha_lif(
        parse_neuron(ALPHA_LIF),
        parse_synapse(ALPHA_SYNAPSE),
        receptor="i_exc",
        output="current",
        synapse_bindings={"tau_s": tau_s},
    )


def _driven_model():
    source = ALPHA_LIF.replace(
        "v_rest = -65.0", "v_rest = -65.0\n        drive = 20.0"
    ).replace(
        "-(v - v_rest)/tau_m + i_exc",
        "-(v - v_rest)/tau_m + drive/tau_m + i_exc",
    )
    return resolve_folded_alpha_lif(
        parse_neuron(source),
        parse_synapse(ALPHA_SYNAPSE),
        receptor="i_exc",
        output="current",
    )


def _high_precision_bisect(function, low: float, high: float) -> float:
    left = mpmath.mpf(low)
    right = mpmath.mpf(high)
    f_left = function(left)
    for _ in range(240):
        middle = (left + right) / 2
        f_middle = function(middle)
        if (f_left > 0) == (f_middle > 0):
            left = middle
            f_left = f_middle
        else:
            right = middle
    return float((left + right) / 2)


def test_alpha_structure_is_resolved_from_equations() -> None:
    model = _model()
    assert model.state_names == ("v", "alpha_exc.s", "alpha_exc.z")
    assert model.tier is SynapseTier.FOLDED_SHARED
    assert model.dispatch is DispatchForm.ROOT_FIND
    assert model.a == pytest.approx(-0.1)
    assert model.synaptic_decay == pytest.approx(-0.2)
    assert model.tau_m == pytest.approx(10.0)
    assert model.tau_s == pytest.approx(5.0)
    assert model.normal_roots == ("next_v", "next_s", "next_z")
    assert model.clamped_roots == ("clamped_v", "clamped_s", "clamped_z")
    assert model.reset_roots == ("reset_v", "reset_s", "reset_z")


def test_alpha_impulse_matches_hand_derived_trajectory(core: CoreEvaluator) -> None:
    model = _model()
    weight = 20.0
    duration = 5.0
    state = core.deposit_alpha(model, AugmentedState((-65.0, 0.0, 0.0), 0.0), weight)
    advanced = core.advance_alpha(model, state, duration)

    a = -1.0 / 10.0
    q = -1.0 / 5.0
    z0 = weight / 5.0**2
    difference = q - a
    integral_z = (
        math.exp(difference * duration) * (difference * duration - 1.0) + 1.0
    ) / difference**2
    expected_v = -65.0 + math.exp(a * duration) * z0 * integral_z
    expected_s = z0 * duration * math.exp(q * duration)
    expected_z = z0 * math.exp(q * duration)
    assert advanced.values == pytest.approx(
        (expected_v, expected_s, expected_z), abs=1e-12
    )
    assert advanced.t_last == duration


def test_alpha_propagation_has_the_semigroup_property(core: CoreEvaluator) -> None:
    model = _model()
    initial = AugmentedState((-61.0, 0.7, 0.3), 0.0)
    split = core.advance_alpha(model, initial, 1.75)
    split = core.advance_alpha(model, split, 7.0)
    combined = core.advance_alpha(model, initial, 7.0)
    assert split.values == pytest.approx(combined.values, abs=2e-13)
    assert split.t_last == combined.t_last == 7.0


def test_near_equal_decay_rates_use_stable_divided_differences(
    core: CoreEvaluator,
) -> None:
    model = _model(tau_s=9.99999999)
    initial = core.deposit_alpha(
        model, AugmentedState((-65.0, 0.0, 0.0), 0.0), 20.0
    )
    split = core.advance_alpha(model, initial, 2.5)
    split = core.advance_alpha(model, split, 8.0)
    combined = core.advance_alpha(model, initial, 8.0)
    assert split.values == pytest.approx(combined.values, abs=2e-12)


def test_alpha_root_find_matches_high_precision_reference(core: CoreEvaluator) -> None:
    model = _model()
    weight = mpmath.mpf(40)
    state = core.deposit_alpha(
        model, AugmentedState((-65.0, 0.0, 0.0), 0.0), float(weight)
    )
    prediction = core.predict_alpha(model, state)
    a = mpmath.mpf("-0.1")
    q = mpmath.mpf("-0.2")
    difference = q - a
    z0 = weight / 25

    def crossing(delta):
        integral = (
            mpmath.exp(difference * delta) * (difference * delta - 1) + 1
        ) / difference**2
        voltage = -65 + mpmath.exp(a * delta) * z0 * integral
        return -50 - voltage

    expected = _high_precision_bisect(crossing, 5.0, 12.0)
    assert prediction.t_spike == pytest.approx(expected, abs=5e-10)
    assert prediction.diagnostics.extrema_count == 1
    assert (
        prediction.diagnostics.bracket_high - prediction.diagnostics.bracket_low
        <= prediction.diagnostics.tolerance
    )
    assert prediction.diagnostics.iterations <= model.root_hint.iteration_cap


def test_alpha_root_find_reports_no_crossing_for_subthreshold_peak(
    core: CoreEvaluator,
) -> None:
    model = _model()
    state = core.deposit_alpha(
        model, AugmentedState((-65.0, 0.0, 0.0), 0.0), 20.0
    )
    prediction = core.predict_alpha(model, state)
    assert prediction.t_spike is None
    assert prediction.diagnostics.extrema_count == 1
    assert prediction.diagnostics.horizon > 0.0


def test_exact_threshold_tangency_is_not_a_rising_crossing(core: CoreEvaluator) -> None:
    model = _model()
    tangent_weight = 36.83111223426192
    state = core.deposit_alpha(
        model, AugmentedState((-65.0, 0.0, 0.0), 0.0), tangent_weight
    )
    assert core.predict_alpha(model, state).t_spike is None
    above = core.deposit_alpha(
        model,
        AugmentedState((-65.0, 0.0, 0.0), 0.0),
        tangent_weight * (1.0 + 1e-10),
    )
    assert core.predict_alpha(model, above).t_spike is not None


def test_first_of_two_upcrossings_is_selected(core: CoreEvaluator) -> None:
    model = _driven_model()
    initial = AugmentedState(
        (-63.944537107184296, 9.428834570662865, -2.644240074924669), 0.0
    )
    prediction = core.predict_alpha(model, initial)
    a = mpmath.mpf("-0.1")
    q = mpmath.mpf("-0.2")
    asymptote = mpmath.mpf(-45)
    v0, s0, z0 = map(mpmath.mpf, initial.values)
    difference = q - a
    ca = v0 - asymptote - s0 / difference + z0 / difference**2
    c0 = s0 / difference - z0 / difference**2
    c1 = z0 / difference

    def crossing(delta):
        voltage = (
            asymptote
            + ca * mpmath.exp(a * delta)
            + (c0 + c1 * delta) * mpmath.exp(q * delta)
        )
        return -50 - voltage

    first = _high_precision_bisect(crossing, 1.0, 4.0)
    second = _high_precision_bisect(crossing, 30.0, 38.0)
    assert first < second
    assert prediction.t_spike == pytest.approx(first, abs=5e-10)
    assert prediction.diagnostics.extrema_count == 2
    assert prediction.diagnostics.iterations <= model.root_hint.iteration_cap


def test_alpha_root_find_handles_driven_and_asymptotic_threshold_cases(
    core: CoreEvaluator,
) -> None:
    driven = _driven_model()
    prediction = core.predict_alpha(
        driven, AugmentedState((-65.0, 0.0, 0.0), 3.0)
    )
    assert prediction.t_spike == pytest.approx(3.0 + 10.0 * math.log(4.0), abs=5e-10)

    equality_source = ALPHA_LIF.replace(
        "v_rest = -65.0", "v_rest = -65.0\n        drive = 15.0"
    ).replace(
        "-(v - v_rest)/tau_m + i_exc",
        "-(v - v_rest)/tau_m + drive/tau_m + i_exc",
    )
    equality = resolve_folded_alpha_lif(
        parse_neuron(equality_source),
        parse_synapse(ALPHA_SYNAPSE),
        receptor="i_exc",
        output="current",
    )
    assert core.predict_alpha(
        equality, AugmentedState((-65.0, 0.0, 0.0), 0.0)
    ).t_spike is None


def test_alpha_delta_limit_crossings_converge_to_delivery_time(
    core: CoreEvaluator,
) -> None:
    crossing_times = []
    for tau_s in (0.2, 0.1, 0.05):
        model = _model(tau_s=tau_s)
        state = core.deposit_alpha(
            model, AugmentedState((-65.0, 0.0, 0.0), 0.0), 20.0
        )
        crossing_times.append(core.predict_alpha(model, state).t_spike)
    assert all(value is not None for value in crossing_times)
    times = [float(value) for value in crossing_times if value is not None]
    assert times[0] > times[1] > times[2] > 0.0
    assert times[-1] < 0.14


def test_same_time_alpha_deposits_aggregate(core: CoreEvaluator) -> None:
    model = _model()
    initial = AugmentedState((-65.0, 0.0, 0.0), 3.0)
    separate = core.deposit_alpha(model, initial, 7.0)
    separate = core.deposit_alpha(model, separate, 13.0)
    aggregate = core.deposit_alpha(model, initial, 20.0)
    assert separate.values == pytest.approx(aggregate.values, abs=1e-15)
    assert separate.t_last == aggregate.t_last


def test_clamp_holds_membrane_while_kernel_evolves_and_accepts_deposits(
    core: CoreEvaluator,
) -> None:
    model = _model()
    state = core.deposit_alpha(model, AugmentedState((model.reset, 0.0, 0.0), 0.0), 10.0)
    state = core.advance_alpha(model, state, 1.0, clamped=True)
    membrane_after_first_gap = state.values[0]
    kernel_before_second_deposit = state.values[1:]
    state = core.deposit_alpha(model, state, 5.0)
    assert state.values[0] == membrane_after_first_gap == model.reset
    assert state.values[1] == kernel_before_second_deposit[0]
    assert state.values[2] > kernel_before_second_deposit[1]
    state = core.advance_alpha(model, state, 2.0, clamped=True)
    assert state.values[0] == model.reset
    assert state.values[1] > 0.0
    assert state.values[2] > 0.0


def test_equal_membrane_and_synaptic_rates_use_repeated_real_mode(
    core: CoreEvaluator,
) -> None:
    model = _model(tau_s=10.0)
    state = core.deposit_alpha(
        model, AugmentedState((-65.0, 0.0, 0.0), 0.0), 80.0
    )
    prediction = core.predict_alpha(model, state)
    assert prediction.t_spike == pytest.approx(10.195479618410925, abs=2e-9)
    advanced = core.advance_alpha(model, state, 20.0)
    assert advanced.values[0] == pytest.approx(
        -65.0 + 0.4 * 20.0**2 * math.exp(-2.0), abs=2e-12
    )


def test_non_unit_area_alpha_deposit_is_rejected() -> None:
    invalid = ALPHA_SYNAPSE.replace("w/tau_s^2", "w/tau_s")
    with pytest.raises(CapabilityError, match="unit-area"):
        resolve_folded_alpha_lif(
            parse_neuron(ALPHA_LIF),
            parse_synapse(invalid),
            receptor="i_exc",
            output="current",
        )


def test_alpha_advance_rejects_time_reversal(core: CoreEvaluator) -> None:
    with pytest.raises(CoreError, match="precedes"):
        core.advance_alpha(_model(), AugmentedState((-65.0, 0.0, 0.0), 2.0), 1.0)


def test_alpha_deposit_rejects_nonfinite_weight(core: CoreEvaluator) -> None:
    with pytest.raises(CoreError, match="invalid argument"):
        core.deposit_alpha(
            _model(), AugmentedState((-65.0, 0.0, 0.0), 0.0), math.inf
        )
