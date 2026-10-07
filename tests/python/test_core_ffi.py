from __future__ import annotations

import math

import pytest

from lacuna import parse_neuron, resolve_scalar_lif
from lacuna.errors import CoreError
from lacuna.ffi import CoreEvaluator, ScalarState
from lacuna.ir import DispatchForm

from .test_dsl_resolver import LIF


def test_end_to_end_advance_uses_c_core(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    state = core.advance(model, ScalarState(-65.0, 0.0), 10.0)
    assert state.value == pytest.approx(-45.0 - 20.0 * math.exp(-1.0), abs=1e-12)
    assert state.t_last == 10.0


def test_end_to_end_closed_form_prediction(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    prediction = core.predict(model, ScalarState(-65.0, 0.0))
    assert prediction.dispatch is DispatchForm.CLOSED_FORM
    assert prediction.t_spike == pytest.approx(10.0 * math.log(4.0), abs=1e-12)


def test_loaded_core_reports_the_expected_abi_layout(core: CoreEvaluator) -> None:
    assert core.abi_version == 17
    assert core.network_error_size > 0


def test_reactive_prediction_schedules_nothing(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF), {"drive": 10.0})
    prediction = core.predict(model, ScalarState(-65.0, 3.0))
    assert prediction.dispatch is DispatchForm.REACTIVE
    assert prediction.t_spike is None


def test_time_reversal_is_rejected(core: CoreEvaluator) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF))
    with pytest.raises(CoreError, match="precedes"):
        core.advance(model, ScalarState(-65.0, 5.0), 4.0)
