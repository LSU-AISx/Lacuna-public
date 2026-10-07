from __future__ import annotations

import pytest

from lacuna import parse_neuron, resolve_scalar_lif
from lacuna.errors import CapabilityError, ResolutionError
from lacuna.ir import DispatchForm


LIF = """
neuron LIF {
    params {
        tau_m : positive = 10.0
        v_rest = -65.0
        drive = 20.0
        v_th = -50.0
        v_reset = -65.0
    }
    state {
        v : membrane
    }
    dynamics {
        dv/dt = -(v - v_rest)/tau_m + drive/tau_m
    }
    threshold { v > v_th }
    reset { v <- v_reset }
    refractory { 2.0 }
}
"""


def test_parse_and_resolve_driven_lif() -> None:
    resolved = resolve_scalar_lif(parse_neuron(LIF))
    assert resolved.a == pytest.approx(-0.1)
    assert resolved.b == pytest.approx(-4.5)
    assert resolved.asymptote == pytest.approx(-45.0)
    assert resolved.dispatch is DispatchForm.CLOSED_FORM
    assert resolved.refractory == 2.0
    assert resolved.normal_roots == ("next_v",)
    assert resolved.clamped_roots == ("clamped_v",)
    assert resolved.reset_roots == ("reset_v",)
    assert resolved.root_hint.decay_root == "crossing_decay"
    assert resolved.root_hint.affine_root == "crossing_affine"
    assert len(resolved.model_hash) == 64
    assert len(resolved.resolution_key) == 64


def test_value_rebinding_preserves_resolution_key_and_changes_guard() -> None:
    model = parse_neuron(LIF)
    driven = resolve_scalar_lif(model)
    reactive = resolve_scalar_lif(model, {"drive": 10.0})
    assert driven.resolution_key == reactive.resolution_key
    assert reactive.dispatch is DispatchForm.REACTIVE
    assert reactive.asymptote == pytest.approx(-55.0)
    assert driven.propagation_dag.parameters == reactive.propagation_dag.parameters


def test_positive_parameter_domain_is_enforced() -> None:
    with pytest.raises(ResolutionError, match="must be positive"):
        resolve_scalar_lif(parse_neuron(LIF), {"tau_m": 0.0})


def test_unknown_symbol_is_rejected() -> None:
    typo = LIF.replace("drive/tau_m", "driev/tau_m")
    with pytest.raises(ResolutionError, match="unknown symbol"):
        resolve_scalar_lif(parse_neuron(typo))


def test_scientific_numeric_literal_in_expression_is_not_an_unknown_symbol() -> None:
    with_literal = LIF.replace("drive/tau_m", "drive/tau_m + 1e-3")
    resolved = resolve_scalar_lif(parse_neuron(with_literal))
    assert resolved.b == pytest.approx(-4.499)


def test_nonlinear_state_routes_outside_milestone() -> None:
    nonlinear = LIF.replace("-(v - v_rest)/tau_m", "v*v/tau_m")
    with pytest.raises(CapabilityError, match="nonlinear"):
        resolve_scalar_lif(parse_neuron(nonlinear))
