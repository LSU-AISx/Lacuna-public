from __future__ import annotations

from dataclasses import replace

import pytest

from lacuna.codec import (
    BurstEncoder,
    HeldCurrentEncoder,
    LatencyBurstEncoder,
    NativeEventEncoder,
    PoissonRateEncoder,
    RateDecoder,
    RateMode,
    RegularRateEncoder,
    TemporalWeightDecoder,
    TTFSEncoder,
    TTFSDecoder,
)
from lacuna.errors import PrecisionResolutionError
from lacuna.graph import (
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    GraphSynapse,
    InputMode,
    InputPort,
    OutputPort,
)
from lacuna.plasticity import (
    ModulatedSTDP,
    PairSTDP,
    SoftExcursionModulated,
    TripletSTDP,
    VoltageModulatedSTDP,
)
from lacuna.precision import PrecisionProfile
from lacuna.precision_authoring import prepare_precision_graph


NEURON = """
neuron Rounded {
    params { tau : positive = 20.1; drive = 0.1 }
    state { v : membrane }
    dynamics { dv/dt = (-v + drive + 0.1) / tau }
    threshold { v > 1.1 }
    reset { v <- -0.1 }
    refractory { duration = 0.1 }
}
"""

SYNAPSE = """
synapse RoundedCurrent {
    params { tau_s : positive = 3.1 }
    state { s }
    dynamics { ds/dt = -s/tau_s + 0.1 }
    on_spike { s <- s + 0.1*w }
    output { current = 0.1*s }
}
"""


def _graph(**kwargs):
    return Graph(
        models=(GraphModel("neuron", NEURON),),
        nodes=(GraphNode(0, "neuron", 0.1, {}),),
        **kwargs,
    )


def _prepare(graph, profile=PrecisionProfile.FLOAT32):
    observed = []

    def record(value, path, is_time):
        try:
            method = profile.round_time if is_time else profile.round_real
            rounded = method(value, name=path)
        except (TypeError, ValueError) as exc:
            raise PrecisionResolutionError(str(exc)) from exc
        observed.append((path, is_time, value, rounded))
        return rounded

    prepared, models, synapses = prepare_precision_graph(
        graph, profile, record=record,
    )
    return prepared, models, synapses, observed


def test_authoring_inputs_are_rounded_without_mutating_originals():
    graph = _graph(edges=(GraphEdge(0, 0, 0, 0.1, 0.1),))
    prepared, models, _, observed = _prepare(graph)
    rounded = PrecisionProfile.FLOAT32.round_real(0.1)
    assert graph.nodes[0].bindings == {}
    assert graph.nodes[0].initial == 0.1
    assert graph.edges[0].weight == 0.1
    assert prepared.nodes[0].initial == rounded
    assert prepared.nodes[0].bindings["drive"] == rounded
    assert prepared.edges[0].weight == rounded
    assert prepared.edges[0].delay == rounded
    assert models["neuron"].parameters[1].default == rounded
    assert models["neuron"].refractory.duration == rounded
    assert any(path == "edges[0].delay" and time for path, time, _, _ in observed)
    prepared.nodes[0].bindings["drive"] = 7.0
    models["neuron"].dynamics["v"] = "0"
    assert graph.nodes[0].bindings == {}
    assert "0.1" in graph.models[0].source


def test_unused_unrepresentable_defaults_do_not_override_explicit_bindings():
    source = NEURON.replace("drive = 0.1", "drive = 1e100")
    graph = replace(
        _graph(), models=(GraphModel("neuron", source),),
        nodes=(GraphNode(0, "neuron", 0.0, {"drive": 0.25}),),
    )
    prepared, models, _, observed = _prepare(graph)
    assert prepared.nodes[0].bindings["drive"] == 0.25
    assert models["neuron"].parameters[1].default == 1e100
    assert all(value != 1e100 for _, _, value, _ in observed)
    graph = replace(graph, nodes=graph.nodes + (GraphNode(1, "neuron", 0.0, {}),))
    with pytest.raises(PrecisionResolutionError, match="overflows float32"):
        _prepare(graph)


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
def test_model_and_time_roles_follow_selected_profile(profile):
    graph = _graph(edges=(GraphEdge(0, 0, 0, 0.1, 0.1),))
    prepared, models, _, _ = _prepare(graph, profile)
    assert prepared.edges[0].weight == profile.round_real(0.1)
    assert prepared.edges[0].delay == profile.round_time(0.1)
    assert models["neuron"].refractory.duration == profile.round_time(0.1)


def test_numeric_literals_round_without_changing_integer_powers_or_identifiers():
    source = NEURON.replace(
        "(-v + drive + 0.1) / tau", "(-v + drive + 0.1*v**2 + 1e-1)/tau",
    )
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    _, models, _, _ = _prepare(graph)
    model = models["neuron"]
    rounded = repr(PrecisionProfile.FLOAT32.round_real(0.1))
    assert f"{rounded}*v**2" in model.dynamics["v"]
    assert "**2.0" not in model.dynamics["v"]
    assert model.dynamics["v"].count(rounded) == 2
    assert model.threshold.level == repr(PrecisionProfile.FLOAT32.round_real(1.1))
    assert model.reset["v"] == f"-{rounded}"
    _, models64, _, _ = _prepare(graph, PrecisionProfile.FLOAT64)
    assert models64["neuron"].dynamics["v"] == "(-v + drive + 0.1*v**2 + 1e-1)/tau"


def test_hazard_literals_are_rounded_before_resolution():
    source = NEURON.replace("threshold { v > 1.1 }", "hazard { rate = exp(0.1*v) }")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    _, models, _, _ = _prepare(graph)
    rounded = repr(PrecisionProfile.FLOAT32.round_real(0.1))
    assert models["neuron"].hazard.rate == f"exp({rounded}*v)"


def test_parameter_names_with_digits_are_not_treated_as_numeric_literals():
    source = NEURON.replace("drive", "drive123")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    _, models, _, observed = _prepare(graph)
    assert "drive123" in models["neuron"].dynamics["v"]
    assert not any(value == 123.0 for _, _, value, _ in observed)


@pytest.mark.parametrize("literal", (
    "1e100", "1e-100", "1e999", "1e-999", "-1e-999", "1e99999999999999999999",
))
def test_unrepresentable_equation_literals_fail_before_resolution(literal):
    source = NEURON.replace("+ 0.1)", f"+ {literal})")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    with pytest.raises(PrecisionResolutionError, match="literal"):
        _prepare(graph)


@pytest.mark.parametrize("expression", (
    "16777217", "v**16777217", "v**(16777217)", "v**(-16777217)",
    "v^(16777217)", "v**(16777217 + 1)",
))
def test_inexact_integer_tokens_are_rejected_without_changing_structure(expression):
    source = NEURON.replace("+ 0.1)", f"+ {expression})")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    with pytest.raises(PrecisionResolutionError, match="integer token"):
        _prepare(graph)
    _, models, _, _ = _prepare(graph, PrecisionProfile.FLOAT64)
    assert expression in models["neuron"].dynamics["v"]


def test_float64_also_rejects_unrepresentable_integer_tokens():
    source = NEURON.replace("+ 0.1)", "+ v**9007199254740993)")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    with pytest.raises(PrecisionResolutionError, match="integer token"):
        _prepare(graph, PrecisionProfile.FLOAT64)


@pytest.mark.parametrize("literal", ("1e-999", "-1e-999", "0.00001e-999"))
def test_original_nonzero_literal_cannot_silently_underflow_in_host_parser(literal):
    source = NEURON.replace("+ 0.1)", f"+ {literal})")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    with pytest.raises(PrecisionResolutionError, match="before target rounding"):
        _prepare(graph, PrecisionProfile.FLOAT64)


@pytest.mark.parametrize("literal", ("0e-999", "-0e-999", "0.0e-999"))
def test_exact_zero_literals_with_small_exponents_remain_valid(literal):
    source = NEURON.replace("+ 0.1)", f"+ {literal})")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    _, models, _, _ = _prepare(graph)
    assert literal in models["neuron"].dynamics["v"]


def test_synapse_bindings_initial_state_and_literals_are_rounded():
    graph = _graph(
        synapses=(GraphSynapse("synapse", SYNAPSE),),
        edges=(GraphEdge(
            0, 0, 0, 0.1, synapse="synapse", synapse_bindings={"tau_s": 4.1},
            initial=(0.1,), receptor="v", output="current",
        ),),
    )
    prepared, _, synapses, observed = _prepare(graph)
    edge = prepared.edges[0]
    profile = PrecisionProfile.FLOAT32
    assert edge.synapse_bindings == {"tau_s": profile.round_real(4.1)}
    assert edge.initial == (profile.round_real(0.1),)
    synapse = synapses["synapse"]
    literal = repr(profile.round_real(0.1))
    assert synapse.dynamics["s"] == f"-s/tau_s + {literal}"
    assert synapse.spike_update == f"s + {literal}*w"
    assert synapse.outputs["current"] == f"{literal}*s"
    assert all("defaults.tau_s" not in path for path, _, _, _ in observed)


def test_folded_synapse_effective_defaults_are_rounded():
    graph = replace(
        _graph(synapses=(GraphSynapse("synapse", SYNAPSE),)),
        nodes=(GraphNode(
            0, "neuron", (0.1, 0.1), {}, synapse="synapse",
            receptor="v", output="current",
        ),),
    )
    prepared, _, synapses, _ = _prepare(graph)
    value = PrecisionProfile.FLOAT32.round_real(3.1)
    assert prepared.nodes[0].synapse_bindings == {"tau_s": value}
    assert synapses["synapse"].parameters[0].default == value


@pytest.mark.parametrize("rule", (
    PairSTDP(learning_rate=0.1), TripletSTDP(learning_rate=0.1),
    ModulatedSTDP(learning_rate=0.1), VoltageModulatedSTDP(learning_rate=0.1),
    SoftExcursionModulated(learning_rate=0.1),
))
def test_learning_parameters_use_model_precision_in_mixed_profile(rule):
    prepared, _, _, _ = _prepare(
        _graph(edges=(GraphEdge(0, 0, 0, 0.5, plasticity=rule),)),
        PrecisionProfile.FLOAT32_TIME64,
    )
    target = prepared.edges[0].plasticity
    assert target is not rule
    assert type(target) is type(rule)
    assert target.learning_rate == PrecisionProfile.FLOAT32.round_real(0.1)
    assert target.learning_rate != rule.learning_rate


def test_learning_bounds_that_collapse_in_target_precision_are_rejected():
    rule = PairSTDP(bounds=(1.0, 1.0 + 2**-25))
    with pytest.raises(PrecisionResolutionError, match="bounds require minimum"):
        _prepare(_graph(edges=(GraphEdge(0, 0, 0, 1.0, plasticity=rule),)))
    assert rule.bounds[0] < rule.bounds[1]


@pytest.mark.parametrize("encoder,time_fields,real_fields", (
    (NativeEventEncoder(), (), ()),
    (RegularRateEncoder(0.1, 1.1), (), ("min_rate", "max_rate")),
    (PoissonRateEncoder(0.1, 1.1), (), ("min_rate", "max_rate")),
    (
        TTFSEncoder(0.1, 1.1, amplitude=0.1),
        ("min_latency", "max_latency"), ("amplitude",),
    ),
    (BurstEncoder(0.1, 1.1, 0.1), ("duration",), ("min_rate", "max_rate")),
    (
        LatencyBurstEncoder(0.1, 1.1, 0.1, 0.1),
        ("min_latency", "max_latency", "duration"), ("rate",),
    ),
    (HeldCurrentEncoder(0.1, 0.1, 0.1), (), ("gain", "offset", "baseline")),
))
def test_encoder_fields_match_native_time_and_model_roles(
    encoder, time_fields, real_fields,
):
    graph = _graph(input_ports=(
        InputPort("input", 0, InputMode.SPIKE, encoder=encoder),
    ))
    prepared, _, _, _ = _prepare(graph, PrecisionProfile.FLOAT32_TIME64)
    target = prepared.input_ports[0].encoder
    for name in time_fields:
        assert getattr(target, name) == getattr(encoder, name)
    for name in real_fields:
        assert getattr(target, name) == PrecisionProfile.FLOAT32.round_real(
            getattr(encoder, name)
        )


@pytest.mark.parametrize("decoder,time_fields,real_fields", (
    (RateDecoder(RateMode.SLIDING, width=0.1, origin=0.1), ("width", "origin"), ()),
    (TTFSDecoder(), (), ()),
    (TemporalWeightDecoder(0.1), (), ("tau",)),
))
def test_decoder_fields_match_native_time_and_model_roles(
    decoder, time_fields, real_fields,
):
    graph = _graph(output_ports=(OutputPort("output", 0, decoder),))
    prepared, _, _, _ = _prepare(graph, PrecisionProfile.FLOAT32_TIME64)
    target = prepared.output_ports[0].decoder
    for name in time_fields:
        assert getattr(target, name) == getattr(decoder, name)
    for name in real_fields:
        assert getattr(target, name) == PrecisionProfile.FLOAT32.round_real(
            getattr(decoder, name)
        )


@pytest.mark.parametrize("field", ("encoder", "decoder", "plasticity"))
def test_unknown_authoring_types_fail_closed(field):
    if field == "encoder":
        graph = _graph(input_ports=(
            InputPort("input", 0, InputMode.SPIKE, encoder=object()),
        ))
    elif field == "decoder":
        graph = _graph(output_ports=(OutputPort("output", 0, object()),))
    else:
        graph = _graph(edges=(GraphEdge(0, 0, 0, 0.5, plasticity=object()),))
    with pytest.raises(PrecisionResolutionError, match="unsupported"):
        _prepare(graph)


@pytest.mark.parametrize("initial", (1e-100, 1e100, float("nan"), float("inf")))
def test_invalid_target_initial_state_is_rejected(initial):
    graph = replace(_graph(), nodes=(GraphNode(0, "neuron", initial, {}),))
    with pytest.raises(PrecisionResolutionError):
        _prepare(graph)


def test_long_timestamps_and_small_model_values_have_distinct_ranges():
    graph = _graph(edges=(GraphEdge(0, 0, 0, 0.1, 1e100),))
    prepared, _, _, _ = _prepare(graph, PrecisionProfile.FLOAT32_TIME64)
    assert prepared.edges[0].delay == 1e100
    with pytest.raises(PrecisionResolutionError, match="delay overflows"):
        _prepare(graph, PrecisionProfile.FLOAT32)


def test_subclasses_with_unknown_numeric_fields_fail_closed():
    class FutureEncoder(RegularRateEncoder):
        extra_gain = 1e100

    graph = _graph(input_ports=(
        InputPort("input", 0, InputMode.SPIKE, encoder=FutureEncoder(0.1, 1.1)),
    ))
    with pytest.raises(PrecisionResolutionError, match="unsupported codec"):
        _prepare(graph)


def test_shared_source_is_parsed_once_and_cached_ir_stays_unchanged(monkeypatch):
    import lacuna.precision_authoring as authoring

    parser = authoring.parse_neuron
    cached = parser(NEURON)
    calls = []

    def counted_parser(source):
        calls.append(source)
        return parser(source)

    monkeypatch.setattr(authoring, "parse_neuron", counted_parser)
    graph = replace(
        _graph(),
        models=(GraphModel("neuron", NEURON), GraphModel("second", NEURON)),
        nodes=(
            GraphNode(0, "neuron", 0.0, {}),
            GraphNode(1, "second", 0.0, {"drive": 0.25}),
        ),
    )
    prepared, models, _, _ = _prepare(graph)
    assert calls == [NEURON]
    assert cached.dynamics["v"] == "(-v + drive + 0.1) / tau"
    assert cached.parameters[1].default == 0.1
    assert models["neuron"].parameters[1].default != 0.1
    assert models["second"].parameters[1].default == 0.1
    assert prepared.nodes[1].bindings["drive"] == 0.25


@pytest.mark.parametrize("literal", ("1e-999", "-1e-999", "0x1p-9999"))
@pytest.mark.parametrize("overridden", (False, True))
def test_source_defaults_lost_by_host_parser_are_rejected(literal, overridden):
    source = NEURON.replace("drive = 0.1", f"drive = {literal}")
    graph = replace(
        _graph(), models=(GraphModel("neuron", source),),
        nodes=(GraphNode(
            0, "neuron", 0.0, {"drive": 0.25} if overridden else {},
        ),),
    )
    with pytest.raises(PrecisionResolutionError, match="defaults.drive.*source parser"):
        _prepare(graph)


@pytest.mark.parametrize("literal", ("1e-999", "0x1p-9999"))
def test_refractory_literals_lost_by_host_parser_are_rejected(literal):
    source = NEURON.replace("duration = 0.1", f"duration = {literal}")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    with pytest.raises(PrecisionResolutionError, match="refractory.*source parser"):
        _prepare(graph)


@pytest.mark.parametrize("literal", ("1e-999", "0x1p-9999"))
def test_synapse_default_host_underflow_is_rejected_even_if_unused(literal):
    source = SYNAPSE.replace("3.1", literal)
    graph = _graph(synapses=(GraphSynapse("synapse", source),))
    with pytest.raises(PrecisionResolutionError, match="defaults.tau_s.*source parser"):
        _prepare(graph)


@pytest.mark.parametrize("literal", ("0e-999", "-0e-999", "0x0p-9999"))
def test_true_zero_source_defaults_and_refractory_are_not_underflow(literal):
    source = NEURON.replace("drive = 0.1", f"drive = {literal}")
    source = source.replace("duration = 0.1", f"duration = {literal}")
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    prepared, models, _, _ = _prepare(graph)
    assert prepared.nodes[0].bindings["drive"] == 0.0
    assert models["neuron"].refractory.duration == 0.0


def test_underflow_literals_in_source_comments_are_ignored():
    source = NEURON.replace(
        "params {", "# omitted = 1e-999\n    params {\n# foo = 0x1p-9999\n",
    )
    source = source.replace(
        "refractory { duration = 0.1 }",
        "refractory {\n# duration = 1e-999\n duration = 0.1 }",
    )
    graph = replace(_graph(), models=(GraphModel("neuron", source),))
    prepared, models, _, _ = _prepare(graph)
    assert prepared.nodes[0].bindings["drive"] > 0.0
    assert models["neuron"].refractory.duration > 0.0
