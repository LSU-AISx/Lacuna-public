"""Representation preflight must remain distinct from executable lowering."""

from dataclasses import FrozenInstanceError
import importlib.util
import json
from pathlib import Path

import pytest

from lacuna import (
    AdaptiveLIF, Engine, ExponentialCurrent, Graph, GraphModel, GraphNode,
    LIF, NetworkBuilder, PairSTDP, PrecisionProfile, PrecisionResolutionError,
    TargetPrecisionAnalysis, analyze_precision, clear_resolution_cache,
    resolution_cache_disabled,
)
from lacuna.errors import CapabilityError, CoreError, ResolutionError
from lacuna import target_precision

_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "precision_matrix", Path(__file__).with_name("test_execution_plan_matrix.py")
)
_MATRIX = importlib.util.module_from_spec(_MATRIX_SPEC)
_MATRIX_SPEC.loader.exec_module(_MATRIX)
PLAN_CASES = _MATRIX.PLAN_CASES


def _network(*, drive=24.1, delay=0.1):
    builder = NetworkBuilder("target-precision")
    first = builder.neuron("first", LIF(name="first", drive=drive))
    second = builder.neuron("second", LIF(name="second"))
    builder.connect(first, second, weight=0.123, delay=delay)
    builder.output("readout", second)
    return builder.build()


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
@pytest.mark.parametrize("name,factory", PLAN_CASES, ids=[
    name for name, _ in PLAN_CASES
])
def test_matrix_representation_checks_never_claim_native_execution(
    name, factory, profile
):
    graph = factory()
    before = graph.to_text()
    if profile is PrecisionProfile.FLOAT16 and name == "mixed-static-pair-triplet-modulated":
        with pytest.raises(PrecisionResolutionError, match="a2_plus underflows"):
            analyze_precision(graph, profile)
        assert graph.to_text() == before
        return
    report = analyze_precision(graph, profile)
    assert isinstance(report, TargetPrecisionAnalysis)
    assert report.precision is profile
    assert report.values
    assert report.nodes
    assert report.executable is False
    assert len(report.outstanding_checks) == 4
    assert graph.to_text() == before
    assert all(value.bits in (16, 32, 64) for value in report.values)
    assert all(
        value.bits == (profile.time_bits if value.role == "time" else profile.real_bits)
        for value in report.values
    )


def test_report_retains_exact_values_without_a_compilable_graph(core):
    network = _network()
    report = analyze_precision(network, "float32")
    assert report.changed_values
    document = json.loads(json.dumps(report.to_document()))
    assert document["precision"] == PrecisionProfile.FLOAT32.to_record()
    assert document["executable"] is False
    assert "graph" not in document
    for raw, checked in zip(document["values"], report.values):
        assert float.fromhex(raw["source"]) == checked.source_value
        assert float.fromhex(raw["target"]) == checked.target_value
    with pytest.raises(FrozenInstanceError):
        report.executable = True
    with pytest.raises(TypeError, match="ExecutionPlan"):
        core.compile_execution_plan(report)
    with pytest.raises(ResolutionError, match="Network"):
        Engine(core._lib._name).compile(report)
    with pytest.raises(CoreError, match="ABI mismatch"):
        Engine(core._lib._name, precision=report.precision)


def test_default_float64_run_and_image_unchanged_after_preflight(core):
    network = _network()
    before = network.graph.to_text()
    with Engine(core._lib._name).compile(network) as simulation:
        original = simulation.run(100.0)
        image = simulation.compiled_graph_image()
    analyze_precision(network, "float32")
    analyze_precision(network, "float32-time64")
    with Engine(core._lib._name).compile(network) as simulation:
        assert simulation.run(100.0) == original
        assert simulation.compiled_graph_image() == image
    assert network.graph.to_text() == before


def test_profile_source_and_horizon_are_part_of_analysis_identity():
    network = _network()
    reports = [analyze_precision(network, profile) for profile in PrecisionProfile]
    assert len({report.source_hash for report in reports}) == 1
    assert len({report.analysis_key for report in reports}) == 4
    altered = analyze_precision(_network(drive=24.2), "float32")
    assert altered.source_hash != reports[-1].source_hash
    assert altered.analysis_key != reports[-1].analysis_key
    horizon = analyze_precision(network, "float32", time_horizon=100.0)
    assert horizon.analysis_key != reports[-1].analysis_key
    assert horizon.time_horizon == 100.0
    assert analyze_precision(
        network, "float32", time_precision="float64"
    ) == reports[1]


def test_precision_analysis_cache_is_partitioned_and_returns_detached_records(monkeypatch):
    clear_resolution_cache()
    original = target_precision.prepare_precision_graph
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(target_precision, "prepare_precision_graph", counted)
    network = _network()
    first = analyze_precision(network, "float32")
    replay = analyze_precision(network, "float32")
    assert first == replay
    assert first is not replay
    assert calls == [PrecisionProfile.FLOAT32]
    with resolution_cache_disabled():
        assert analyze_precision(network, "float32") == first
    assert len(calls) == 2
    analyze_precision(network, "float32-time64")
    analyze_precision(network, "float64")
    assert len(calls) == 4
    clear_resolution_cache()


@pytest.mark.parametrize("family", ["adaptive", "filtered"])
def test_distinct_reciprocals_that_round_equal_do_not_reuse_host_formula(family):
    tau_m, tau_s = 24.00001335144043, 24.000015258789062
    assert PrecisionProfile.FLOAT32.round_real(tau_m) == tau_m
    assert PrecisionProfile.FLOAT32.round_real(tau_s) == tau_s
    assert -1.0 / tau_m != -1.0 / tau_s
    assert PrecisionProfile.FLOAT32.round_real(-1.0 / tau_m) == (
        PrecisionProfile.FLOAT32.round_real(-1.0 / tau_s)
    )
    builder = NetworkBuilder("rate-collapse")
    if family == "adaptive":
        builder.neuron(
            "adaptive", AdaptiveLIF(tau_m=tau_m, tau_adaptation=tau_s)
        )
    else:
        source = builder.neuron("source", LIF(name="source"))
        target = builder.neuron(
            "target", LIF(name="target", tau_m=tau_m, synaptic_input=True)
        )
        builder.connect(
            source, target, synapse=ExponentialCurrent(tau_s), weight=0.1
        )
    network = builder.build()
    analyze_precision(network, "float64")
    with pytest.raises(PrecisionResolutionError, match="rates collapse"):
        analyze_precision(network, "float32")
    assert not issubclass(PrecisionResolutionError, CapabilityError)


def test_lost_refractory_or_synaptic_delay_fails_conservative_horizon_check():
    network = _network(delay=0.1)
    with pytest.raises(PrecisionResolutionError, match="horizon check"):
        analyze_precision(network, "float32", time_horizon=2**24)
    mixed = analyze_precision(network, "float32-time64", time_horizon=2**24)
    assert mixed.time_horizon == 2**24
    assert mixed.executable is False


def test_rounded_analytical_model_must_not_silently_fall_back_to_stepped():
    builder = NetworkBuilder("equal-adaptation")
    builder.neuron(
        "adaptive", AdaptiveLIF(tau_m=10.0, tau_adaptation=10.00000001)
    )
    network = builder.build()
    assert network.graph.resolve().models[0].dispatch.value == "ROOT_FIND"
    with pytest.raises(PrecisionResolutionError, match="changes the resolved model family"):
        analyze_precision(network, "float32")
    assert network.graph.resolve().models[0].dispatch.value == "ROOT_FIND"


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True])
def test_invalid_horizon_rejected(value):
    with pytest.raises(PrecisionResolutionError):
        analyze_precision(_network(), "float32", time_horizon=value)


def test_rounded_plasticity_bounds_cannot_collapse():
    builder = NetworkBuilder("bounds-collapse")
    first = builder.neuron("first", LIF(name="first"))
    second = builder.neuron("second", LIF(name="second"))
    builder.connect(
        first, second, weight=0.5,
        plasticity=PairSTDP(bounds=(0.5, 0.5 + 1e-9)),
    )
    with pytest.raises(PrecisionResolutionError, match="bounds|bound"):
        analyze_precision(builder.build(), "float32")


def test_derived_stable_rate_must_not_underflow_to_zero():
    source = """
neuron Slow {
    params { gain : positive = 1e-30; tau : positive = 1e30 }
    state { v : membrane }
    dynamics { dv/dt = -gain * v / tau }
    threshold { v > 1.0 }
    reset { v <- 0.0 }
}
"""
    graph = Graph(
        models=(GraphModel("slow", source),),
        nodes=(GraphNode(0, "slow", 0.0, {}),),
    )
    assert graph.resolve().models[0].a < 0.0
    with pytest.raises(PrecisionResolutionError, match="underflows to zero"):
        analyze_precision(graph, "float32")


def test_threshold_and_reset_literals_are_checked_before_resolution():
    source = """
neuron LiteralThreshold {
    params { tau : positive = 10.0 }
    state { v : membrane }
    dynamics { dv/dt = (2.0-v)/tau }
    threshold { v > 1.000000001 }
    reset { v <- 1.0 }
}
"""
    graph = Graph(
        models=(GraphModel("literal", source),),
        nodes=(GraphNode(0, "literal", 0.0, {}),),
    )
    assert graph.resolve().models[0].threshold > 1.0
    with pytest.raises(PrecisionResolutionError, match="reset|threshold"):
        analyze_precision(graph, "float32")


def test_precision_preflight_does_not_accept_compiled_or_arbitrary_objects():
    with pytest.raises(TypeError, match="authored Network or Graph"):
        analyze_precision(object(), "float32")
