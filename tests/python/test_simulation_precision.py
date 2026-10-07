"""Precision provenance must survive high-level runs and learned snapshots."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from lacuna import (
    Engine, LIF, Network, NetworkBuilder, PrecisionProfile, RecordingConfig,
    SimulationResult, TraceRecord,
)
from lacuna.errors import ResolutionError


def _network(metadata=None):
    builder = NetworkBuilder("provenance", metadata=metadata)
    source = builder.neuron("source", LIF(name="source", drive=24.1))
    target = builder.neuron("target", LIF(name="target", drive=0.0))
    builder.connect(source, target, weight=0.1, delay=0.5)
    builder.output("spikes", source)
    return builder.build()


@pytest.fixture(params=tuple(PrecisionProfile), ids=lambda item: item.value)
def profile_engine(request):
    profile = request.param
    folder = "build" if profile is PrecisionProfile.FLOAT64 else "build-" + profile.value
    libraries = sorted(Path(folder).glob("liblacuna_core.*"))
    if not libraries:
        pytest.skip(f"native {profile.value} build is not available")
    return Engine(libraries[0], precision=profile)


@pytest.mark.parametrize("metadata", (
    None, "float32", {}, {"profile": "float32"},
    {**PrecisionProfile.FLOAT32.to_record(), "time_bits": 64},
    {**PrecisionProfile.FLOAT64.to_record(), "arithmetic_revision": True},
))
def test_malformed_precision_provenance_is_rejected_before_resolution(core, metadata):
    network = _network({"lacuna_precision": metadata})
    engine = Engine(core._lib._name)
    with pytest.raises(ResolutionError, match="invalid lacuna_precision metadata"):
        engine.compile(network)


def test_valid_but_different_profile_provenance_is_rejected(profile_engine):
    other = next(profile for profile in PrecisionProfile if profile is not profile_engine.precision)
    network = _network({"lacuna_precision": other.to_record()})
    with pytest.raises(ResolutionError, match="does not match engine precision"):
        profile_engine.compile(network)


def test_default_float64_result_and_learned_snapshot_have_explicit_provenance(core):
    original_metadata = {"purpose": "example", "nested": {"keep": True}}
    network = _network(original_metadata)
    with Engine(core._lib._name).compile(network) as compiled:
        result = compiled.run(60.0)
    assert result.precision is PrecisionProfile.FLOAT64
    assert result.network is network
    with pytest.raises(FrozenInstanceError):
        result.precision = PrecisionProfile.FLOAT32
    learned = result.learned_network()
    assert learned.metadata == {
        **original_metadata, "lacuna_precision": PrecisionProfile.FLOAT64.to_record(),
    }
    assert network.metadata == original_metadata
    assert tuple(edge.weight for edge in learned.graph.edges) == result.weights


def test_target_runtime_and_checkpoint_use_the_selected_profile(profile_engine, tmp_path):
    profile = profile_engine.precision
    network = _network({"purpose": "retain me"})
    before = network.to_text()
    with profile_engine.compile(network) as compiled:
        assert compiled.precision is profile
        assert compiled.compiled.resolved.precision is profile
        assert compiled.execution_plan.precision is profile
        result = compiled.run(60.0)
        assert result.spikes.events
        assert result.precision is profile
        assert result.weights == (profile.round_real(0.1),)
        assert compiled.run(60.0) == result
        observed = []
        scheduler = compiled.compiled._ensure_compiled_scheduler()
        with scheduler.create_run(compiled.compiled.resolved.initial_values) as run:
            native = run.execute(
                t_end=60.0, recording=RecordingConfig(consumer=observed.append),
            )
        assert observed
        assert all(isinstance(item, TraceRecord) for item in observed)
        assert native.spikes == result.raw.core.spikes
        assert native.weights == result.weights
    assert network.to_text() == before
    learned = result.save_learned_network(tmp_path / "learned.json")
    reloaded = Network.load(tmp_path / "learned.json")
    assert reloaded.metadata == learned.metadata == {
        "purpose": "retain me", "lacuna_precision": profile.to_record(),
    }
    assert tuple(edge.weight for edge in reloaded.graph.edges) == result.weights
    with profile_engine.compile(reloaded) as resumed:
        assert resumed.run(60.0).spikes.events == result.spikes.events
    if profile is not PrecisionProfile.FLOAT64:
        with pytest.raises(ResolutionError, match="does not match engine precision"):
            Engine().compile(reloaded)


def test_result_cannot_be_relabelled_by_supplied_network_metadata(core):
    network = _network()
    with Engine(core._lib._name).compile(network) as compiled:
        result = compiled.run(1.0)
        resolved = compiled.compiled.resolved
    tagged = replace(network, metadata={"lacuna_precision": PrecisionProfile.FLOAT32.to_record()})
    with pytest.raises(ResolutionError, match="target-resolved graph"):
        SimulationResult(tagged, result.raw)
    with pytest.raises(ResolutionError, match="does not match network provenance"):
        SimulationResult(tagged, result.raw, resolved=resolved)
