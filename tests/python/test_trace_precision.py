"""Trace storage and reconstruction retain explicit native numeric identity."""

from dataclasses import replace

import pytest

from lacuna import (
    CoreEvaluator, Graph, PrecisionProfile, RecordingConfig, RecordingPlan,
    StateInspectionRequest, TraceArtifactError, TraceArtifactIdentityError,
    TraceArtifactMetadata, TraceArtifactReader, TraceArtifactWriter, TraceKind,
    TracePhase, TraceRecord, TraceRecording, read_trace_artifact,
)
from lacuna import tracefile

from .test_simulation_precision import _network, profile_engine


def _record_trace(engine, path):
    network = _network()
    with engine.compile(network) as compiled:
        resolved = compiled.compiled.resolved
        requests = tuple(StateInspectionRequest(t, resolved.node_ids[0], (0,))
                         for t in (0.1, 1.3, 3.0))
        recording = RecordingConfig(capacity=0)
        metadata = TraceArtifactMetadata.from_resolved_graph(resolved, recording)
        with TraceArtifactWriter(path, metadata, chunk_records=2) as writer:
            result = resolved.run(
                engine.core, t_end=3.0, inspections=requests,
                recording=replace(recording, consumer=writer),
            )
    return network, resolved, result, requests


@pytest.mark.parametrize("incremental", (False, True))
def test_exported_high_level_trace_has_selected_execution_precision(
    profile_engine, tmp_path, incremental,
):
    path = tmp_path / "profile.lctrace"
    recording = RecordingPlan(trace=TraceRecording(path=path, chunk_records=2))
    with profile_engine.compile(_network()) as compiled:
        if incremental:
            with compiled.start_run(3.0, recording=recording) as run:
                assert run.advance(1.0).precision is profile_engine.precision
                result = run.finish()
        else:
            result = compiled.run(3.0, recording=recording)
    artifact = read_trace_artifact(path)
    assert result.precision is artifact.metadata.precision is profile_engine.precision
    if profile_engine.precision is PrecisionProfile.FLOAT64:
        assert "lacuna_precision" not in artifact.metadata.extra
    else:
        assert artifact.metadata.extra["lacuna_precision"] == profile_engine.precision.to_record()
    assert artifact.metadata.to_document()["floating_point"] == "IEEE754_BINARY64_LE"


def test_matching_trace_reconstruction_matches_native_inspection_from_graph_json(
    profile_engine, tmp_path,
):
    path = tmp_path / "reconstruct.lctrace"
    network, resolved, result, requests = _record_trace(profile_engine, path)
    loaded = Graph.from_text(network.graph.to_text())
    with TraceArtifactReader(path) as reader:
        assert reader.metadata.precision is profile_engine.precision
        reconstructor = reader.reconstructor(profile_engine.core, loaded)
        assert reconstructor.resolved.precision is profile_engine.precision
        assert tuple(model.resolution_key for model in reconstructor.resolved.models) == tuple(
            model.resolution_key for model in resolved.models
        )
        restored = reconstructor.reconstruct(requests)
        assert tuple(item.values for item in restored) == tuple(
            item.values for item in result.inspections
        )
        assert tuple(item.t for item in restored) == tuple(
            profile_engine.precision.round_time(item.t) for item in requests
        )


def test_trace_mismatches_fail_before_any_native_reconstruction(
    profile_engine, tmp_path, monkeypatch,
):
    path = tmp_path / "mismatch.lctrace"
    network, resolved, _, _ = _record_trace(profile_engine, path)
    monkeypatch.setattr(profile_engine.core, "advance_analytical_selected",
                        lambda *args, **kwargs: pytest.fail("mismatch executed numerics"))
    other = next(profile for profile in PrecisionProfile if profile is not profile_engine.precision)
    with TraceArtifactReader(path) as reader:
        reader.metadata = replace(reader.metadata, extra={"lacuna_precision": other.to_record()})
        with pytest.raises(TraceArtifactIdentityError, match="C evaluator"):
            reader.reconstructor(profile_engine.core, network.graph)
    if profile_engine.precision is not PrecisionProfile.FLOAT64:
        with TraceArtifactReader(path) as reader:
            with pytest.raises(TraceArtifactIdentityError, match="C evaluator"):
                reader.reconstructor(CoreEvaluator(), resolved)
            with pytest.raises(TraceArtifactIdentityError, match="resolved graph"):
                reader.reconstructor(profile_engine.core, network.graph.resolve())
            reader.metadata = replace(reader.metadata, extra={})
            assert reader.metadata.precision is PrecisionProfile.FLOAT64
            with pytest.raises(TraceArtifactIdentityError, match="C evaluator"):
                reader.reconstructor(profile_engine.core, resolved)


def test_trace_metadata_rejects_conflicting_profiles(profile_engine):
    with profile_engine.compile(_network()) as compiled:
        resolved = compiled.compiled.resolved
        other = next(profile for profile in PrecisionProfile if profile is not profile_engine.precision)
        with pytest.raises(ValueError, match="trace precision"):
            TraceArtifactMetadata.from_resolved_graph(
                resolved, extra={"lacuna_precision": other.to_record()},
            )
        metadata = TraceArtifactMetadata.from_resolved_graph(
            resolved, extra={"keep": True, "lacuna_precision": profile_engine.precision.to_record()},
        )
        assert metadata.extra["keep"] is True
        for bad in (None, "float32", {}, {"profile": "float32"}):
            with pytest.raises((ValueError, TypeError)):
                replace(metadata, extra={"lacuna_precision": bad})


def test_trace_numeric_payload_uses_real_width_and_clock_width(profile_engine, tmp_path):
    profile = profile_engine.precision
    with profile_engine.compile(_network()) as compiled:
        metadata = TraceArtifactMetadata.from_resolved_graph(compiled.compiled.resolved)
    record = TraceRecord(
        t=(2.0 ** 11 if profile is PrecisionProfile.FLOAT16 else 2.0 ** 24) + 1.0,
        sequence=0, generation=0, kind=TraceKind.FINAL_STATE,
        phase=TracePhase.FINAL, node=metadata.nodes[0].node, subject=None,
        value=profile.round_real(0.1), state_indices=(0,), state_names=("v",),
        before=(profile.round_real(-64.9),), after=(profile.round_real(-64.8),),
    )
    path = tmp_path / "width.lctrace"
    if profile in (PrecisionProfile.FLOAT32, PrecisionProfile.FLOAT16):
        with TraceArtifactWriter(path, metadata) as writer:
            with pytest.raises(TraceArtifactError, match="precision"):
                writer.append(record)
        record = replace(record, t=profile.round_time(record.t))
    with TraceArtifactWriter(path, metadata) as writer:
        writer.append(record)
    with TraceArtifactReader(path) as reader:
        assert reader.read_all() == (record,)
    if profile is not PrecisionProfile.FLOAT64:
        with TraceArtifactWriter(path, metadata) as writer:
            for changes in ({"value": 0.1}, {"before": (-64.9,)}, {"after": (-64.8,)}):
                with pytest.raises(TraceArtifactError, match="precision"):
                    writer.append(replace(record, **changes))


def test_reader_rejects_payload_wider_than_declared_profile(profile_engine, tmp_path, monkeypatch):
    if profile_engine.precision is PrecisionProfile.FLOAT64:
        return
    path = tmp_path / "invalid-width.lctrace"
    _, resolved, _, _ = _record_trace(profile_engine, path)
    metadata = TraceArtifactMetadata.from_resolved_graph(resolved)
    record = TraceRecord(1.0, 0, 0, TraceKind.FINAL_STATE, TracePhase.FINAL,
                         resolved.node_ids[0], None, 0.1)
    with monkeypatch.context() as bypass:
        bypass.setattr(tracefile, "_validate_profile_record", lambda *args: None)
        with TraceArtifactWriter(path, metadata) as writer:
            writer.append(record)
    with TraceArtifactReader(path) as reader:
        with pytest.raises(TraceArtifactError, match="precision"):
            reader.read_all()
