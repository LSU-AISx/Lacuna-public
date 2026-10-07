from __future__ import annotations

from dataclasses import replace

import pytest

from lacuna import (
    DriveInput,
    RecordingConfig,
    SpikeInput,
    TraceArtifactMetadata,
    TraceArtifactReader,
    TraceArtifactWriter,
    TraceAuditError,
    TraceKind,
    audit_causal_trace,
    write_trace_artifact,
)
from lacuna.ffi import CoreEvaluator

from .test_graph import _graph


_T_END = 21.0
_SPIKES = (SpikeInput(1.0, "stimulus", 20.0),)
_DRIVES = (DriveInput(5.0, "bias", 20.0),)


def _stream_complete(core: CoreEvaluator, tmp_path, *, chunk_records: int = 2):
    resolved = _graph().resolve()
    recording = RecordingConfig(capacity=0)
    metadata = TraceArtifactMetadata.from_resolved_graph(resolved, recording)
    path = tmp_path / "complete-audit.lctrace"
    with TraceArtifactWriter(
        path, metadata, chunk_records=chunk_records
    ) as writer:
        result = resolved.run(
            core,
            spike_inputs=_SPIKES,
            drive_inputs=_DRIVES,
            t_end=_T_END,
            recording=replace(recording, consumer=writer),
        )
        writer.set_summary({"t_end": _T_END})
    return resolved, result, path


def test_persisted_audit_matches_in_memory_audit_and_public_node_mapping(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved, streamed, path = _stream_complete(core, tmp_path)
    buffered = resolved.run(
        core,
        spike_inputs=_SPIKES,
        drive_inputs=_DRIVES,
        t_end=_T_END,
        recording=RecordingConfig(capacity=256),
    )
    in_memory = audit_causal_trace(
        buffered.core,
        models=resolved.models,
        edges=resolved.edges,
        t_end=_T_END,
        expected_input_count=1,
        expected_drive_count=1,
    )

    with TraceArtifactReader(path) as reader:
        persisted = reader.audit(
            streamed,
            resolved,
            t_end=_T_END,
            expected_input_count=1,
            expected_drive_count=1,
        )

    assert streamed.trace == ()
    assert persisted.record_count == in_memory.record_count
    assert persisted.input_records == in_memory.input_records
    assert persisted.delivery_records == in_memory.delivery_records
    assert persisted.spike_records == in_memory.spike_records
    assert persisted.final_state_records == in_memory.final_state_records
    assert persisted.checks[1:] == in_memory.checks
    assert persisted.checks[0] == "persisted_artifact_identity_and_completeness"


def test_persisted_audit_reads_each_small_chunk_once(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved, result, path = _stream_complete(core, tmp_path, chunk_records=2)

    with TraceArtifactReader(path) as reader:
        original = reader._read_chunk
        observed_sizes: list[int] = []

        def tracked(chunk):
            records = original(chunk)
            observed_sizes.append(len(records))
            return records

        reader._read_chunk = tracked
        report = reader.audit(result, resolved, t_end=_T_END)
        chunk_count = len(reader.chunks)

    assert report.record_count > 2
    assert len(observed_sizes) == chunk_count
    assert max(observed_sizes) <= 2


def test_persisted_audit_rejects_validly_encoded_scheduler_corruption(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    result = resolved.run(
        core,
        spike_inputs=_SPIKES,
        drive_inputs=_DRIVES,
        t_end=_T_END,
        recording=RecordingConfig(capacity=256),
    )
    records = list(result.trace)
    delivery = next(
        index for index, record in enumerate(records)
        if record.kind is TraceKind.DELIVERY
    )
    records[delivery] = replace(
        records[delivery], value=records[delivery].value + 1.0
    )
    path = tmp_path / "corrupt-scheduler.lctrace"
    write_trace_artifact(
        path,
        records,
        TraceArtifactMetadata.from_resolved_graph(resolved),
        chunk_records=3,
    )

    with TraceArtifactReader(path) as reader:
        with pytest.raises(TraceAuditError) as caught:
            reader.audit(result, resolved, t_end=_T_END)
    assert caught.value.invariant == "delivery_conservation"


def test_persisted_audit_rejects_filtered_and_incomplete_artifacts(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    kinds = frozenset(kind for kind in TraceKind if kind is not TraceKind.RESET)
    recording = RecordingConfig(kinds=kinds, capacity=0)
    metadata = TraceArtifactMetadata.from_resolved_graph(resolved, recording)
    filtered_path = tmp_path / "filtered.lctrace"
    with TraceArtifactWriter(filtered_path, metadata, chunk_records=2) as writer:
        filtered_result = resolved.run(
            core,
            spike_inputs=_SPIKES,
            drive_inputs=_DRIVES,
            t_end=_T_END,
            recording=replace(recording, consumer=writer),
        )

    with TraceArtifactReader(filtered_path) as reader:
        with pytest.raises(TraceAuditError) as caught:
            reader.audit(filtered_result, resolved, t_end=_T_END)
    assert caught.value.invariant == "persisted_artifact_identity_and_completeness"

    incomplete_path = tmp_path / "incomplete.lctrace"
    writer = TraceArtifactWriter(
        incomplete_path,
        TraceArtifactMetadata.from_resolved_graph(resolved),
        chunk_records=2,
    )
    writer.extend(filtered_result.trace)
    writer.abort()
    with TraceArtifactReader(incomplete_path, allow_incomplete=True) as reader:
        with pytest.raises(TraceAuditError) as caught:
            reader.audit(filtered_result, resolved, t_end=_T_END)
    assert caught.value.invariant == "persisted_artifact_identity_and_completeness"
