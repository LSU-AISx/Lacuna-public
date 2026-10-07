from __future__ import annotations

import math
import struct
from dataclasses import asdict, replace

import pytest

import lacuna.tracefile as tracefile
from lacuna import (
    IncompleteTraceArtifactError,
    RecordingConfig,
    SpikeInput,
    TraceArtifactError,
    TraceArtifactIdentityError,
    TraceArtifactMetadata,
    TraceArtifactReader,
    TraceArtifactWriter,
    TraceKind,
    TraceNodeMetadata,
    TracePhase,
    TraceRecord,
    read_trace_artifact,
    write_trace_artifact,
)
from lacuna.ffi import CoreEvaluator

from .test_graph import _graph


def _metadata() -> TraceArtifactMetadata:
    return TraceArtifactMetadata(
        graph_sha256="0" * 64,
        nodes=(
            TraceNodeMetadata(
                node=20,
                state_names=("v", "w"),
                model_hash="1" * 64,
                resolution_key="2" * 64,
                solver_tolerance=1e-12,
            ),
        ),
        extra={"purpose": "round-trip"},
    )


def _records() -> tuple[TraceRecord, ...]:
    close_to_one = math.nextafter(1.0, 2.0)
    return (
        TraceRecord(
            t=0.0,
            sequence=0,
            generation=0,
            kind=TraceKind.INPUT_SPIKE,
            phase=TracePhase.BOUNDARY,
            node=20,
            subject=None,
            value=-0.0,
        ),
        TraceRecord(
            t=close_to_one,
            sequence=1,
            generation=7,
            kind=TraceKind.RESET,
            phase=TracePhase.FIRE,
            node=20,
            subject=9,
            value=close_to_one,
            state_indices=(1, 0),
            state_names=("w", "v"),
            before=(-0.0, close_to_one),
            after=(close_to_one, -0.0),
        ),
        TraceRecord(
            t=close_to_one,
            sequence=2,
            generation=8,
            kind=TraceKind.FINAL_STATE,
            phase=TracePhase.FINAL,
            node=20,
            subject=None,
            value=0.0,
            state_indices=(0,),
            state_names=("v",),
            before=(close_to_one,),
            after=(close_to_one,),
        ),
    )


def _double_bits(value: float) -> bytes:
    return struct.pack("<d", value)


def test_trace_artifact_round_trip_is_lossless_indexed_and_deterministic(
    tmp_path,
) -> None:
    first_path = tmp_path / "first.lctrace"
    second_path = tmp_path / "second.lctrace"
    records = _records()
    summary = {"events": 3, "finished": True}

    write_trace_artifact(
        first_path,
        records,
        _metadata(),
        summary=summary,
        chunk_records=2,
    )
    write_trace_artifact(
        second_path,
        records,
        _metadata(),
        summary=summary,
        chunk_records=2,
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    artifact = read_trace_artifact(
        first_path,
        expected_graph_sha256="0" * 64,
    )
    assert artifact.complete is True
    assert artifact.chunk_count == 2
    assert artifact.metadata == _metadata()
    assert artifact.summary == summary
    assert artifact.records == records
    for expected, observed in zip(records, artifact.records):
        assert _double_bits(observed.t) == _double_bits(expected.t)
        assert _double_bits(observed.value) == _double_bits(expected.value)
        assert tuple(map(_double_bits, observed.before)) == tuple(
            map(_double_bits, expected.before)
        )
        assert tuple(map(_double_bits, observed.after)) == tuple(
            map(_double_bits, expected.after)
        )

    with TraceArtifactReader(first_path) as reader:
        filtered = tuple(
            reader.iter_records(
                t_start=records[1].t,
                t_end=records[1].t,
                nodes=(20,),
                kinds=(TraceKind.RESET,),
            )
        )
    assert filtered == (records[1],)


def test_trace_writer_streams_graph_records_without_buffering_in_run_result(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    selection = RecordingConfig(
        kinds=frozenset(
            {
                TraceKind.INPUT_SPIKE,
                TraceKind.DEPOSIT_APPLY,
                TraceKind.SPIKE,
                TraceKind.RESET,
                TraceKind.FINAL_STATE,
            }
        ),
        nodes=(20,),
        capture_state=True,
        state_indices=(0,),
        capacity=64,
    )
    metadata = TraceArtifactMetadata.from_resolved_graph(
        resolved,
        selection,
        extra={"run": "streaming-test"},
    )
    path = tmp_path / "network.lctrace"

    with TraceArtifactWriter(path, metadata, chunk_records=2) as writer:
        streamed = resolved.run(
            core,
            spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
            t_end=2.0,
            recording=replace(selection, capacity=0, consumer=writer),
        )
        writer.set_summary({"run_stats": asdict(streamed.core.stats)})

    buffered = resolved.run(
        core,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
        t_end=2.0,
        recording=selection,
    )
    assert streamed.trace == ()

    artifact = read_trace_artifact(
        path,
        expected_graph_sha256=metadata.graph_sha256,
    )
    assert artifact.records == buffered.trace
    assert artifact.summary == {"run_stats": asdict(streamed.core.stats)}
    assert artifact.metadata.recorded_nodes == (20,)
    assert artifact.metadata.state_indices == (0,)


def test_incomplete_artifact_recovers_only_fully_flushed_chunks(tmp_path) -> None:
    path = tmp_path / "interrupted.lctrace"
    writer = TraceArtifactWriter(path, _metadata(), chunk_records=2)
    writer.extend(_records())
    writer.abort()

    with pytest.raises(IncompleteTraceArtifactError):
        TraceArtifactReader(path)
    artifact = read_trace_artifact(path, allow_incomplete=True)
    assert artifact.complete is False
    assert artifact.chunk_count == 1
    assert artifact.records == _records()[:2]
    assert artifact.summary == {}

    partial_path = tmp_path / "partial-chunk.lctrace"
    write_trace_artifact(
        partial_path,
        _records(),
        _metadata(),
        chunk_records=2,
        compress=False,
    )
    with TraceArtifactReader(partial_path) as reader:
        second_offset = reader.chunks[1].offset
    partial = partial_path.read_bytes()
    partial_path.write_bytes(
        partial[: second_offset + tracefile._CHUNK_HEADER.size + 1]
    )
    recovered = read_trace_artifact(partial_path, allow_incomplete=True)
    assert recovered.complete is False
    assert recovered.records == _records()[:2]


def test_trace_artifact_detects_identity_and_chunk_corruption(tmp_path) -> None:
    path = tmp_path / "corrupt.lctrace"
    write_trace_artifact(
        path,
        _records(),
        _metadata(),
        chunk_records=2,
        compress=False,
    )

    with pytest.raises(TraceArtifactIdentityError):
        TraceArtifactReader(path, expected_graph_sha256="f" * 64)

    with TraceArtifactReader(path) as reader:
        payload_offset = reader.chunks[0].offset + tracefile._CHUNK_HEADER.size
    data = bytearray(path.read_bytes())
    data[payload_offset] ^= 0x01
    path.write_bytes(data)

    with TraceArtifactReader(path) as reader:
        with pytest.raises(TraceArtifactError, match="checksum"):
            reader.read_all()


def test_trace_metadata_rejects_inconsistent_state_selection() -> None:
    with pytest.raises(ValueError, match="capture_state"):
        TraceArtifactMetadata(
            graph_sha256="0" * 64,
            nodes=(
                TraceNodeMetadata(20, ("v",), "1" * 64, "2" * 64),
            ),
            capture_state=False,
            state_indices=(0,),
        )


def test_empty_selected_trace_has_a_complete_index(tmp_path) -> None:
    path = tmp_path / "empty.lctrace"
    metadata = TraceArtifactMetadata(
        graph_sha256="0" * 64,
        nodes=(TraceNodeMetadata(20, ("v",), "1" * 64, "2" * 64),),
        recorded_kinds=(),
        recorded_nodes=(),
    )
    write_trace_artifact(path, (), metadata)
    artifact = read_trace_artifact(path)
    assert artifact.complete is True
    assert artifact.chunk_count == 0
    assert artifact.records == ()
