"""Versioned, lossless, chunked persistence for Lacuna causal traces."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Sequence

from .ffi import RecordingConfig, TraceKind, TracePhase, TraceRecord
from .ir import ResolvedScalarLIF, ResolvedSteppedNeuron
from .precision import PrecisionProfile

if TYPE_CHECKING:
    from .audit import TraceAuditReport
    from .ffi import CoreEvaluator
    from .ffi import MixedRunResult
    from .graph import Graph, GraphRunResult, ResolvedGraph
    from .reconstruction import TraceReconstructor

TRACE_ARTIFACT_SCHEMA = 1

_FILE_MAGIC = b"LCTRC001"
_TRAILER_MAGIC = b"LCTREND1"
_ENDIAN_MARKER = 0x01020304
_CODEC_RAW = 0
_CODEC_ZLIB = 1

_FILE_HEADER = struct.Struct("<8sHHIII")
_CHUNK_HEADER = struct.Struct("<4sBBHIQQQIII")
_INDEX_HEADER = struct.Struct("<4sI")
_INDEX_ENTRY = struct.Struct("<QIQQQ")
_TRAILER = struct.Struct("<8sQQII")
_DOUBLE = struct.Struct("<d")
_UINT64 = struct.Struct("<Q")
_UINT32 = struct.Struct("<I")

_CHUNK_MAGIC = b"CHNK"
_INDEX_MAGIC = b"INDX"
_UINT64_MAX = 0xFFFFFFFFFFFFFFFF
_UINT32_MAX = 0xFFFFFFFF


class TraceArtifactError(ValueError):
    """A persisted trace is invalid, corrupt, or incompatible."""


class IncompleteTraceArtifactError(TraceArtifactError):
    """A trace has no valid completion index and trailer."""


class TraceArtifactIdentityError(TraceArtifactError):
    """A trace does not belong to the expected graph."""


@dataclass(frozen=True)
class TraceNodeMetadata:
    """Persisted state layout and reconstruction capability for one node."""

    node: int
    state_names: tuple[str, ...]
    model_hash: str
    resolution_key: str
    solver_tolerance: float | None = None


@dataclass(frozen=True)
class TraceArtifactMetadata:
    """Versioned identity and recording contract stored in a trace header."""

    graph_sha256: str
    nodes: tuple[TraceNodeMetadata, ...]
    time_unit: str = "ms"
    recorded_kinds: tuple[str, ...] = tuple(kind.name for kind in TraceKind)
    recorded_nodes: tuple[int, ...] | None = None
    state_indices: tuple[int, ...] | None = None
    capture_state: bool = True
    extra: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_sha256(self.graph_sha256, "graph_sha256")
        if not isinstance(self.time_unit, str) or not self.time_unit:
            raise ValueError("trace artifact time_unit must be a nonempty string")
        if any(not isinstance(item, TraceNodeMetadata) for item in self.nodes):
            raise ValueError("trace artifact nodes must be TraceNodeMetadata values")
        if len({item.node for item in self.nodes}) != len(self.nodes):
            raise ValueError("trace artifact node identifiers must be unique")
        for item in self.nodes:
            if (
                not isinstance(item.node, int)
                or isinstance(item.node, bool)
                or not 0 <= item.node <= _UINT64_MAX
            ):
                raise ValueError("trace artifact node identifiers must be nonnegative integers")
            if not item.state_names or any(
                not isinstance(name, str) or not name for name in item.state_names
            ):
                raise ValueError("trace artifact state names must be nonempty strings")
            _validate_sha256(item.model_hash, "model_hash")
            _validate_sha256(item.resolution_key, "resolution_key")
            if item.solver_tolerance is not None and (
                not isinstance(item.solver_tolerance, (int, float))
                or isinstance(item.solver_tolerance, bool)
                or not math.isfinite(item.solver_tolerance)
                or item.solver_tolerance < 0.0
            ):
                raise ValueError("trace artifact solver tolerances must be nonnegative")
        valid_kinds = {kind.name for kind in TraceKind}
        if (
            any(not isinstance(kind, str) for kind in self.recorded_kinds)
            or len(set(self.recorded_kinds)) != len(self.recorded_kinds)
            or any(kind not in valid_kinds for kind in self.recorded_kinds)
        ):
            raise ValueError("trace artifact recorded_kinds are invalid")
        if self.recorded_nodes is not None:
            known = {item.node for item in self.nodes}
            if (
                any(
                    not isinstance(node, int) or isinstance(node, bool)
                    for node in self.recorded_nodes
                )
                or len(set(self.recorded_nodes)) != len(self.recorded_nodes)
                or any(node not in known for node in self.recorded_nodes)
            ):
                raise ValueError("trace artifact recorded_nodes are invalid")
        if self.state_indices is not None and (
            any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or index < 0
                for index in self.state_indices
            )
            or len(set(self.state_indices)) != len(self.state_indices)
        ):
            raise ValueError("trace artifact state_indices are invalid")
        if not isinstance(self.capture_state, bool):
            raise ValueError("trace artifact capture_state must be boolean")
        if not self.capture_state and self.state_indices is not None:
            raise ValueError(
                "trace artifact state_indices require capture_state to be enabled"
            )
        if self.state_indices is not None:
            selected = (
                self.nodes
                if self.recorded_nodes is None
                else tuple(
                    item for item in self.nodes if item.node in self.recorded_nodes
                )
            )
            if any(
                index >= len(item.state_names)
                for item in selected
                for index in self.state_indices
            ):
                raise ValueError(
                    "trace artifact state_indices are absent from a recorded node"
                )
        if not isinstance(self.extra, Mapping):
            raise ValueError("trace artifact extra metadata must be a mapping")
        if "lacuna_precision" in self.extra:
            PrecisionProfile.from_record(self.extra["lacuna_precision"])
        if self.extra.get("lacuna_numerical_execution", "legacy") not in (
            "legacy", "bounded_dense_v1", "bounded_dense_v2"
        ):
            raise ValueError("unknown numerical execution policy in trace metadata")
        _canonical_json(dict(self.extra))

    @property
    def precision(self) -> PrecisionProfile:
        """Return execution precision, with untagged legacy traces fixed to float64.

        Binary64 trace storage is a lossless transport for every supported
        profile. It does not choose the arithmetic used for reconstruction.
        """

        if "lacuna_precision" not in self.extra:
            return PrecisionProfile.FLOAT64
        return PrecisionProfile.from_record(self.extra["lacuna_precision"])

    @classmethod
    def from_resolved_graph(
        cls,
        resolved,
        recording: RecordingConfig | None = None,
        *,
        extra: Mapping[str, object] | None = None,
    ) -> "TraceArtifactMetadata":
        """Build identity and state metadata for a resolved graph trace."""

        provenance = {} if extra is None else dict(extra)
        if resolved.precision is PrecisionProfile.FLOAT64 and any(
            isinstance(model, ResolvedSteppedNeuron) for model in resolved.models
        ):
            provenance.setdefault("lacuna_numerical_execution", "bounded_dense_v2")
        if "lacuna_precision" in provenance:
            profile = PrecisionProfile.from_record(provenance["lacuna_precision"])
            if profile is not resolved.precision:
                raise ValueError("trace precision does not match the resolved graph")
        elif resolved.precision is not PrecisionProfile.FLOAT64:
            provenance["lacuna_precision"] = resolved.precision.to_record()
        nodes = []
        for node, model in zip(resolved.node_ids, resolved.models):
            names = (
                (model.state_name,)
                if isinstance(model, ResolvedScalarLIF)
                else tuple(model.state_names)
            )
            hint = getattr(model, "root_hint", None)
            tolerance = getattr(hint, "relative_tolerance", None)
            if tolerance is None:
                numerical = getattr(model, "numerical", None)
                tolerance = getattr(numerical, "relative_tolerance", None)
            nodes.append(
                TraceNodeMetadata(
                    node=int(node),
                    state_names=tuple(names),
                    model_hash=model.model_hash,
                    resolution_key=model.resolution_key,
                    solver_tolerance=(
                        None if tolerance is None else float(tolerance)
                    ),
                )
            )
        kinds = (
            tuple(kind.name for kind in TraceKind)
            if recording is None or recording.kinds is None
            else tuple(kind.name for kind in sorted(recording.kinds, key=int))
        )
        return cls(
            graph_sha256=hashlib.sha256(
                resolved.graph.to_text().encode("utf-8")
            ).hexdigest(),
            nodes=tuple(nodes),
            time_unit=resolved.graph.time_unit,
            recorded_kinds=kinds,
            recorded_nodes=(
                None if recording is None else recording.nodes
            ),
            state_indices=(
                None if recording is None else recording.state_indices
            ),
            capture_state=(
                True if recording is None else recording.capture_state
            ),
            extra=provenance,
        )

    def to_document(self) -> dict[str, object]:
        """Convert metadata into canonical JSON-compatible data."""

        return {
            "schema": TRACE_ARTIFACT_SCHEMA,
            "graph_sha256": self.graph_sha256,
            "time_unit": self.time_unit,
            "floating_point": "IEEE754_BINARY64_LE",
            "timestamp_encoding": "IEEE754_XOR_UVARINT",
            "sampling": "SETTLED_POST_EVENT",
            "recorded_kinds": list(self.recorded_kinds),
            "recorded_nodes": (
                None if self.recorded_nodes is None else list(self.recorded_nodes)
            ),
            "state_indices": (
                None if self.state_indices is None else list(self.state_indices)
            ),
            "capture_state": self.capture_state,
            "nodes": [
                {
                    "node": item.node,
                    "state_names": list(item.state_names),
                    "model_hash": item.model_hash,
                    "resolution_key": item.resolution_key,
                    "solver_tolerance": item.solver_tolerance,
                }
                for item in self.nodes
            ],
            "extra": dict(self.extra),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> "TraceArtifactMetadata":
        """Validate and load metadata from a trace header document."""

        if document.get("schema") != TRACE_ARTIFACT_SCHEMA:
            raise TraceArtifactError("unsupported trace artifact schema")
        if (
            document.get("floating_point") != "IEEE754_BINARY64_LE"
            or document.get("timestamp_encoding") != "IEEE754_XOR_UVARINT"
            or document.get("sampling") != "SETTLED_POST_EVENT"
        ):
            raise TraceArtifactError("unsupported trace artifact numeric contract")
        try:
            node_documents = document["nodes"]
            if not isinstance(node_documents, list):
                raise TypeError
            nodes = []
            for item in node_documents:
                if not isinstance(item, dict):
                    raise TypeError
                node = item["node"]
                state_names = item["state_names"]
                model_hash = item["model_hash"]
                resolution_key = item["resolution_key"]
                tolerance = item.get("solver_tolerance")
                if (
                    not isinstance(node, int)
                    or isinstance(node, bool)
                    or not isinstance(state_names, list)
                    or not isinstance(model_hash, str)
                    or not isinstance(resolution_key, str)
                    or (
                        tolerance is not None
                        and (
                            not isinstance(tolerance, (int, float))
                            or isinstance(tolerance, bool)
                        )
                    )
                ):
                    raise TypeError
                nodes.append(
                    TraceNodeMetadata(
                        node=node,
                        state_names=tuple(state_names),
                        model_hash=model_hash,
                        resolution_key=resolution_key,
                        solver_tolerance=(
                            None if tolerance is None else float(tolerance)
                        ),
                    )
                )
            recorded_nodes = document.get("recorded_nodes")
            state_indices = document.get("state_indices")
            extra = document.get("extra", {})
            graph_sha256 = document["graph_sha256"]
            time_unit = document["time_unit"]
            recorded_kinds = document["recorded_kinds"]
            capture_state = document["capture_state"]
            if (
                not isinstance(extra, dict)
                or not isinstance(graph_sha256, str)
                or not isinstance(time_unit, str)
                or not isinstance(recorded_kinds, list)
                or not isinstance(capture_state, bool)
                or (
                    recorded_nodes is not None
                    and not isinstance(recorded_nodes, list)
                )
                or (
                    state_indices is not None
                    and not isinstance(state_indices, list)
                )
            ):
                raise TypeError
            return cls(
                graph_sha256=graph_sha256,
                nodes=tuple(nodes),
                time_unit=time_unit,
                recorded_kinds=tuple(recorded_kinds),
                recorded_nodes=(
                    None if recorded_nodes is None else tuple(recorded_nodes)
                ),
                state_indices=(
                    None if state_indices is None else tuple(state_indices)
                ),
                capture_state=capture_state,
                extra=extra,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceArtifactError("invalid trace artifact metadata") from exc


@dataclass(frozen=True)
class TraceChunkIndex:
    """On-disk location and bounds of one trace record chunk."""

    offset: int
    record_count: int
    first_sequence: int
    first_time: float
    last_time: float


@dataclass(frozen=True)
class TraceArtifact:
    """Complete in-memory representation of a persisted trace artifact."""

    metadata: TraceArtifactMetadata
    records: tuple[TraceRecord, ...]
    summary: Mapping[str, object]
    complete: bool
    chunk_count: int


def _validate_sha256(value: str, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"trace artifact {label} must be a lowercase SHA-256 hex digest")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("trace artifact metadata must be canonical JSON data") from exc


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant {value}")


def _parse_canonical_json(data: bytes, label: str) -> object:
    try:
        document = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TraceArtifactError(f"invalid trace artifact {label} JSON") from exc
    try:
        canonical = _canonical_json(document)
    except ValueError as exc:
        raise TraceArtifactError(f"invalid trace artifact {label} JSON") from exc
    if canonical != data:
        raise TraceArtifactError(f"trace artifact {label} JSON is not canonical")
    return document


def _float_bits(value: float) -> int:
    return _UINT64.unpack(_DOUBLE.pack(float(value)))[0]


def _bits_float(value: int) -> float:
    return _DOUBLE.unpack(_UINT64.pack(value))[0]


def _validate_profile_record(record: TraceRecord, profile: PrecisionProfile) -> None:
    """Reject data that could not have originated in its declared native width."""

    if profile is PrecisionProfile.FLOAT64:
        return
    fields = ((record.t, profile.round_time), *((value, profile.round_real)
              for value in (record.value, *record.before, *record.after)))
    for value, round_value in fields:
        try:
            rounded = round_value(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TraceArtifactError("trace numeric value violates its precision") from exc
        if _float_bits(rounded) != _float_bits(value):
            raise TraceArtifactError("trace numeric value violates its precision")


def _put_uvarint(target: bytearray, value: int) -> None:
    if value < 0 or value > _UINT64_MAX:
        raise ValueError("trace artifact integer is outside uint64")
    while value >= 0x80:
        target.append((value & 0x7F) | 0x80)
        value >>= 7
    target.append(value)


def _get_uvarint(source: memoryview, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if position >= len(source):
            raise TraceArtifactError("truncated trace artifact varint")
        byte = source[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            if value > 0xFFFFFFFFFFFFFFFF:
                raise TraceArtifactError("trace artifact varint exceeds uint64")
            return value, position
        shift += 7
    raise TraceArtifactError("trace artifact varint is too long")


def _pack_records(records: Sequence[TraceRecord]) -> bytes:
    target = bytearray()
    previous_time_bits = 0
    previous_sequence = 0
    for index, record in enumerate(records):
        time_bits = _float_bits(record.t)
        _put_uvarint(target, time_bits ^ previous_time_bits)
        sequence_value = record.sequence if index == 0 else record.sequence - previous_sequence
        _put_uvarint(target, sequence_value)
        _put_uvarint(target, record.generation)
        target.extend((int(record.kind), int(record.phase)))
        _put_uvarint(target, record.node)
        _put_uvarint(target, 0 if record.subject is None else record.subject + 1)
        target.extend(_DOUBLE.pack(record.value))
        count = len(record.state_indices)
        target.append(count)
        for state_index in record.state_indices:
            _put_uvarint(target, state_index)
        for value in record.before:
            target.extend(_DOUBLE.pack(value))
        for value in record.after:
            target.extend(_DOUBLE.pack(value))
        previous_time_bits = time_bits
        previous_sequence = record.sequence
    return bytes(target)


def _unpack_records(
    payload: bytes,
    record_count: int,
    state_names: Mapping[int, tuple[str, ...]],
) -> tuple[TraceRecord, ...]:
    source = memoryview(payload)
    position = 0
    previous_time_bits = 0
    previous_sequence = 0
    records = []
    for index in range(record_count):
        time_xor, position = _get_uvarint(source, position)
        time_bits = previous_time_bits ^ time_xor
        timestamp = _bits_float(time_bits)
        sequence_value, position = _get_uvarint(source, position)
        sequence = sequence_value if index == 0 else previous_sequence + sequence_value
        if sequence > _UINT64_MAX or (index > 0 and sequence_value != 1):
            raise TraceArtifactError("trace artifact sequence is not contiguous")
        generation, position = _get_uvarint(source, position)
        if position + 2 > len(source):
            raise TraceArtifactError("truncated trace artifact record header")
        try:
            kind = TraceKind(source[position])
            phase = TracePhase(source[position + 1])
        except ValueError as exc:
            raise TraceArtifactError("invalid trace kind or phase") from exc
        position += 2
        node, position = _get_uvarint(source, position)
        if node not in state_names:
            raise TraceArtifactError("trace artifact record references an unknown node")
        subject_value, position = _get_uvarint(source, position)
        if position + _DOUBLE.size + 1 > len(source):
            raise TraceArtifactError("truncated trace artifact record payload")
        value = _DOUBLE.unpack_from(source, position)[0]
        position += _DOUBLE.size
        count = source[position]
        position += 1
        if count > 8:
            raise TraceArtifactError("trace artifact state count exceeds the core bound")
        indices = []
        for _ in range(count):
            state_index, position = _get_uvarint(source, position)
            indices.append(state_index)
        if len(set(indices)) != len(indices):
            raise TraceArtifactError("trace artifact state indices are duplicated")
        state_bytes = count * _DOUBLE.size
        if position + 2 * state_bytes > len(source):
            raise TraceArtifactError("truncated trace artifact state payload")
        before = tuple(
            _DOUBLE.unpack_from(source, position + item * _DOUBLE.size)[0]
            for item in range(count)
        )
        position += state_bytes
        after = tuple(
            _DOUBLE.unpack_from(source, position + item * _DOUBLE.size)[0]
            for item in range(count)
        )
        position += state_bytes
        names = state_names.get(node, ())
        if any(state_index >= len(names) for state_index in indices):
            raise TraceArtifactError("trace artifact state index is absent from metadata")
        if (
            not math.isfinite(timestamp)
            or timestamp < 0.0
            or (records and timestamp < records[-1].t)
            or not math.isfinite(value)
            or any(not math.isfinite(item) for item in (*before, *after))
        ):
            raise TraceArtifactError("trace artifact record has an invalid numeric value")
        records.append(
            TraceRecord(
                t=timestamp,
                sequence=sequence,
                generation=generation,
                kind=kind,
                phase=phase,
                node=node,
                subject=None if subject_value == 0 else subject_value - 1,
                value=value,
                state_indices=tuple(indices),
                state_names=tuple(names[item] for item in indices),
                before=before,
                after=after,
            )
        )
        previous_time_bits = time_bits
        previous_sequence = sequence
    if position != len(source):
        raise TraceArtifactError("trace artifact chunk contains trailing record data")
    return tuple(records)


class TraceArtifactWriter:
    """Streaming callback and bounded chunk writer for one causal trace."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        metadata: TraceArtifactMetadata,
        *,
        chunk_records: int = 4096,
        compress: bool = True,
        compression_level: int = 6,
        fsync: bool = False,
    ) -> None:
        if not isinstance(metadata, TraceArtifactMetadata):
            raise TypeError("metadata must be TraceArtifactMetadata")
        if (
            not isinstance(chunk_records, int)
            or isinstance(chunk_records, bool)
            or not 1 <= chunk_records <= _UINT32_MAX
        ):
            raise ValueError("chunk_records must be a positive uint32")
        if not isinstance(compress, bool) or not isinstance(fsync, bool):
            raise ValueError("compress and fsync must be boolean")
        if (
            not isinstance(compression_level, int)
            or isinstance(compression_level, bool)
            or not 0 <= compression_level <= 9
        ):
            raise ValueError("compression_level must be between zero and nine")
        document = metadata.to_document()
        metadata_bytes = _canonical_json(document)
        if len(metadata_bytes) > _UINT32_MAX:
            raise TraceArtifactError("trace artifact metadata exceeds uint32 byte size")
        normalized_document = _parse_canonical_json(metadata_bytes, "metadata")
        if not isinstance(normalized_document, dict):
            raise TraceArtifactError("trace artifact metadata must be an object")
        normalized_metadata = TraceArtifactMetadata.from_document(normalized_document)
        self.path = Path(path)
        self.metadata = normalized_metadata
        self.chunk_records = chunk_records
        self.compress = compress
        self.compression_level = compression_level
        self.fsync = fsync
        self._file = self.path.open("wb")
        self._buffer: list[TraceRecord] = []
        self._chunks: list[TraceChunkIndex] = []
        self._nodes = {item.node: item for item in normalized_metadata.nodes}
        self._recorded_kinds = frozenset(normalized_metadata.recorded_kinds)
        self._recorded_nodes = (
            None
            if normalized_metadata.recorded_nodes is None
            else frozenset(normalized_metadata.recorded_nodes)
        )
        self._state_indices = (
            None
            if normalized_metadata.state_indices is None
            else frozenset(normalized_metadata.state_indices)
        )
        self._summary: Mapping[str, object] = {}
        self._next_sequence = 0
        self._last_time = -math.inf
        self._closed = False
        self._complete = False
        metadata_crc = zlib.crc32(metadata_bytes) & 0xFFFFFFFF
        try:
            self._file.write(
                _FILE_HEADER.pack(
                    _FILE_MAGIC,
                    TRACE_ARTIFACT_SCHEMA,
                    0,
                    _ENDIAN_MARKER,
                    len(metadata_bytes),
                    metadata_crc,
                )
            )
            self._file.write(metadata_bytes)
            self._file.flush()
            if self.fsync:
                os.fsync(self._file.fileno())
        except Exception:
            self._file.close()
            self._closed = True
            raise

    @property
    def closed(self) -> bool:
        """Return whether the artifact file is closed."""

        return self._closed

    @property
    def complete(self) -> bool:
        """Return whether a completion index and trailer were written."""

        return self._complete

    @property
    def record_count(self) -> int:
        """Return the number of accepted trace records."""

        return self._next_sequence

    def set_summary(self, summary: Mapping[str, object]) -> None:
        """Set canonical summary data before closing the artifact."""

        if self._closed:
            raise RuntimeError("trace artifact writer is closed")
        if not isinstance(summary, Mapping):
            raise TypeError("trace artifact summary must be a mapping")
        normalized = dict(summary)
        summary_bytes = _canonical_json(normalized)
        parsed = _parse_canonical_json(summary_bytes, "summary")
        if not isinstance(parsed, dict):
            raise ValueError("trace artifact summary must be an object")
        self._summary = parsed

    def __call__(self, record: TraceRecord) -> None:
        self.append(record)

    def append(self, record: TraceRecord) -> None:
        """Validate and buffer one chronological trace record."""

        if self._closed:
            raise RuntimeError("trace artifact writer is closed")
        self._validate_record(record)
        self._buffer.append(record)
        self._next_sequence += 1
        self._last_time = record.t
        if len(self._buffer) >= self.chunk_records:
            self.flush_chunk()

    def extend(self, records: Iterable[TraceRecord]) -> None:
        """Append an iterable of chronological trace records."""

        for record in records:
            self.append(record)

    def _validate_record(self, record: TraceRecord) -> None:
        if not isinstance(record, TraceRecord):
            raise TypeError("trace artifact records must be TraceRecord values")
        if (
            not isinstance(record.sequence, int)
            or isinstance(record.sequence, bool)
            or record.sequence != self._next_sequence
            or record.sequence >= _UINT64_MAX
        ):
            raise ValueError(
                f"trace sequence must be contiguous; expected {self._next_sequence}, "
                f"got {record.sequence}"
            )
        if (
            not isinstance(record.t, (int, float))
            or isinstance(record.t, bool)
            or not math.isfinite(record.t)
            or record.t < 0.0
            or record.t < self._last_time
        ):
            raise ValueError("trace artifact record times must be finite and chronological")
        if (
            not isinstance(record.generation, int)
            or isinstance(record.generation, bool)
            or not 0 <= record.generation <= _UINT64_MAX
        ):
            raise ValueError("trace generation must be an unsigned 64-bit integer")
        if not isinstance(record.kind, TraceKind) or not isinstance(
            record.phase, TracePhase
        ):
            raise ValueError("trace record kind and phase must be typed enum values")
        if record.kind.name not in self._recorded_kinds:
            raise ValueError("trace record kind is excluded by artifact metadata")
        if (
            not isinstance(record.node, int)
            or isinstance(record.node, bool)
            or not 0 <= record.node <= _UINT64_MAX
        ):
            raise ValueError("trace record node must be a nonnegative bounded integer")
        node = self._nodes.get(record.node)
        if node is None:
            raise ValueError(f"trace record references unknown metadata node {record.node}")
        if self._recorded_nodes is not None and record.node not in self._recorded_nodes:
            raise ValueError("trace record node is excluded by artifact metadata")
        if record.subject is not None and (
            not isinstance(record.subject, int)
            or isinstance(record.subject, bool)
            or not 0 <= record.subject < _UINT64_MAX
        ):
            raise ValueError("trace subject must be a nonnegative bounded integer")
        if (
            not isinstance(record.value, (int, float))
            or isinstance(record.value, bool)
            or not math.isfinite(record.value)
        ):
            raise ValueError("trace record value must be finite")
        count = len(record.state_indices)
        if (
            count > 8
            or len(record.before) != count
            or len(record.after) != count
            or len(set(record.state_indices)) != count
            or any(
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(node.state_names)
                for index in record.state_indices
            )
            or (
                not self.metadata.capture_state
                and bool(record.state_indices)
            )
            or (
                self._state_indices is not None
                and any(index not in self._state_indices for index in record.state_indices)
            )
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in (*record.before, *record.after)
            )
        ):
            raise ValueError("trace record state payload is invalid")
        expected_names = tuple(node.state_names[index] for index in record.state_indices)
        if record.state_names and record.state_names != expected_names:
            raise ValueError("trace record state names disagree with artifact metadata")
        _validate_profile_record(record, self.metadata.precision)

    def flush_chunk(self) -> None:
        """Encode and flush the current record buffer as one chunk."""

        if self._closed:
            raise RuntimeError("trace artifact writer is closed")
        if not self._buffer:
            return
        raw = _pack_records(self._buffer)
        if len(raw) > _UINT32_MAX:
            raise TraceArtifactError("trace artifact chunk exceeds uint32 byte size")
        payload = raw
        codec = _CODEC_RAW
        if self.compress:
            compressed = zlib.compress(raw, self.compression_level)
            if len(compressed) < len(raw):
                payload = compressed
                codec = _CODEC_ZLIB
        if len(payload) > _UINT32_MAX:
            raise TraceArtifactError("stored trace artifact chunk exceeds uint32 byte size")
        first = self._buffer[0]
        last = self._buffer[-1]
        offset = self._file.tell()
        raw_crc = zlib.crc32(raw) & 0xFFFFFFFF
        self._file.write(
            _CHUNK_HEADER.pack(
                _CHUNK_MAGIC,
                codec,
                0,
                0,
                len(self._buffer),
                first.sequence,
                _float_bits(first.t),
                _float_bits(last.t),
                len(raw),
                len(payload),
                raw_crc,
            )
        )
        self._file.write(payload)
        self._file.flush()
        if self.fsync:
            os.fsync(self._file.fileno())
        self._chunks.append(
            TraceChunkIndex(
                offset=offset,
                record_count=len(self._buffer),
                first_sequence=first.sequence,
                first_time=first.t,
                last_time=last.t,
            )
        )
        self._buffer.clear()

    def close(self) -> None:
        """Flush records and write the completion index once."""

        if self._closed:
            return
        try:
            self.flush_chunk()
            summary_bytes = _canonical_json(dict(self._summary))
            if len(self._chunks) > _UINT32_MAX or len(summary_bytes) > _UINT32_MAX:
                raise TraceArtifactError(
                    "trace artifact footer component exceeds uint32 byte size"
                )
            payload = bytearray()
            payload.extend(_UINT32.pack(len(self._chunks)))
            for item in self._chunks:
                payload.extend(
                    _INDEX_ENTRY.pack(
                        item.offset,
                        item.record_count,
                        item.first_sequence,
                        _float_bits(item.first_time),
                        _float_bits(item.last_time),
                    )
                )
            payload.extend(_UINT32.pack(len(summary_bytes)))
            payload.extend(summary_bytes)
            if len(payload) > _UINT32_MAX:
                raise TraceArtifactError("trace artifact index exceeds uint32 byte size")
            index_crc = zlib.crc32(payload) & 0xFFFFFFFF
            index_offset = self._file.tell()
            self._file.write(_INDEX_HEADER.pack(_INDEX_MAGIC, len(payload)))
            self._file.write(payload)
            self._file.write(_UINT32.pack(index_crc))
            self._file.write(
                _TRAILER.pack(
                    _TRAILER_MAGIC,
                    index_offset,
                    self._next_sequence,
                    len(payload),
                    index_crc,
                )
            )
            self._file.flush()
            if self.fsync:
                os.fsync(self._file.fileno())
        except Exception:
            self._buffer.clear()
            self._file.close()
            self._closed = True
            self._complete = False
            raise
        else:
            self._file.close()
            self._closed = True
            self._complete = True

    def abort(self) -> None:
        """Close without a footer and preserve only flushed chunks."""

        if self._closed:
            return
        self._buffer.clear()
        try:
            self._file.flush()
            if self.fsync:
                os.fsync(self._file.fileno())
        finally:
            self._file.close()
            self._closed = True
            self._complete = False

    def __enter__(self) -> "TraceArtifactWriter":
        if self._closed:
            raise RuntimeError("trace artifact writer is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


class TraceArtifactReader:
    """Indexed reader for complete files and recoverable completed chunks."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        allow_incomplete: bool = False,
        expected_graph_sha256: str | None = None,
    ) -> None:
        if not isinstance(allow_incomplete, bool):
            raise ValueError("allow_incomplete must be boolean")
        if expected_graph_sha256 is not None:
            _validate_sha256(expected_graph_sha256, "expected_graph_sha256")
        self.path = Path(path)
        self._file = self.path.open("rb")
        self._closed = False
        self.complete = False
        self.summary: Mapping[str, object] = {}
        self.chunks: tuple[TraceChunkIndex, ...] = ()
        try:
            self.metadata, self._data_offset = self._read_header()
            if (
                expected_graph_sha256 is not None
                and self.metadata.graph_sha256 != expected_graph_sha256
            ):
                raise TraceArtifactIdentityError(
                    "trace artifact graph hash does not match the expected graph"
                )
            loaded = self._read_complete_index()
            if loaded is None:
                if not allow_incomplete:
                    raise IncompleteTraceArtifactError(
                        "trace artifact is incomplete; enable recovery explicitly"
                    )
                self.chunks = self._scan_complete_chunks()
            else:
                self.chunks, self.summary = loaded
                self.complete = True
        except Exception:
            self.close()
            raise
        self.record_count = sum(item.record_count for item in self.chunks)

    @property
    def closed(self) -> bool:
        """Return whether the artifact file is closed."""

        return self._closed

    def _read_header(self) -> tuple[TraceArtifactMetadata, int]:
        header = self._file.read(_FILE_HEADER.size)
        if len(header) != _FILE_HEADER.size:
            raise TraceArtifactError("truncated trace artifact header")
        magic, schema, flags, endian, metadata_size, metadata_crc = _FILE_HEADER.unpack(
            header
        )
        if magic != _FILE_MAGIC or schema != TRACE_ARTIFACT_SCHEMA or flags != 0:
            raise TraceArtifactError("unsupported trace artifact header")
        if endian != _ENDIAN_MARKER:
            raise TraceArtifactError("trace artifact endian marker mismatch")
        metadata_bytes = self._file.read(metadata_size)
        if len(metadata_bytes) != metadata_size:
            raise TraceArtifactError("truncated trace artifact metadata")
        if zlib.crc32(metadata_bytes) & 0xFFFFFFFF != metadata_crc:
            raise TraceArtifactError("trace artifact metadata checksum mismatch")
        document = _parse_canonical_json(metadata_bytes, "metadata")
        if not isinstance(document, dict):
            raise TraceArtifactError("trace artifact metadata must be an object")
        metadata = TraceArtifactMetadata.from_document(document)
        return metadata, _FILE_HEADER.size + metadata_size

    def _read_complete_index(
        self,
    ) -> tuple[tuple[TraceChunkIndex, ...], Mapping[str, object]] | None:
        self._file.seek(0, os.SEEK_END)
        size = self._file.tell()
        if size < self._data_offset + _TRAILER.size:
            return None
        self._file.seek(size - _TRAILER.size)
        trailer = self._file.read(_TRAILER.size)
        magic, index_offset, total_records, payload_size, trailer_crc = _TRAILER.unpack(
            trailer
        )
        if magic != _TRAILER_MAGIC:
            return None
        expected_end = (
            index_offset
            + _INDEX_HEADER.size
            + payload_size
            + _UINT32.size
            + _TRAILER.size
        )
        if index_offset < self._data_offset or expected_end != size:
            raise TraceArtifactError("trace artifact footer offset is invalid")
        self._file.seek(index_offset)
        index_header = self._file.read(_INDEX_HEADER.size)
        if len(index_header) != _INDEX_HEADER.size:
            raise TraceArtifactError("truncated trace artifact index")
        index_magic, observed_size = _INDEX_HEADER.unpack(index_header)
        if index_magic != _INDEX_MAGIC or observed_size != payload_size:
            raise TraceArtifactError("trace artifact index header mismatch")
        payload = self._file.read(payload_size)
        checksum_bytes = self._file.read(_UINT32.size)
        if len(payload) != payload_size or len(checksum_bytes) != _UINT32.size:
            raise TraceArtifactError("truncated trace artifact index payload")
        checksum = _UINT32.unpack(checksum_bytes)[0]
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if checksum != actual_crc or trailer_crc != actual_crc:
            raise TraceArtifactError("trace artifact index checksum mismatch")
        chunks, summary = self._parse_index_payload(payload, index_offset)
        if sum(item.record_count for item in chunks) != total_records:
            raise TraceArtifactError("trace artifact total record count mismatch")
        return chunks, summary

    def _parse_index_payload(
        self, payload: bytes, index_offset: int
    ) -> tuple[tuple[TraceChunkIndex, ...], Mapping[str, object]]:
        source = memoryview(payload)
        if len(source) < _UINT32.size:
            raise TraceArtifactError("truncated trace artifact index entries")
        count = _UINT32.unpack_from(source, 0)[0]
        position = _UINT32.size
        chunks = []
        previous_offset = self._data_offset - 1
        expected_sequence = 0
        previous_time = -math.inf
        for _ in range(count):
            if position + _INDEX_ENTRY.size > len(source):
                raise TraceArtifactError("truncated trace artifact index entry")
            offset, records, first_sequence, first_bits, last_bits = _INDEX_ENTRY.unpack_from(
                source, position
            )
            position += _INDEX_ENTRY.size
            first_time = _bits_float(first_bits)
            last_time = _bits_float(last_bits)
            if (
                records == 0
                or not self._data_offset <= offset < index_offset
                or offset <= previous_offset
                or first_sequence != expected_sequence
                or not math.isfinite(first_time)
                or not math.isfinite(last_time)
                or first_time < 0.0
                or first_time > last_time
                or first_time < previous_time
                or first_sequence + records > _UINT64_MAX
            ):
                raise TraceArtifactError("trace artifact index ordering is invalid")
            chunks.append(
                TraceChunkIndex(
                    offset=offset,
                    record_count=records,
                    first_sequence=first_sequence,
                    first_time=first_time,
                    last_time=last_time,
                )
            )
            previous_offset = offset
            expected_sequence = first_sequence + records
            previous_time = last_time
        if position + _UINT32.size > len(source):
            raise TraceArtifactError("truncated trace artifact summary length")
        summary_size = _UINT32.unpack_from(source, position)[0]
        position += _UINT32.size
        if position + summary_size != len(source):
            raise TraceArtifactError("trace artifact summary size mismatch")
        summary = _parse_canonical_json(bytes(source[position:]), "summary")
        if not isinstance(summary, dict):
            raise TraceArtifactError("trace artifact summary must be an object")
        expected_offset = self._data_offset
        for chunk in chunks:
            if chunk.offset != expected_offset:
                raise TraceArtifactError("trace artifact chunks are not contiguous")
            _, _, stored_size, _ = self._read_chunk_header(chunk)
            expected_offset += _CHUNK_HEADER.size + stored_size
        if expected_offset != index_offset:
            raise TraceArtifactError("trace artifact chunk span disagrees with footer")
        return tuple(chunks), summary

    def _scan_complete_chunks(self) -> tuple[TraceChunkIndex, ...]:
        chunks = []
        offset = self._data_offset
        expected_sequence = 0
        previous_time = -math.inf
        self._file.seek(0, os.SEEK_END)
        file_size = self._file.tell()
        while offset + _CHUNK_HEADER.size <= file_size:
            self._file.seek(offset)
            header = self._file.read(_CHUNK_HEADER.size)
            if len(header) != _CHUNK_HEADER.size or header[:4] != _CHUNK_MAGIC:
                break
            unpacked = _CHUNK_HEADER.unpack(header)
            _, _, _, _, count, first_sequence, first_bits, last_bits, _, stored, _ = unpacked
            first_time = _bits_float(first_bits)
            last_time = _bits_float(last_bits)
            if (
                count == 0
                or first_sequence != expected_sequence
                or first_sequence + count > _UINT64_MAX
                or not math.isfinite(first_time)
                or not math.isfinite(last_time)
                or first_time < 0.0
                or first_time > last_time
                or first_time < previous_time
            ):
                raise TraceArtifactError("recovered trace chunk ordering is invalid")
            if offset + _CHUNK_HEADER.size + stored > file_size:
                break
            chunk = TraceChunkIndex(
                offset=offset,
                record_count=count,
                first_sequence=first_sequence,
                first_time=first_time,
                last_time=last_time,
            )
            self._read_chunk(chunk)
            chunks.append(chunk)
            expected_sequence += count
            previous_time = last_time
            offset += _CHUNK_HEADER.size + stored
        return tuple(chunks)

    def _read_chunk_header(
        self, index: TraceChunkIndex
    ) -> tuple[int, int, int, int]:
        self._file.seek(index.offset)
        header = self._file.read(_CHUNK_HEADER.size)
        if len(header) != _CHUNK_HEADER.size:
            raise TraceArtifactError("truncated trace artifact chunk header")
        (
            magic,
            codec,
            flags,
            reserved,
            count,
            first_sequence,
            first_bits,
            last_bits,
            raw_size,
            stored_size,
            raw_crc,
        ) = _CHUNK_HEADER.unpack(header)
        if (
            magic != _CHUNK_MAGIC
            or flags != 0
            or reserved != 0
            or count != index.record_count
            or first_sequence != index.first_sequence
            or first_bits != _float_bits(index.first_time)
            or last_bits != _float_bits(index.last_time)
            or codec not in (_CODEC_RAW, _CODEC_ZLIB)
            or raw_size == 0
            or stored_size == 0
            or (codec == _CODEC_RAW and stored_size != raw_size)
            or (codec == _CODEC_ZLIB and stored_size >= raw_size)
        ):
            raise TraceArtifactError("trace artifact chunk header disagrees with index")
        return codec, raw_size, stored_size, raw_crc

    def _read_chunk(self, index: TraceChunkIndex) -> tuple[TraceRecord, ...]:
        codec, raw_size, stored_size, raw_crc = self._read_chunk_header(index)
        count = index.record_count
        first_sequence = index.first_sequence
        first_bits = _float_bits(index.first_time)
        last_bits = _float_bits(index.last_time)
        stored = self._file.read(stored_size)
        if len(stored) != stored_size:
            raise TraceArtifactError("truncated trace artifact chunk payload")
        if codec == _CODEC_RAW:
            raw = stored
        elif codec == _CODEC_ZLIB:
            try:
                decompressor = zlib.decompressobj()
                raw = decompressor.decompress(stored, raw_size + 1)
                if len(raw) > raw_size or decompressor.unconsumed_tail:
                    raise TraceArtifactError("trace artifact chunk size mismatch")
                raw += decompressor.flush()
            except zlib.error as exc:
                raise TraceArtifactError("trace artifact chunk decompression failed") from exc
            if not decompressor.eof or decompressor.unused_data:
                raise TraceArtifactError("trace artifact compressed stream is invalid")
        if len(raw) != raw_size:
            raise TraceArtifactError("trace artifact chunk size mismatch")
        if zlib.crc32(raw) & 0xFFFFFFFF != raw_crc:
            raise TraceArtifactError("trace artifact chunk checksum mismatch")
        names = {item.node: item.state_names for item in self.metadata.nodes}
        records = _unpack_records(raw, count, names)
        if (
            not records
            or records[0].sequence != first_sequence
            or _float_bits(records[0].t) != first_bits
            or _float_bits(records[-1].t) != last_bits
        ):
            raise TraceArtifactError("decoded trace artifact chunk bounds mismatch")
        recorded_nodes = (
            None
            if self.metadata.recorded_nodes is None
            else frozenset(self.metadata.recorded_nodes)
        )
        state_indices = (
            None
            if self.metadata.state_indices is None
            else frozenset(self.metadata.state_indices)
        )
        for record in records:
            _validate_profile_record(record, self.metadata.precision)
            if (
                record.kind.name not in self.metadata.recorded_kinds
                or (
                    recorded_nodes is not None
                    and record.node not in recorded_nodes
                )
                or (
                    not self.metadata.capture_state
                    and bool(record.state_indices)
                )
                or (
                    state_indices is not None
                    and any(
                        item not in state_indices for item in record.state_indices
                    )
                )
            ):
                raise TraceArtifactError(
                    "trace artifact record disagrees with recording metadata"
                )
        return records

    def iter_records(
        self,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
        nodes: Iterable[int] | None = None,
        kinds: Iterable[TraceKind] | None = None,
    ) -> Iterator[TraceRecord]:
        """Yield records matching optional time, node, and kind filters."""

        if self._closed:
            raise RuntimeError("trace artifact reader is closed")
        low = -math.inf if t_start is None else float(t_start)
        high = math.inf if t_end is None else float(t_end)
        if math.isnan(low) or math.isnan(high) or low > high:
            raise ValueError("trace artifact time range is invalid")
        node_filter = None if nodes is None else frozenset(nodes)
        kind_filter = None if kinds is None else frozenset(kinds)
        if node_filter is not None and any(
            not isinstance(node, int) or isinstance(node, bool) or node < 0
            for node in node_filter
        ):
            raise ValueError("trace artifact node filter is invalid")
        if kind_filter is not None and any(
            not isinstance(kind, TraceKind) for kind in kind_filter
        ):
            raise ValueError("trace artifact kind filter is invalid")
        for chunk in self.chunks:
            if chunk.last_time < low or chunk.first_time > high:
                continue
            for record in self._read_chunk(chunk):
                if record.t < low or record.t > high:
                    continue
                if node_filter is not None and record.node not in node_filter:
                    continue
                if kind_filter is not None and record.kind not in kind_filter:
                    continue
                yield record

    def read_all(self) -> tuple[TraceRecord, ...]:
        """Read all retained records in sequence order."""

        return tuple(self.iter_records())

    def reconstructor(
        self, core: "CoreEvaluator", resolved: "ResolvedGraph | Graph"
    ) -> "TraceReconstructor":
        """Bind this open artifact to its graph and C evaluator for state queries."""

        if self._closed:
            raise RuntimeError("trace artifact reader is closed")
        from .reconstruction import TraceReconstructor

        return TraceReconstructor(core, resolved, self)

    def audit(
        self,
        result: "MixedRunResult | GraphRunResult",
        resolved: "ResolvedGraph",
        *,
        t_end: float,
        expected_input_count: int | None = None,
        expected_drive_count: int | None = None,
    ) -> "TraceAuditReport":
        """Incrementally audit this complete artifact against its C run result."""

        if self._closed:
            raise RuntimeError("trace artifact reader is closed")
        from .audit import audit_persisted_trace

        return audit_persisted_trace(
            self,
            result,
            resolved=resolved,
            t_end=t_end,
            expected_input_count=expected_input_count,
            expected_drive_count=expected_drive_count,
        )

    def close(self) -> None:
        """Close the artifact file once."""

        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "TraceArtifactReader":
        if self._closed:
            raise RuntimeError("trace artifact reader is closed")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def write_trace_artifact(
    path: str | os.PathLike[str],
    records: Iterable[TraceRecord],
    metadata: TraceArtifactMetadata,
    *,
    summary: Mapping[str, object] | None = None,
    chunk_records: int = 4096,
    compress: bool = True,
) -> None:
    """Write trace records atomically as a chunked artifact."""

    with TraceArtifactWriter(
        path,
        metadata,
        chunk_records=chunk_records,
        compress=compress,
    ) as writer:
        writer.extend(records)
        if summary is not None:
            writer.set_summary(summary)


def read_trace_artifact(
    path: str | os.PathLike[str],
    *,
    allow_incomplete: bool = False,
    expected_graph_sha256: str | None = None,
) -> TraceArtifact:
    """Read and validate a complete trace artifact into memory."""

    with TraceArtifactReader(
        path,
        allow_incomplete=allow_incomplete,
        expected_graph_sha256=expected_graph_sha256,
    ) as reader:
        return TraceArtifact(
            metadata=reader.metadata,
            records=reader.read_all(),
            summary=reader.summary,
            complete=reader.complete,
            chunk_count=len(reader.chunks),
        )
