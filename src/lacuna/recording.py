"""High-level recording schedules and resource options."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Sequence

from .errors import ResolutionError
from .ffi import TraceKind


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{context} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{context} must be finite")
    return result


class SamplingSchedule:
    """Generate explicit sample times within a run horizon."""

    def times(self, t_end: float) -> tuple[float, ...]:
        """Return ordered unique sample times through ``t_end``."""

        raise NotImplementedError


@dataclass(frozen=True)
class AtTimes(SamplingSchedule):
    """Sample state at an explicit ordered set of times."""

    values: tuple[float, ...] | Sequence[float]

    def times(self, t_end: float) -> tuple[float, ...]:
        """Validate and return the authored sample times."""

        end = _finite(t_end, "run end")
        result = tuple(_finite(value, "sample time") for value in self.values)
        if any(value < 0.0 or value > end for value in result):
            raise ResolutionError("sample times must lie within the run")
        if tuple(sorted(result)) != result or len(set(result)) != len(result):
            raise ResolutionError("sample times must be unique and ordered")
        return result


@dataclass(frozen=True)
class Every(SamplingSchedule):
    """Sample state at a fixed interval over a bounded range."""

    interval: float
    start: float = 0.0
    stop: float | None = None
    include_stop: bool = True

    def times(self, t_end: float) -> tuple[float, ...]:
        """Expand the interval without accumulating repeated-addition error."""

        interval = _finite(self.interval, "sampling interval")
        start = _finite(self.start, "sampling start")
        end = _finite(t_end, "run end")
        stop = end if self.stop is None else _finite(self.stop, "sampling stop")
        if interval <= 0.0:
            raise ResolutionError("sampling interval must be positive")
        if start < 0.0 or stop < start or stop > end:
            raise ResolutionError("sampling range must satisfy 0 <= start <= stop <= run end")
        count = int(math.floor((stop - start) / interval))
        result = tuple(start + index * interval for index in range(count + 1))
        tolerance = 8.0 * math.ulp(max(1.0, abs(stop)))
        result = tuple(
            stop if abs(value - stop) <= tolerance else value
            for value in result
            if value < stop or value <= stop + tolerance
        )
        if not self.include_stop and result and abs(result[-1] - stop) <= tolerance:
            result = result[:-1]
        return result


@dataclass(frozen=True)
class SpikeRecording:
    """Retain emitted spikes, optionally exposing only a node selection."""

    targets: object | None = None


@dataclass(frozen=True)
class StateRecording:
    """Record selected state variables on an explicit sampling schedule."""

    targets: object
    sampling: SamplingSchedule
    variables: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, SamplingSchedule):
            raise ResolutionError("state recording requires a sampling schedule")
        if self.variables is not None and (
            not self.variables
            or any(not isinstance(name, str) or not name for name in self.variables)
            or len(set(self.variables)) != len(self.variables)
        ):
            raise ResolutionError("recorded state variable names must be unique and nonempty")


@dataclass(frozen=True)
class TraceRecording:
    """Select internal causal records for memory or a persisted trace artifact."""

    targets: object | None = None
    kinds: frozenset[TraceKind] | Sequence[TraceKind | str] | None = None
    capture_state: bool = True
    variables: tuple[str, ...] | None = None
    capacity: int = 4096
    path: str | os.PathLike[str] | None = None
    chunk_records: int = 4096
    compress: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.capture_state, bool) or not isinstance(self.compress, bool):
            raise ResolutionError("trace capture_state and compress must be boolean")
        if (
            not isinstance(self.capacity, int)
            or isinstance(self.capacity, bool)
            or self.capacity < 0
        ):
            raise ResolutionError("trace capacity must be nonnegative")
        if (
            not isinstance(self.chunk_records, int)
            or isinstance(self.chunk_records, bool)
            or self.chunk_records <= 0
        ):
            raise ResolutionError("trace chunk_records must be positive")
        if not self.capture_state and self.variables is not None:
            raise ResolutionError("trace variables require capture_state=True")
        if self.variables is not None and (
            not self.variables
            or any(not isinstance(name, str) or not name for name in self.variables)
            or len(set(self.variables)) != len(self.variables)
        ):
            raise ResolutionError("trace variable names must be unique and nonempty")


@dataclass(frozen=True)
class RecordingPlan:
    """Independent spike, explicit-time state, and causal-trace selections."""

    spikes: SpikeRecording | None = field(default_factory=SpikeRecording)
    states: tuple[StateRecording, ...] | Sequence[StateRecording] = ()
    trace: TraceRecording | None = None

    def __post_init__(self) -> None:
        if self.spikes is not None and not isinstance(self.spikes, SpikeRecording):
            raise ResolutionError("spikes must be SpikeRecording or None")
        if any(not isinstance(item, StateRecording) for item in self.states):
            raise ResolutionError("states must contain StateRecording values")
        if self.trace is not None and not isinstance(self.trace, TraceRecording):
            raise ResolutionError("trace must be TraceRecording or None")


@dataclass(frozen=True)
class RunOptions:
    """Bounded runtime resources and final-state retention policy."""

    queue_capacity: int = 4096
    output_capacity: int = 4096
    encoder_spike_capacity: int = 4096
    encoder_drive_capacity: int = 4096
    decoder_event_capacity: int = 4096
    same_time_cascade_limit: int = 1024
    return_final_state: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.return_final_state, bool):
            raise ResolutionError("return_final_state must be boolean")
        for name, value in vars(self).items():
            if name == "return_final_state":
                continue
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                or (name == "same_time_cascade_limit" and value == 0)
            ):
                raise ResolutionError(f"{name} has an invalid bounded-resource value")


__all__ = [
    "SamplingSchedule",
    "AtTimes",
    "Every",
    "SpikeRecording",
    "StateRecording",
    "TraceRecording",
    "RecordingPlan",
    "RunOptions",
]
