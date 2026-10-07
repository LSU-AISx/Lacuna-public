"""Per-port encoder and decoder configuration for Lacuna's host C runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TypeAlias


class EncoderKind(IntEnum):
    """C ABI identifiers for supported input encoders."""

    NATIVE_EVENT = 0
    REGULAR_RATE = 1
    POISSON_RATE = 2
    TTFS = 3
    BURST = 4
    LATENCY_BURST = 5
    HELD_CURRENT = 6


@dataclass(frozen=True)
class NativeEventEncoder:
    """Accept an already timestamped spike without changing it."""

    kind: EncoderKind = field(default=EncoderKind.NATIVE_EVENT, init=False)


@dataclass(frozen=True)
class RegularRateEncoder:
    """Map a normalized value linearly to a phase-continuous regular rate."""

    min_rate: float
    max_rate: float
    amplitude: float = 1.0
    kind: EncoderKind = field(default=EncoderKind.REGULAR_RATE, init=False)


@dataclass(frozen=True)
class PoissonRateEncoder:
    """Map a normalized value to an independent deterministic Poisson stream."""

    min_rate: float
    max_rate: float
    amplitude: float = 1.0
    kind: EncoderKind = field(default=EncoderKind.POISSON_RATE, init=False)


@dataclass(frozen=True)
class TTFSEncoder:
    """Emit at most one spike, with larger values producing shorter latency."""

    min_latency: float
    max_latency: float
    amplitude: float = 1.0
    silence_threshold: float = 0.0
    kind: EncoderKind = field(default=EncoderKind.TTFS, init=False)


@dataclass(frozen=True)
class BurstEncoder:
    """Start immediately and encode value as burst rate for a fixed duration."""

    min_rate: float
    max_rate: float
    duration: float
    amplitude: float = 1.0
    kind: EncoderKind = field(default=EncoderKind.BURST, init=False)


@dataclass(frozen=True)
class LatencyBurstEncoder:
    """Encode value as onset latency, then emit a fixed-rate bounded burst."""

    min_latency: float
    max_latency: float
    rate: float
    duration: float
    amplitude: float = 1.0
    silence_threshold: float = 0.0
    kind: EncoderKind = field(default=EncoderKind.LATENCY_BURST, init=False)


@dataclass(frozen=True)
class HeldCurrentEncoder:
    """Apply a zero-order-held, affine-mapped drive during each presentation."""

    gain: float = 1.0
    offset: float = 0.0
    baseline: float = 0.0
    kind: EncoderKind = field(default=EncoderKind.HELD_CURRENT, init=False)


Encoder: TypeAlias = (
    NativeEventEncoder
    | RegularRateEncoder
    | PoissonRateEncoder
    | TTFSEncoder
    | BurstEncoder
    | LatencyBurstEncoder
    | HeldCurrentEncoder
)


class RateMode(IntEnum):
    """Window policy used by a rate decoder."""

    FINITE = 0
    SLIDING = 1
    CUMULATIVE = 2


class EmissionPolicy(IntEnum):
    """Choose when a streaming decoder makes a value externally visible."""

    ON_EVENT = 0
    ON_WINDOW_CLOSE = 1
    ON_EVENT_AND_WINDOW_CLOSE = 2
    ON_QUERY = 3


class DecodeEventKind(IntEnum):
    """Reason a streaming decoder emitted a value."""

    UPDATE = 0
    FINAL = 1
    NO_SPIKE = 2
    QUERY = 3


@dataclass(frozen=True)
class RateDecoder:
    """Decode spike count per unit time over a finite, sliding, or cumulative window."""

    mode: RateMode = RateMode.FINITE
    width: float | None = None
    origin: float = 0.0
    emission: EmissionPolicy = EmissionPolicy.ON_WINDOW_CLOSE


@dataclass(frozen=True)
class TTFSDecoder:
    """Return first-spike latency, optionally normalized by the decode window."""

    normalize: bool = False
    emission: EmissionPolicy = EmissionPolicy.ON_EVENT


class TemporalSpikeMode(str, Enum):
    """Choose which spikes contribute to temporal weighting."""

    ALL = "ALL"
    FIRST = "FIRST"


@dataclass(frozen=True)
class TemporalWeightDecoder:
    """Weight earlier spikes by exp(-(t - window_start) / tau)."""

    tau: float
    spikes: TemporalSpikeMode = TemporalSpikeMode.ALL
    normalize: bool = False
    emission: EmissionPolicy = EmissionPolicy.ON_WINDOW_CLOSE


Decoder: TypeAlias = RateDecoder | TTFSDecoder | TemporalWeightDecoder


@dataclass(frozen=True)
class DecodeWindow:
    """One half-open observation window in a streaming decoder schedule."""

    t_start: float
    t_end: float


@dataclass(frozen=True)
class DecoderWindowBinding:
    """Bind one decoder index and local window id to an observation interval."""

    decoder: int
    window: int
    t_start: float
    t_end: float


@dataclass(frozen=True)
class DecoderQueryBinding:
    """Request one exact-time snapshot from a scheduled ON_QUERY decoder."""

    decoder: int
    window: int
    t: float


@dataclass(frozen=True)
class DecodeQuery:
    """Graph-level query for one local decode window."""

    window: int
    t: float


@dataclass(frozen=True)
class Presentation:
    """Normalized input value presented over a half-open interval."""

    t_start: float
    t_end: float
    encoder: int
    value: float


@dataclass(frozen=True)
class EncodedSpike:
    """Timestamped spike produced by an encoder."""

    t: float
    encoder: int
    value: float


@dataclass(frozen=True)
class EncodedDrive:
    """Timestamped drive update produced by an encoder."""

    t: float
    encoder: int
    value: float


@dataclass(frozen=True)
class EncodedBatch:
    """Spikes and drive updates produced by one encoding call."""

    spikes: tuple[EncodedSpike, ...]
    drives: tuple[EncodedDrive, ...]


@dataclass(frozen=True)
class DecoderBinding:
    """Bind a decoder configuration to one graph node."""

    node: int
    decoder: Decoder


@dataclass(frozen=True)
class DecodeValue:
    """Final value and supporting observations for one decoder window."""

    decoder: int
    valid: bool
    count: int
    value: float | None
    first_spike: float | None
    window: int = 0
    window_start: float | None = None
    window_end: float | None = None


@dataclass(frozen=True)
class DecodeEvent:
    """Streaming value emitted while a decoder window is active."""

    decoder: int
    window: int
    kind: DecodeEventKind
    valid: bool
    emitted_at: float
    source_spike_time: float | None
    window_start: float
    window_end: float
    observed_through: float
    count: int
    value: float | None
    first_spike: float | None
