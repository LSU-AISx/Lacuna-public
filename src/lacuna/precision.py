"""Precision identities and target-representation authoring checks.

These helpers validate scalar representations, not neuronal execution. A profile
identity does not establish that a native implementation supports that profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import math
from numbers import Real
import struct

from .errors import CapabilityError


class PrecisionProfile(str, Enum):
    """Immutable names for model arithmetic and event-time representations."""

    FLOAT64 = "float64"
    FLOAT32_TIME64 = "float32-time64"
    FLOAT32 = "float32"
    FLOAT16 = "float16"

    @property
    def real_bits(self) -> int:
        """Width of model and learning scalars."""
        return 16 if self is PrecisionProfile.FLOAT16 else 64 if self is PrecisionProfile.FLOAT64 else 32

    @property
    def time_bits(self) -> int:
        """Width of scheduler timestamps and intervals."""
        return 16 if self is PrecisionProfile.FLOAT16 else 32 if self is PrecisionProfile.FLOAT32 else 64

    @property
    def native_id(self) -> int:
        """Fixed-width metadata identifier shared with the native interface."""
        return {
            PrecisionProfile.FLOAT64: 1,
            PrecisionProfile.FLOAT32_TIME64: 2,
            PrecisionProfile.FLOAT32: 3,
            PrecisionProfile.FLOAT16: 4,
        }[self]

    @property
    def arithmetic_revision(self) -> int:
        """Revision of the declared arithmetic contract, not a capability flag."""
        return 1

    @property
    def cache_key(self) -> tuple[str, int, int, int]:
        """Stable identity to include in precision-dependent cache keys."""
        return (
            self.value,
            self.real_bits,
            self.time_bits,
            self.arithmetic_revision,
        )

    def to_record(self) -> dict[str, str | int]:
        """Return versioned metadata suitable for JSON serialization."""
        return {
            "schema_version": 1,
            "profile": self.value,
            "real_bits": self.real_bits,
            "time_bits": self.time_bits,
            "arithmetic_revision": self.arithmetic_revision,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PrecisionProfile:
        """Read metadata without accepting contradictory or unknown fields."""
        if not isinstance(record, Mapping):
            raise TypeError("precision metadata must be a mapping")
        expected_keys = {
            "schema_version",
            "profile",
            "real_bits",
            "time_bits",
            "arithmetic_revision",
        }
        if set(record) != expected_keys:
            raise ValueError("precision metadata fields do not match schema version 1")
        if type(record["profile"]) is not str:
            raise TypeError("precision metadata profile must be a string")
        profile = normalize_precision(record["profile"])
        expected = profile.to_record()
        for field in expected_keys - {"profile"}:
            if type(record[field]) is not int:
                raise TypeError(f"precision metadata {field} must be an integer")
            if record[field] != expected[field]:
                raise ValueError(
                    f"precision metadata {field}={record[field]!r} does not match "
                    f"{profile.value}: expected {expected[field]}"
                )
        return profile

    def round_real(
        self,
        value: Real,
        *,
        name: str = "value",
        allow_zero_underflow: bool = False,
    ) -> float:
        """Convert an authoring value to the model scalar representation."""
        return representable_value(
            value,
            bits=self.real_bits,
            name=name,
            allow_zero_underflow=allow_zero_underflow,
        )

    def round_time(
        self,
        value: Real,
        *,
        name: str = "value",
        allow_zero_underflow: bool = False,
    ) -> float:
        """Convert an authoring value to the event-time representation."""
        return representable_value(
            value,
            bits=self.time_bits,
            name=name,
            allow_zero_underflow=allow_zero_underflow,
        )


def normalize_precision(
    precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
    time_precision: str | None = None,
) -> PrecisionProfile:
    """Resolve supported representation names without claiming execution support.

    ``float32`` alone means strict float32. Adding ``time_precision='float64'``
    selects the mixed profile. The explicit mixed name cannot be paired with
    float32 timestamps.
    """
    if not isinstance(precision, (PrecisionProfile, str)):
        raise TypeError("precision must be a PrecisionProfile or canonical string")
    try:
        profile = PrecisionProfile(precision)
    except ValueError as exc:
        raise ValueError(
            f"unknown precision {precision!r}: expected float64, "
            "float32-time64, float32, or float16"
        ) from exc
    if time_precision is None:
        return profile
    if not isinstance(time_precision, str):
        raise TypeError("time_precision must be 'float64', 'float32', 'float16', or None")
    if time_precision not in {"float64", "float32", "float16"}:
        raise ValueError("time_precision must be 'float64', 'float32', or 'float16'")
    if profile is PrecisionProfile.FLOAT16:
        if time_precision != "float16":
            raise ValueError("float16 model arithmetic requires float16 time")
        return profile
    if time_precision == "float16":
        raise ValueError("float16 time requires float16 model arithmetic")
    if profile is PrecisionProfile.FLOAT64:
        if time_precision != "float64":
            raise ValueError("float64 model arithmetic requires float64 time")
        return profile
    if profile is PrecisionProfile.FLOAT32_TIME64:
        if time_precision != "float64":
            raise ValueError("float32-time64 contradicts float32 time_precision")
        return profile
    if time_precision == "float64":
        return PrecisionProfile.FLOAT32_TIME64
    return profile


def require_supported_precision(
    precision: PrecisionProfile | str = PrecisionProfile.FLOAT64,
    *,
    time_precision: str | None = None,
) -> PrecisionProfile:
    """Validate a native profile before matching it to a runtime library."""
    return normalize_precision(precision, time_precision)


def representable_value(
    value: Real,
    *,
    bits: int,
    name: str = "value",
    allow_zero_underflow: bool = False,
) -> float:
    """Round a real authoring value and reject non-finite or lost nonzero values.

    Inputs first convert to a Python float, then to binary16, binary32 or binary64 using
    the host's IEEE scalar packing. Subnormal values and signed zero are retained.
    This is a representation check, not a target math-library or runtime audit.
    """
    if type(bits) is not int:
        raise TypeError("bits must be 16, 32 or 64")
    if bits not in (16, 32, 64):
        raise ValueError("bits must be 16, 32 or 64")
    if type(allow_zero_underflow) is not bool:
        raise TypeError("allow_zero_underflow must be a bool")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    try:
        host_value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable as finite float{bits}") from exc
    if not math.isfinite(host_value):
        raise ValueError(f"{name} must be finite for float{bits}")
    scalar_format = {16: "!e", 32: "!f", 64: "!d"}[bits]
    try:
        rounded = struct.unpack(
            scalar_format, struct.pack(scalar_format, host_value)
        )[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{name} overflows float{bits}") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{name} overflows float{bits}")
    if value != 0 and rounded == 0.0 and not allow_zero_underflow:
        raise ValueError(f"{name} underflows to zero in float{bits}")
    return rounded


def check_ordered_bounds(
    lower: Real,
    upper: Real,
    *,
    role: str,
    bits: int,
) -> tuple[float, float]:
    """Require strict ordering after conversion, such as reset below threshold.

    Distinct sorted rates can use this check before a distinct-rate form is
    selected. Equal-rate forms require their own target-aware derivation.
    """
    rounded_lower = representable_value(lower, bits=bits, name=f"{role} lower")
    rounded_upper = representable_value(upper, bits=bits, name=f"{role} upper")
    if rounded_lower >= rounded_upper:
        raise ValueError(
            f"{role} requires lower < upper after float{bits} conversion"
        )
    return rounded_lower, rounded_upper


def require_time_progress(
    time: Real,
    interval: Real,
    profile: PrecisionProfile | str,
) -> float:
    """Return the rounded next clock value or reject a lost positive interval.

    Both inputs are converted before addition, and the result is rounded again.
    Checking one point does not establish progress over an entire run horizon.
    """
    selected = normalize_precision(profile)
    current = selected.round_time(time, name="time")
    step = selected.round_time(interval, name="interval")
    if step <= 0.0:
        raise ValueError("interval must be positive")
    next_time = selected.round_time(current + step, name="next time")
    if next_time <= current:
        raise ValueError(
            f"positive interval does not advance time in float{selected.time_bits}"
        )
    return next_time
