from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import struct

import pytest

from lacuna.errors import CapabilityError
from lacuna.precision import (
    PrecisionProfile,
    check_ordered_bounds,
    normalize_precision,
    representable_value,
    require_supported_precision,
    require_time_progress,
)


@pytest.mark.parametrize(
    "profile,real_bits,time_bits,native_id",
    (
        (PrecisionProfile.FLOAT64, 64, 64, 1),
        (PrecisionProfile.FLOAT32_TIME64, 32, 64, 2),
        (PrecisionProfile.FLOAT32, 32, 32, 3),
        (PrecisionProfile.FLOAT16, 16, 16, 4),
    ),
)
def test_precision_profile_identity(profile, real_bits, time_bits, native_id):
    assert profile.real_bits == real_bits
    assert profile.time_bits == time_bits
    assert profile.native_id == native_id
    assert profile.arithmetic_revision == 1
    assert profile.cache_key == (profile.value, real_bits, time_bits, 1)
    assert normalize_precision(profile) is profile
    assert normalize_precision(profile.value) is profile
    with pytest.raises(AttributeError):
        profile.real_bits = 16


def test_precision_default_is_float64():
    assert normalize_precision() is PrecisionProfile.FLOAT64


def test_supported_precision_keeps_float64_as_the_default_execution_profile():
    assert require_supported_precision() is PrecisionProfile.FLOAT64
    assert require_supported_precision("float64") is PrecisionProfile.FLOAT64
    assert require_supported_precision(
        PrecisionProfile.FLOAT64, time_precision="float64"
    ) is PrecisionProfile.FLOAT64


@pytest.mark.parametrize(
    "precision,time_precision",
    (
        (PrecisionProfile.FLOAT32, None),
        (PrecisionProfile.FLOAT32_TIME64, None),
        (PrecisionProfile.FLOAT16, None),
        ("float16", "float16"),
        ("float32", "float64"),
        ("float32", "float32"),
    ),
)
def test_recognized_profiles_select_native_arithmetic(
    precision, time_precision
):
    assert require_supported_precision(
        precision, time_precision=time_precision,
    ) is normalize_precision(precision, time_precision)


def test_supported_precision_still_rejects_invalid_profile_names():
    with pytest.raises(ValueError, match="unknown precision"):
        require_supported_precision("float8")


@pytest.mark.parametrize(
    "precision,time_precision,expected",
    (
        ("float64", "float64", PrecisionProfile.FLOAT64),
        ("float32", "float64", PrecisionProfile.FLOAT32_TIME64),
        ("float32", "float32", PrecisionProfile.FLOAT32),
        ("float32-time64", "float64", PrecisionProfile.FLOAT32_TIME64),
        ("float16", "float16", PrecisionProfile.FLOAT16),
    ),
)
def test_precision_time_alias(precision, time_precision, expected):
    assert normalize_precision(precision, time_precision) is expected


@pytest.mark.parametrize("precision", ("FLOAT64", "float", "32", "float8", ""))
def test_precision_requires_canonical_name(precision):
    with pytest.raises(ValueError, match="unknown precision"):
        normalize_precision(precision)


@pytest.mark.parametrize("precision", (None, True, 32, 64.0, object()))
def test_precision_rejects_non_string_names(precision):
    with pytest.raises(TypeError, match="precision must be"):
        normalize_precision(precision)


@pytest.mark.parametrize(
    "precision,time_precision",
    (
        ("float64", "float32"),
        ("float32-time64", "float32"),
        ("float32", "float16"),
        ("float32", "FLOAT64"),
    ),
)
def test_precision_rejects_inconsistent_time_precision(precision, time_precision):
    with pytest.raises(ValueError):
        normalize_precision(precision, time_precision)


@pytest.mark.parametrize("time_precision", (32, 64.0, False, object()))
def test_time_precision_rejects_non_string_names(time_precision):
    with pytest.raises(TypeError, match="time_precision must be"):
        normalize_precision("float32", time_precision)


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
def test_precision_record_roundtrip(profile):
    record = profile.to_record()
    assert record == {
        "schema_version": 1,
        "profile": profile.value,
        "real_bits": profile.real_bits,
        "time_bits": profile.time_bits,
        "arithmetic_revision": 1,
    }
    restored = PrecisionProfile.from_record(json.loads(json.dumps(record)))
    assert restored is profile
    assert hash(restored) == hash(profile)
    assert restored.cache_key == profile.cache_key
    record["real_bits"] = 8
    assert profile.real_bits in (16, 32, 64)


def test_precision_record_and_cache_identities_are_distinct_and_stable():
    keys = {profile.cache_key for profile in PrecisionProfile}
    assert len(keys) == 4
    records = [
        json.dumps(profile.to_record(), sort_keys=True, separators=(",", ":"))
        for profile in PrecisionProfile
    ]
    assert records == [
        '{"arithmetic_revision":1,"profile":"float64","real_bits":64,'
        '"schema_version":1,"time_bits":64}',
        '{"arithmetic_revision":1,"profile":"float32-time64","real_bits":32,'
        '"schema_version":1,"time_bits":64}',
        '{"arithmetic_revision":1,"profile":"float32","real_bits":32,'
        '"schema_version":1,"time_bits":32}',
        '{"arithmetic_revision":1,"profile":"float16","real_bits":16,'
        '"schema_version":1,"time_bits":16}',
    ]
    digests = {hashlib.sha256(record.encode()).hexdigest() for record in records}
    assert len(digests) == 4


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", 0),
        ("schema_version", 2),
        ("real_bits", 32),
        ("time_bits", 32),
        ("arithmetic_revision", 0),
        ("arithmetic_revision", 2),
        ("profile", "float16"),
        ("profile", "float32"),
    ),
)
def test_precision_record_rejects_incompatible_metadata(field, value):
    record = PrecisionProfile.FLOAT64.to_record()
    record[field] = value
    with pytest.raises(ValueError):
        PrecisionProfile.from_record(record)


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", True),
        ("real_bits", 64.0),
        ("time_bits", "64"),
        ("arithmetic_revision", True),
        ("profile", None),
    ),
)
def test_precision_record_rejects_wrong_field_types(field, value):
    record = PrecisionProfile.FLOAT64.to_record()
    record[field] = value
    with pytest.raises(TypeError):
        PrecisionProfile.from_record(record)


def test_precision_record_rejects_missing_and_unknown_fields():
    record = PrecisionProfile.FLOAT64.to_record()
    for field in record:
        incomplete = dict(record)
        del incomplete[field]
        with pytest.raises(ValueError, match="metadata fields"):
            PrecisionProfile.from_record(incomplete)
    with pytest.raises(ValueError, match="metadata fields"):
        PrecisionProfile.from_record({**record, "capabilities": "all"})
    with pytest.raises(TypeError, match="must be a mapping"):
        PrecisionProfile.from_record(list(record.items()))


@pytest.mark.parametrize("bits", (32, 64))
@pytest.mark.parametrize("value", (False, True, "1.25", 1 + 0j, None, object()))
def test_representable_value_requires_non_boolean_real_values(bits, value):
    with pytest.raises(TypeError, match="weight must be a real number"):
        representable_value(value, bits=bits, name="weight")


@pytest.mark.parametrize("bits", (32, 64))
@pytest.mark.parametrize("value", (math.nan, math.inf, -math.inf))
def test_representable_value_rejects_non_finite(bits, value):
    with pytest.raises(ValueError, match="weight must be finite"):
        representable_value(value, bits=bits, name="weight")


@pytest.mark.parametrize("value", (3.5e38, -3.5e38, 1e300, -1e300))
def test_float32_rejects_overflow(value):
    with pytest.raises(ValueError, match="weight overflows float32"):
        representable_value(value, bits=32, name="weight")
    assert representable_value(value, bits=64) == value


def test_integer_overflow_has_value_error_with_parameter_name():
    with pytest.raises(ValueError, match="weight is not representable"):
        representable_value(10 ** 400, bits=64, name="weight")


@pytest.mark.parametrize("value", (2.0 ** -150, -(2.0 ** -150), 1e-300))
def test_float32_nonzero_underflow_is_rejected_by_default(value):
    with pytest.raises(ValueError, match="rate underflows to zero in float32"):
        PrecisionProfile.FLOAT32.round_real(value, name="rate")
    assert PrecisionProfile.FLOAT64.round_real(value) == value


def test_source_value_lost_during_host_conversion_is_rejected():
    with pytest.raises(ValueError, match="underflows to zero in float64"):
        representable_value(Fraction(1, 2 ** 1100), bits=64)


@pytest.mark.parametrize("bits", (32, 64))
@pytest.mark.parametrize("value", (0.0, -0.0))
def test_signed_zero_is_preserved(bits, value):
    rounded = representable_value(value, bits=bits)
    assert rounded == 0.0
    assert math.copysign(1.0, rounded) == math.copysign(1.0, value)


def test_explicit_underflow_opt_in_preserves_sign():
    rounded = representable_value(
        -(2.0 ** -150), bits=32, allow_zero_underflow=True
    )
    assert rounded == 0.0
    assert math.copysign(1.0, rounded) == -1.0


@pytest.mark.parametrize(
    "bits,value",
    (
        (32, 2.0 ** -149),
        (32, -(2.0 ** -149)),
        (32, 2.0 ** -126),
        (32, (2.0 - 2.0 ** -23) * 2.0 ** 127),
        (64, 2.0 ** -1074),
        (64, -(2.0 ** -1074)),
        (64, 2.0 ** -1022),
        (64, (2.0 - 2.0 ** -52) * 2.0 ** 1023),
    ),
)
def test_representable_extremes_are_preserved(bits, value):
    assert representable_value(value, bits=bits) == value


def test_float32_rounding_uses_ties_to_even():
    assert representable_value(1.0 + 2.0 ** -24, bits=32) == 1.0
    assert representable_value(1.0 + 3.0 * 2.0 ** -24, bits=32) == (
        1.0 + 2.0 ** -22
    )


def test_profile_rounding_keeps_model_and_clock_roles_separate():
    value = 1.0 + 2.0 ** -24
    mixed = PrecisionProfile.FLOAT32_TIME64
    assert mixed.round_real(value) == 1.0
    assert mixed.round_time(value) == value
    assert PrecisionProfile.FLOAT32.round_time(value) == 1.0
    assert PrecisionProfile.FLOAT64.round_real(value) == value


def test_numpy_real_values_are_accepted():
    np = pytest.importorskip("numpy")
    for value in (np.float32(1.25), np.float64(1.25), np.int64(3)):
        assert representable_value(value, bits=32) == float(value)
    with pytest.raises(TypeError, match="real number"):
        representable_value(np.bool_(True), bits=32)


@pytest.mark.parametrize("bits", (8, 128, 0, -32))
def test_representable_value_rejects_unknown_width(bits):
    with pytest.raises(ValueError, match="bits must be 16, 32 or 64"):
        representable_value(1.0, bits=bits)


@pytest.mark.parametrize("bits", (True, 32.0, "32", None))
def test_representable_value_rejects_non_integer_width(bits):
    with pytest.raises(TypeError, match="bits must be 16, 32 or 64"):
        representable_value(1.0, bits=bits)


def test_underflow_opt_in_must_be_explicit_boolean():
    with pytest.raises(TypeError, match="allow_zero_underflow must be a bool"):
        representable_value(1.0, bits=32, allow_zero_underflow="yes")


def test_reset_threshold_separation_lost_in_float32_is_rejected():
    reset = 1.0
    threshold = 1.0 + 2.0 ** -25
    assert check_ordered_bounds(
        reset, threshold, role="reset/threshold", bits=64
    ) == (reset, threshold)
    with pytest.raises(ValueError, match="reset/threshold requires lower < upper"):
        check_ordered_bounds(reset, threshold, role="reset/threshold", bits=32)


def test_distinct_rates_collapsing_after_conversion_are_rejected():
    slow_rate = -1.0 - 2.0 ** -25
    fast_rate = -1.0
    assert slow_rate < fast_rate
    assert PrecisionProfile.FLOAT32.round_real(slow_rate) == fast_rate
    with pytest.raises(ValueError, match="distinct rates requires lower < upper"):
        check_ordered_bounds(
            slow_rate, fast_rate, role="distinct rates", bits=32
        )


@pytest.mark.parametrize("lower,upper", ((1.0, 1.0), (2.0, 1.0), (-0.0, 0.0)))
def test_ordered_bounds_require_strict_order(lower, upper):
    with pytest.raises(ValueError, match="requires lower < upper"):
        check_ordered_bounds(lower, upper, role="bounds", bits=64)


def test_ordered_bounds_return_converted_values():
    assert check_ordered_bounds(
        1.0 + 2.0 ** -25, 2.0 + 2.0 ** -24, role="bounds", bits=32
    ) == (1.0, 2.0)


@pytest.mark.parametrize("profile", tuple(PrecisionProfile))
def test_time_progress_returns_next_representable_time(profile):
    assert require_time_progress(1.0, 0.5, profile) == 1.5
    assert require_time_progress(-1.0, 1.0, profile) == 0.0


def test_large_clock_loses_small_positive_float32_interval():
    current = 2.0 ** 24
    with pytest.raises(ValueError, match="positive interval does not advance time"):
        require_time_progress(current, 1.0, PrecisionProfile.FLOAT32)
    assert require_time_progress(current, 2.0, "float32") == current + 2.0
    assert require_time_progress(current, 1.0, "float32-time64") == current + 1.0
    assert require_time_progress(current, 1.0, "float64") == current + 1.0


def test_large_float64_clock_also_requires_representable_progress():
    with pytest.raises(ValueError, match="does not advance time in float64"):
        require_time_progress(2.0 ** 53, 1.0, PrecisionProfile.FLOAT64)


def test_time_progress_rounds_inputs_before_addition():
    current = 1.0 + 2.0 ** -24
    step = 2.0 ** -24
    assert struct.unpack("!f", struct.pack("!f", current + step))[0] > 1.0
    with pytest.raises(ValueError, match="does not advance time"):
        require_time_progress(current, step, PrecisionProfile.FLOAT32)


@pytest.mark.parametrize("interval", (0.0, -0.0, -1.0))
def test_time_progress_rejects_non_positive_intervals(interval):
    with pytest.raises(ValueError, match="interval must be positive"):
        require_time_progress(0.0, interval, PrecisionProfile.FLOAT32)


def test_time_progress_rejects_interval_underflow():
    with pytest.raises(ValueError, match="interval underflows to zero"):
        require_time_progress(0.0, 2.0 ** -150, PrecisionProfile.FLOAT32)


@pytest.mark.parametrize("time,interval", ((math.inf, 1.0), (0.0, math.nan)))
def test_time_progress_rejects_non_finite_inputs(time, interval):
    with pytest.raises(ValueError, match="must be finite"):
        require_time_progress(time, interval, PrecisionProfile.FLOAT32)


@pytest.mark.parametrize(
    "profile,current,interval",
    (
        (PrecisionProfile.FLOAT32, 3.0e38, 3.0e38),
        (PrecisionProfile.FLOAT64, 1.0e308, 1.0e308),
    ),
)
def test_time_progress_rejects_next_time_overflow(profile, current, interval):
    with pytest.raises(ValueError, match="next time"):
        require_time_progress(current, interval, profile)
