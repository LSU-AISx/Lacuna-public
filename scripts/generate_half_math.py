#!/usr/bin/env python3
"""Generate binary16 math tables using high-precision interval enclosures.

This is a host-side development utility, not part of the embedded runtime.
Every finite result must have interval endpoints that round to the same half.
Ambiguous intervals are reevaluated with additional precision or rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp


BLOCK_BITS = 6
BLOCK_SIZE = 1 << BLOCK_BITS
NAN = 0x7E00
INF = 0x7C00
SIGN = 0x8000
ROWS = (
    ("exp", None), ("expm1", None), ("log", None), ("log1p", None),
    ("sqrt", None), ("sin", None), ("cos", None), ("tanh", None),
    ("phi1", None), ("phi1_deriv", None),
    ("pow_neg1", 0xBC00), ("pow_2", 0x4000), ("pow_3", 0x4200),
    ("pow_4", 0x4400), ("pow_5", 0x4500), ("pow_6", 0x4600),
    ("pow_7", 0x4700), ("pow_8", 0x4800), ("pow_neg2", 0xC000),
    ("pow_neg_point2", 0xB266), ("pow_point2", 0x3266),
)


def half_value(bits: int):
    exponent = (bits >> 10) & 31
    fraction = bits & 1023
    if exponent == 31:
        return mp.nan if fraction else (-mp.inf if bits & SIGN else mp.inf)
    value = mp.ldexp(mp.mpf(fraction if exponent == 0 else 1024 + fraction),
                     -24 if exponent == 0 else exponent - 25)
    return -value if bits & SIGN else value


def nearest_integer(value) -> int:
    lower = int(mp.floor(value))
    fraction = value - lower
    return lower + int(fraction > mp.mpf("0.5") or
                       (fraction == mp.mpf("0.5") and lower & 1))


def round_half(value) -> int:
    if mp.isnan(value):
        return NAN
    sign = SIGN if value < 0 else 0
    magnitude = abs(value)
    if mp.isinf(magnitude):
        return sign | INF
    if magnitude == 0:
        return sign
    if magnitude < mp.ldexp(mp.mpf(1), -14):
        return sign | nearest_integer(mp.ldexp(magnitude, 24))
    _, exponent = mp.frexp(magnitude)
    significand = nearest_integer(mp.ldexp(magnitude, 11 - exponent))
    if significand == 2048:
        significand = 1024
        exponent += 1
    encoded_exponent = exponent + 14
    if encoded_exponent >= 31:
        return sign | INF
    return sign | (encoded_exponent << 10) | (significand - 1024)


def special_result(name: str, bits: int, exponent_bits: int | None):
    magnitude = bits & 0x7FFF
    negative = bool(bits & SIGN)
    if magnitude > INF:
        return NAN
    if exponent_bits is not None:
        exponent = half_value(exponent_bits)
        odd = exponent == int(exponent) and int(exponent) & 1
        sign = SIGN if negative and odd else 0
        if magnitude == 0:
            return sign | (INF if exponent < 0 else 0)
        if magnitude == INF:
            return sign | (0 if exponent < 0 else INF)
        if negative and exponent != int(exponent):
            return NAN
        return None
    if magnitude == 0:
        if name == "phi1_deriv":
            return 0x3800
        if name in ("expm1", "log1p", "sqrt", "sin", "tanh"):
            return bits
        if name == "log":
            return SIGN | INF
        return 0x3C00
    if magnitude == INF:
        if name in ("phi1", "phi1_deriv"):
            return 0 if negative else INF
        if name == "exp":
            return 0 if negative else INF
        if name == "expm1":
            return 0xBC00 if negative else INF
        if name == "tanh":
            return 0xBC00 if negative else 0x3C00
        if name in ("sin", "cos") or negative:
            return NAN
        return INF
    if negative and name in ("log", "sqrt"):
        return NAN
    if name == "log1p" and negative and magnitude >= 0x3C00:
        return SIGN | INF if magnitude == 0x3C00 else NAN
    return None


def evaluate_interval(name: str, value, exponent_bits: int | None):
    x = mp.iv.mpf(value)
    if exponent_bits is not None:
        return mp.iv.power(x, mp.iv.mpf(half_value(exponent_bits)))
    if name == "log":
        return mp.iv.ln(x)
    if name == "tanh":
        tail = mp.iv.exp(-2 * abs(x))
        positive = (1 - tail) / (1 + tail)
        return -positive if value < 0 else positive
    if name == "phi1":
        return mp.iv.expm1(x) / x
    if name == "phi1_deriv":
        return (x * mp.iv.exp(x) - mp.iv.expm1(x)) / (x * x)
    return getattr(mp.iv, name)(x)


def certified_result(name: str, bits: int, exponent_bits: int | None,
                     base_precision: int) -> tuple[int, int]:
    special = special_result(name, bits, exponent_bits)
    if special is not None:
        return special, 0
    value = half_value(bits)
    for precision in (base_precision, base_precision * 2, base_precision * 4):
        mp.mp.prec = precision + 64
        mp.iv.prec = precision
        enclosure = evaluate_interval(name, value, exponent_bits)
        lower = mp.mpf(enclosure._mpi_[0])
        upper = mp.mpf(enclosure._mpi_[1])
        low_bits, high_bits = round_half(lower), round_half(upper)
        if low_bits == high_bits:
            return low_bits, precision
    raise ArithmeticError(f"Unresolved half rounding: {name} input=0x{bits:04x}")


def lines_of_values(values, count=12):
    return ["    " + ", ".join(f"0x{value:04x}U" for value in values[start:start + count]) + ","
            for start in range(0, len(values), count)]


def generate(destination: Path, precision: int, verify: bool = False):
    if precision < 80:
        raise ValueError("At least 80 interval bits are required")
    mp.mp.prec = precision + 64
    blocks = []
    block_ids = {}
    indexes = []
    row_digests = {}
    maximum_precision = 0
    for name, exponent in ROWS:
        values = []
        for bits in range(65536):
            result, used_precision = certified_result(name, bits, exponent, precision)
            maximum_precision = max(maximum_precision, used_precision)
            values.append(result)
        packed = b"".join(value.to_bytes(2, "little") for value in values)
        row_digests[name] = hashlib.sha256(packed).hexdigest()
        for start in range(0, 65536, BLOCK_SIZE):
            block = tuple(values[start:start + BLOCK_SIZE])
            if block not in block_ids:
                block_ids[block] = len(blocks)
                blocks.append(block)
            indexes.append(block_ids[block])
        print(f"{name}: {len(blocks)} shared blocks", flush=True)
    if len(blocks) > 65536:
        raise OverflowError("The shared block index does not fit uint16_t")
    output = ["/* Generated by scripts/generate_half_math.py. Do not edit. */",
              f"#define LC_HALF_TABLE_BLOCK_BITS {BLOCK_BITS}U",
              f"#define LC_HALF_TABLE_BLOCK_MASK {BLOCK_SIZE - 1}U",
              f"#define LC_HALF_TABLE_ROW_BLOCKS {65536 // BLOCK_SIZE}U"]
    for index, (name, _) in enumerate(ROWS):
        output.append(f"#define LC_HALF_ROW_{name.upper()} {index}U")
    output.append(f"static const uint16_t lc_half_block_index[{len(indexes)}] = {{")
    output.extend(lines_of_values(indexes))
    output.append("};")
    flattened = [value for block in blocks for value in block]
    output.append(f"static const uint16_t lc_half_blocks[{len(flattened)}] = {{")
    output.extend(lines_of_values(flattened))
    output.append("};")
    contents = "\n".join(output) + "\n"
    report = {
        "format": 1, "mpmath_version": mp.__version__,
        "rounding": "nearest_ties_even", "interval_precision_bits": precision,
        "maximum_interval_precision_bits": maximum_precision,
        "verification": "Both high-precision interval endpoints round to the same binary16 value",
        "entries_checked": len(ROWS) * 65536, "function_rows": len(ROWS),
        "block_size": BLOCK_SIZE, "unique_blocks": len(blocks),
        "index_bytes": len(indexes) * 2, "value_bytes": len(flattened) * 2,
        "runtime_table_bytes": (len(indexes) + len(flattened)) * 2,
        "uncompressed_bytes": len(ROWS) * 65536 * 2,
        "row_sha256": row_digests,
        "generated_source_sha256": hashlib.sha256(contents.encode()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if verify:
        if destination.read_text() != contents:
            raise AssertionError("Generated table differs from the checked-in table")
        print(json.dumps(report, indent=2), flush=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents)
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] /
                        "c/src/generated/half_math_tables.inc")
    parser.add_argument("--precision", type=int, default=160)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.precision, args.verify)


if __name__ == "__main__":
    main()
