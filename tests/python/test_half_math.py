"""Check the half math data, its integer transport, and rounding boundaries."""

import ctypes
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess

import mpmath as mp
import pytest


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/generate_half_math.py"
TABLE = ROOT / "c/src/generated/half_math_tables.inc"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("half_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("clang")
    if compiler is None:
        pytest.skip("Clang is required for the strict-half test fixture")
    macros = subprocess.run([compiler, "-dM", "-E", "-x", "c", "-"],
                            input="", capture_output=True, text=True, check=True).stdout
    if ("__ARM_FEATURE_FP16_SCALAR_ARITHMETIC" not in macros and
            "__AVX512FP16__" not in macros):
        pytest.skip("The host compiler target lacks native binary16 arithmetic")
    output = tmp_path_factory.mktemp("half-math") / "half_math.dylib"
    command = [compiler, "-std=c99", "-shared", "-fPIC", "-O2", "-Wall", "-Wextra",
               "-Wpedantic", "-Werror", "-fno-fast-math", "-ffp-contract=off",
               "-ffp-eval-method=source", "-I", str(ROOT / "c/include"),
               str(ROOT / "c/src/half_math.c"),
               str(ROOT / "tests/c/half_math_bridge.c"), "-lm", "-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(output))
    library.lc_half_test_row.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint16)]
    library.lc_half_test_row.restype = None
    library.lc_half_math_table_bytes.restype = ctypes.c_uint32
    library.lc_half_test_environment.argtypes = [ctypes.c_uint32]
    library.lc_half_test_environment.restype = ctypes.c_int
    library.lc_half_test_flush_subnormals.restype = ctypes.c_int
    return library


def test_half_environment_requires_nearest_even(native):
    assert native.lc_half_test_environment(0) == 1
    for mode in (1, 2, 3):
        result = native.lc_half_test_environment(mode)
        if result == -1:
            pytest.skip("The host does not support all directed rounding modes")
        assert result == 0


def test_half_environment_rejects_flush_to_zero(native):
    result = native.lc_half_test_flush_subnormals()
    if result == -1:
        pytest.skip("The host test cannot set a native half flush-to-zero mode")
    assert result == 0
    assert native.lc_half_test_environment(0) == 1


def test_generated_source_and_generator_fingerprints():
    report = json.loads(TABLE.with_suffix(".json").read_text())
    assert hashlib.sha256(TABLE.read_bytes()).hexdigest() == report["generated_source_sha256"]
    assert hashlib.sha256(GENERATOR.read_bytes()).hexdigest() == report["generator_sha256"]


def test_all_half_value_and_midpoint_roundings(generator):
    with mp.workprec(100):
        for bits in range(0x7BFF):
            left, right = generator.half_value(bits), generator.half_value(bits + 1)
            assert generator.round_half(left) == bits
            midpoint = (left + right) / 2
            expected = bits if bits % 2 == 0 else bits + 1
            assert generator.round_half(midpoint) == expected
            assert generator.round_half(-midpoint) == (expected | 0x8000)
            packed = struct.unpack("<H", struct.pack("<e", float(midpoint)))[0]
            assert packed == expected
        assert generator.round_half(mp.mpf(65520)) == 0x7C00
        assert generator.round_half(mp.mpf(-65520)) == 0xFC00


def test_every_runtime_lookup_matches_interval_checked_row(native, generator):
    report = json.loads(TABLE.with_suffix(".json").read_text())
    output = (ctypes.c_uint16 * 65536)()
    for index, (name, _) in enumerate(generator.ROWS):
        native.lc_half_test_row(index, output)
        packed = b"".join(value.to_bytes(2, "little") for value in output)
        assert hashlib.sha256(packed).hexdigest() == report["row_sha256"][name], name
    assert native.lc_half_math_table_bytes() == report["runtime_table_bytes"]


@pytest.mark.parametrize("row,operation", [(21, math.floor), (22, math.ceil), (23, abs)])
def test_every_integer_helper_matches_independent_host_reference(native, row, operation):
    output = (ctypes.c_uint16 * 65536)()
    native.lc_half_test_row(row, output)
    for bits in range(65536):
        magnitude = bits & 0x7FFF
        if magnitude >= 0x7C00:
            continue
        value = struct.unpack("<e", bits.to_bytes(2, "little"))[0]
        expected = operation(value)
        if expected == 0 and row != 23:
            expected = math.copysign(0.0, value)
        packed = struct.unpack("<H", struct.pack("<e", expected))[0]
        assert output[bits] == packed, (row, hex(bits))


def test_independent_decimal_transcendental_samples(native, generator):
    from decimal import Decimal, localcontext

    output = (ctypes.c_uint16 * 65536)()
    samples = tuple(range(1, 0x7C00, 31)) + (1, 2, 0x3BFF, 0x3C00, 0x3C01, 0x7BFF)
    with localcontext() as decimal_context, mp.workprec(400):
        decimal_context.prec = 90
        for row, operation in ((0, "exp"), (2, "ln"), (4, "sqrt")):
            native.lc_half_test_row(row, output)
            for bits in samples:
                value = struct.unpack("<e", bits.to_bytes(2, "little"))[0]
                if row == 0 and value > 12:
                    continue
                expected = getattr(Decimal.from_float(value), operation)()
                assert output[bits] == generator.round_half(mp.mpf(str(expected)))
