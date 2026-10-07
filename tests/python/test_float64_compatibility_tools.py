"""Fast unit checks for the independent old/new regression harness."""

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path
import struct

import pytest


_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/validate_float64_compatibility.py"
)
_SPEC = importlib.util.spec_from_file_location("float64_compatibility_tools", _SCRIPT)
_TOOLS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_TOOLS)


def test_exact_float_serialization_retains_signed_zero_and_nan_payloads():
    assert _TOOLS.exact(0.0) != _TOOLS.exact(-0.0)
    for bits in ("7ff8000000000001", "7ff8000000000012", "fff8000000000001"):
        value = struct.unpack(">d", bytes.fromhex(bits))[0]
        assert _TOOLS.exact(value) == {"binary64": bits}


def test_only_explicit_walltime_is_excluded_not_other_noncomparing_fields():
    @dataclass
    class Result:
        value: float
        retained: float = field(compare=False)
        kernel_seconds: float = field(compare=False)

    a = _TOOLS.exact(Result(1.0, 2.0, 3.0))
    assert a == _TOOLS.exact(Result(1.0, 2.0, 99.0))
    assert a != _TOOLS.exact(Result(1.0, 3.0, 3.0))
    assert "kernel_seconds" not in a["fields"]


def test_first_difference_reports_nested_float_bit_drift():
    a = _TOOLS.exact({"states": [1.0, 0.0]})
    b = _TOOLS.exact({"states": [1.0, -0.0]})
    assert "$.states[1].binary64" in _TOOLS.first_difference(a, b)
    assert _TOOLS.first_difference(a, a) is None
    with pytest.raises(TypeError, match="Unsupported fixture"):
        _TOOLS.exact(object())


def test_public_structure_parser_preserves_pointers_arrays_and_order():
    assert _TOOLS.public_structures("""
        typedef struct opaque opaque;
        typedef struct lc_example {
            const double *values; /* retained pointer */
            unsigned indices[COUNT + 1U];
            callback_type callback;
        } lc_example;
    """) == {"lc_example": ["values", "indices", "callback"]}
    source = _TOOLS.probe_source({"lc_example": ["values"]})
    assert "_Alignof(lc_example)" in source
    assert "offsetof(lc_example, values)" in source
    assert "sizeof(((lc_example *)0)->values)" in source


@pytest.mark.parametrize(
    "body",
    ("int a, b;", "int a:3;", "void (*f)(void);", "struct { int x; } inner;"),
)
def test_public_structure_parser_fails_closed_on_unhandled_syntax(body):
    with pytest.raises(ValueError, match="Unrecognized"):
        _TOOLS.public_structures("typedef struct lc_example {" + body + "} lc_example;")


def test_abi_probe_uses_build_sdk_and_compiler_flags():
    assert _TOOLS.probe_flags(
        {
            "CMAKE_C_FLAGS": "-fno-fast-math",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_C_FLAGS_RELEASE": "-O3 -DNDEBUG",
            "CMAKE_OSX_SYSROOT": "/SDK Path",
            "CMAKE_OSX_DEPLOYMENT_TARGET": "14.0",
            "CMAKE_OSX_ARCHITECTURES": "arm64",
        }
    ) == [
        "-fno-fast-math", "-O3", "-DNDEBUG", "-isysroot", "/SDK Path",
        "-mmacosx-version-min=14.0", "-arch", "arm64",
    ]
