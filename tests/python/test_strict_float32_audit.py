"""The compiler-IR audit must detect hidden wider floating arithmetic."""

import importlib.util
from pathlib import Path
import shutil

import pytest


_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "strict_float32_audit", _ROOT / "scripts/audit_strict_float32.py"
)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


@pytest.mark.parametrize("instruction", (
    "%x = fadd double %a, %b",
    "%x = fmul <4 x double> %a, %b",
    "%x = fneg x86_fp80 %a",
    "%x = fcmp oeq fp128 %a, %b",
    "%x = fpext float %a to double",
    "%x = fptrunc double %a to float",
    "%x = uitofp i32 %a to double",
    "%x = fptosi double %a to i32",
    "%x = call double @exp(double %a)",
    "%x = call float @adapter(double %a)",
    "call void @sink(ppc_fp128 %a)",
))
def test_wide_operations_are_detected(instruction):
    found = _AUDIT.forbidden_operations(instruction)
    assert found == [{"line": 1, "instruction": instruction}]


@pytest.mark.parametrize("instruction", (
    "%x = fadd float %a, %b",
    "%x = fmul <4 x float> %a, %b",
    "%x = call float @expf(float %a)",
    "%x = load i64, ptr %p",
    "store i64 %bits, ptr %p",
    "declare double @unused(double)",
    "; %x = fadd double %a, %b",
    '@message = constant [6 x i8] c"double"',
))
def test_float32_and_bit_transport_are_allowed(instruction):
    assert _AUDIT.forbidden_operations(instruction) == []


def test_strict_runtime_ir_has_no_wide_operations(tmp_path):
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("Clang is unavailable for the compiler-IR audit")
    report = _AUDIT.audit(_ROOT, tmp_path / "audit", clang)
    assert report["passed"], report["records"]
    assert len(report["records"]) == 10
    assert all(record["ir_sha256"] for record in report["records"])
    assert sum(record["binary32_arithmetic_instructions"]
               for record in report["records"]) > 0
