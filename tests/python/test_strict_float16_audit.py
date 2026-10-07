import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/audit_strict_float16.py"
SPEC = importlib.util.spec_from_file_location("strict_half_audit", SOURCE)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


@pytest.mark.parametrize("instruction", [
    "%1 = fadd float %a, %b", "%1 = fmul double %a, %b",
    "%1 = fpext half %a to float", "%1 = fptrunc float %a to half",
    "%1 = call half @hidden(float %a)", "%1 = fcmp olt double %a, %b",
    "%1 = fadd <4 x float> %a, %b",
])
def test_rejects_wider_ir(instruction):
    assert AUDIT.forbidden_ir(instruction)


@pytest.mark.parametrize("instruction", [
    "%1 = fadd half %a, %b", "%1 = load double, ptr %wire",
    "%1 = call half @lookup(half %a)", "%1 = add i64 %a, %b",
])
def test_allows_half_arithmetic_and_integer_wire_operations(instruction):
    assert not AUDIT.forbidden_ir(instruction)


@pytest.mark.parametrize("instruction", [
    "fadd s0, s1, s2", "fadd d0, d1, d2", "fcvt s0, h0",
    "fcvt h0, s0", "fcmp d0, d1", "fmla v0.4s, v1.4s, v2.4s",
])
def test_rejects_wider_arm_operations(instruction):
    assert AUDIT.forbidden_assembly(instruction, "arm64-apple-darwin")


@pytest.mark.parametrize("instruction", [
    "fadd h0, h1, h2", "fcmp h0, h1", "scvtf h0, w0", "fmov s0, w0",
])
def test_allows_half_arm_operations_and_bit_moves(instruction):
    assert not AUDIT.forbidden_assembly(instruction, "arm64-apple-darwin")


def test_externals_are_integer_or_memory_operations_only():
    assert not AUDIT.forbidden_symbols("_malloc\n_memcpy\n___stack_chk_fail\n")
    assert AUDIT.forbidden_symbols("_expf\n___extendhfsf2\n") == ["expf", "extendhfsf2"]


def test_final_backend_widening_cannot_hide_behind_half_ir():
    assert not AUDIT.forbidden_ir("%1 = call half @llvm.exp.f16(half %a)")
    assert AUDIT.forbidden_assembly("fcvt s0, h0\nbl _expf\nfcvt h0, s0", "arm64")
    assert AUDIT.forbidden_symbols("_expf")
