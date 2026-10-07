"""Audit binary16 runtime IR, final assembly, and linked external dependencies.

The host compiler and host-only FFI bridges are excluded. Results apply only to
the recorded compiler target, settings, source files, and generated tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("lacuna", "network", "expr", "codec", "compiled_graph_io", "half_math")
WIDE_TYPE = re.compile(r"\b(?:float|double|x86_fp80|fp128|ppc_fp128)\b")
FP_OPERATION = re.compile(r"\b(?:fadd|fsub|fmul|fdiv|frem|fneg|fcmp|fpext|fptrunc|"
                          r"fptoui|fptosi|uitofp|sitofp|call|invoke|callbr)\b")
ARM_FP = re.compile(r"^(?:f(?:add|sub|mul|div|sqrt|max|min|nmul|madd|msub|nmadd|nmsub|"
                    r"cmp|cmpe|csel|cvt|recp|rsqrt|rint|abd|mla|mls)|[su]cvtf)")
ARM_WIDE_REGISTER = re.compile(r"(?:\b[sd]\d+\b|\bv\d+\.(?:[1248][sd]|[sd])\b)")
X86_FP = re.compile(r"^v?(?:add|sub|mul|div|max|min|sqrt|rcp|rsqrt|cvt|cmp|comi|ucomi|"
                    r"fmadd|fmsub|fnmadd|fnmsub)")
ALLOWED_SYMBOLS = frozenset({
    "malloc", "calloc", "realloc", "free", "memcpy", "memset", "memmove", "memcmp",
    "qsort", "bsearch", "bzero", "stack_chk_fail", "stack_chk_guard", "chkstk_darwin",
    "memcpy_chk", "memmove_chk", "memset_chk", "cxa_finalize", "gmon_start",
    "ITM_deregisterTMCloneTable", "ITM_registerTMCloneTable",
})


def forbidden_ir(ir: str):
    findings = []
    for number, original in enumerate(ir.splitlines(), 1):
        line = original.split(";", 1)[0].strip()
        if not line or line.startswith(("@", "!", "define", "declare")):
            continue
        if FP_OPERATION.search(line) and WIDE_TYPE.search(line):
            findings.append({"line": number, "instruction": original.strip()})
    return findings


def forbidden_assembly(assembly: str, target: str):
    arm = target.startswith(("arm64", "aarch64"))
    x86 = target.startswith(("x86_64", "amd64"))
    if not arm and not x86:
        raise ValueError(f"No final-assembly checker is implemented for {target}")
    findings = []
    for number, original in enumerate(assembly.splitlines(), 1):
        line = original.split(";", 1)[0].strip()
        match = re.match(r"([a-z][a-z0-9.]*)\s+(.+)", line)
        if match is None:
            continue
        opcode, operands = match.groups()
        forbidden = (arm and ARM_FP.match(opcode) and ARM_WIDE_REGISTER.search(operands))
        if x86 and X86_FP.match(opcode):
            forbidden = any(ending in opcode for ending in ("ss", "sd", "ps", "pd"))
        if forbidden:
            findings.append({"line": number, "instruction": original.strip()})
    return findings


def forbidden_symbols(symbols: str):
    findings = []
    for line in symbols.splitlines():
        fields = line.split()
        if not fields or line.endswith(":"):
            continue
        symbol = fields[-1].split("@", 1)[0].lstrip("_")
        if symbol not in ALLOWED_SYMBOLS:
            findings.append(symbol)
    return findings


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, output: Path, compiler="clang"):
    root = root.resolve()
    executable = shutil.which(compiler)
    if executable is None:
        raise ValueError(f"Compiler not found: {compiler}")
    version = subprocess.check_output([executable, "--version"], text=True)
    if "clang" not in version.lower():
        raise ValueError("This LLVM IR audit requires Clang")
    target = subprocess.check_output([executable, "-dumpmachine"], text=True).strip()
    forbidden_assembly("", target)
    output.mkdir(parents=True, exist_ok=False)
    flags = ["-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-Wdouble-promotion",
             "-fno-fast-math", "-ffp-contract=off", "-ffp-eval-method=source",
             "-DLACUNA_REAL_BITS=16", "-DLACUNA_TIME_BITS=16", "-DLACUNA_ENABLE_PROFILING=0",
             "-I", str(root / "c/include")]
    records = []
    for optimization in ("-O0", "-O3"):
        for source in SOURCES:
            source_path = root / f"c/src/{source}.c"
            for kind in ("ir", "assembly"):
                artifact = output / f"{source}-{optimization[1:]}.{('ll' if kind == 'ir' else 's')}"
                command = [executable, *flags, optimization, "-S"]
                if kind == "ir":
                    command.append("-emit-llvm")
                command.extend([str(source_path), "-o", str(artifact)])
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                log = artifact.with_suffix(artifact.suffix + ".log")
                log.write_text(result.stdout + result.stderr)
                contents = artifact.read_text() if result.returncode == 0 else ""
                findings = (forbidden_ir(contents) if kind == "ir" else
                            forbidden_assembly(contents, target))
                records.append({"source": str(source_path.relative_to(root)),
                                "source_sha256": sha256(source_path),
                                "optimization": optimization, "kind": kind,
                                "command": command, "returncode": result.returncode,
                                "artifact": artifact.name,
                                "artifact_sha256": sha256(artifact) if artifact.exists() else None,
                                "diagnostics": log.name, "forbidden_operations": findings})
    library = output / "liblacuna_strict16.dylib"
    command = [executable, *flags, "-O3", "-fPIC", "-shared",
               *(str(root / f"c/src/{source}.c") for source in SOURCES), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (output / "link.log").write_text(result.stdout + result.stderr)
    symbols = ""
    if result.returncode == 0:
        symbols = subprocess.check_output(["nm", "-u", str(library)], text=True)
    (output / "undefined-symbols.txt").write_text(symbols)
    dependencies = forbidden_symbols(symbols)
    report = {
        "schema_version": 1,
        "passed": result.returncode == 0 and not dependencies and all(
            item["returncode"] == 0 and not item["forbidden_operations"] for item in records),
        "compiler": executable, "compiler_version": version, "target": target,
        "profile": "float16", "profiling_enabled": False,
        "source_root": str(root), "records": records,
        "link_command": command, "link_returncode": result.returncode,
        "undefined_symbols": symbols.splitlines(), "unexpected_symbols": dependencies,
        "headers": {str(path.relative_to(root)): sha256(path)
                    for path in sorted((root / "c/include").glob("*.h"))},
        "tables": {str(path.relative_to(root)): sha256(path)
                   for path in sorted((root / "c/src/generated").glob("half_math_tables.*"))},
        "scope": "All six runtime translation units at O0 and O3, LLVM IR and final compiler "
                 "assembly, plus a linked O3 library with an integer/memory-only external-symbol "
                 "allowlist. No claim is made about untested targets or compilers."
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="clang")
    args = parser.parse_args()
    try:
        result = audit(args.root, args.output, args.compiler)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Strict float16 audit failed: {error}", file=sys.stderr)
        return 2
    print(args.output / "report.json")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
