"""Compile every runtime source and reject wider arithmetic in Clang's IR.

The audit covers this source snapshot and compiler on the reported target.
It is not a claim about untested MCU toolchains or the internals of libm.
The separate host constant binder is intentionally excluded from the runtime.
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
CORE_SOURCES = (
    "c/src/lacuna.c", "c/src/network.c", "c/src/expr.c",
    "c/src/codec.c", "c/src/compiled_graph_io.c",
)
_WIDE_TYPE = re.compile(r"\b(?:double|x86_fp80|fp128|ppc_fp128)\b")
_FP_OPERATION = re.compile(
    r"\b(?:fadd|fsub|fmul|fdiv|frem|fneg|fcmp|fpext|fptrunc|"
    r"fptoui|fptosi|uitofp|sitofp|call|invoke|callbr)\b"
)
_FP32_OPERATION = re.compile(
    r"\b(?:fadd|fsub|fmul|fdiv|frem|fneg|fcmp)\b.*\bfloat\b"
)


def forbidden_operations(ir: str) -> list[dict[str, object]]:
    """Find wide scalar or vector arithmetic, conversions, and calls."""
    findings = []
    for number, original in enumerate(ir.splitlines(), start=1):
        line = original.split(";", 1)[0].strip()
        if not line or line.startswith(("!", "@", "declare", "define")):
            continue
        if _FP_OPERATION.search(line) and _WIDE_TYPE.search(line):
            findings.append({"line": number, "instruction": original.strip()})
    return findings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path, output: Path, compiler: str = "clang") -> dict[str, object]:
    """Retain source identities, compiler commands, IR, and diagnostics."""
    root = root.resolve()
    executable = shutil.which(compiler)
    if executable is None:
        raise ValueError(f"Clang compiler was not found: {compiler}")
    version = subprocess.check_output([executable, "--version"], text=True)
    if "clang" not in version.lower():
        raise ValueError("This audit requires Clang's LLVM IR output")
    output.mkdir(parents=True, exist_ok=False)
    flags = [
        "-std=c99", "-Wall", "-Wextra", "-Wpedantic", "-Werror",
        "-Wdouble-promotion", "-fno-fast-math", "-ffp-contract=off",
        "-DLACUNA_REAL_BITS=32", "-DLACUNA_TIME_BITS=32",
        "-DLACUNA_ENABLE_PROFILING=0", "-I", str(root / "c/include"),
        "-S", "-emit-llvm",
    ]
    records = []
    for optimization in ("-O0", "-O3"):
        for source in CORE_SOURCES:
            path = root / source
            stem = f"{path.stem}-{optimization[1:]}"
            ir_path = output / f"{stem}.ll"
            log_path = output / f"{stem}.log"
            command = [executable, *flags, optimization, str(path), "-o", str(ir_path)]
            result = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, check=False, timeout=120,
            )
            log_path.write_text(result.stdout)
            ir = ir_path.read_text() if result.returncode == 0 else ""
            findings = forbidden_operations(ir)
            records.append({
                "source": source,
                "source_sha256": _sha256(path),
                "optimization": optimization,
                "command": command,
                "returncode": result.returncode,
                "diagnostics": log_path.name,
                "ir": ir_path.name if result.returncode == 0 else None,
                "ir_sha256": _sha256(ir_path) if result.returncode == 0 else None,
                "binary32_arithmetic_instructions": sum(
                    bool(_FP32_OPERATION.search(line)) for line in ir.splitlines()
                ),
                "forbidden_operations": findings,
            })
    success = all(record["returncode"] == 0 and not record["forbidden_operations"]
                  for record in records)
    report = {
        "schema_version": 1,
        "passed": success,
        "compiler": executable,
        "compiler_version": version,
        "root": str(root),
        "profile": "float32",
        "profiling_enabled": False,
        "headers": {
            str(path.relative_to(root)): _sha256(path)
            for path in sorted((root / "c/include").glob("*.h"))
        },
        "records": records,
        "scope": (
            "All five execution-runtime translation units at O0 and O3. "
            "Checks wide floating arithmetic, comparisons, conversions, and calls. "
            "Integer and bitwise binary-image transport are permitted. "
            "No claim is made about untested compilers, target hardware, "
            "or third-party libm internals."
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="clang")
    args = parser.parse_args()
    try:
        report = audit(args.root, args.output, args.compiler)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"strict-float32 audit failed: {error}", file=sys.stderr)
        return 2
    print(args.output / "report.json")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
