#!/usr/bin/env python3
"""Compare a baseline and candidate build without changing their sources.

Example (both roots must already have a same-toolchain Release build)::

    python scripts/validate_float64_compatibility.py \
        --baseline-root /tmp/lacuna-before --candidate-root . \
        --output /tmp/lacuna-float64-report

Every child imports the *baseline* Python package and baseline matrix fixtures.
Only the library path changes, so this also tests old ctypes-binding compatibility.
The complete results are retained as JSON with each binary64 value represented
by its exact 16-digit bit pattern (including signed zero and NaN payloads).
Only the explicitly named kernel_seconds wall-clock field is excluded. JSON
trees and image bytes are compared directly. Hashes are provenance, not the
comparison. Split runs are compared old versus new, not asserted bit-identical
to one-shot runs, because observation frontiers can change rounding.

Coverage: all existing execution-plan family fixtures, legacy/plan/image paths,
buffered traces with state snapshots, inspections, reset replay, incremental
open boundaries, persistent-weight episode reset, and bidirectional image load.
This is a regression gate for the supplied fixtures, not a claim of equivalence
for every possible network, compiler, architecture, or future precision mode.
Requires the project's Python/test dependencies and a C11 compiler.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import sys


def exact(value):
    """Project complete observable records into a lossless, stable JSON tree."""
    if isinstance(value, float):
        return {"binary64": struct.pack(">d", value).hex()}
    if isinstance(value, enum.Enum):
        return {"enum": type(value).__name__, "value": exact(value.value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "record": type(value).__name__,
            "fields": {
                field.name: exact(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if field.name != "kernel_seconds"
            },
        }
    if isinstance(value, (tuple, list)):
        return [exact(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Fixture dictionaries must use string keys")
        return {key: exact(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported fixture value: {type(value)!r}")


def first_difference(left, right, path="$"):
    """Return the first mismatch without treating NaN or signed zero loosely."""
    if type(left) is not type(right):
        return f"{path}: types differ ({type(left).__name__}, {type(right).__name__})"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ ({sorted(left)}, {sorted(right)})"
        for key in left:
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths differ ({len(left)}, {len(right)})"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def require_equal(left, right, label):
    difference = first_difference(left, right)
    if difference:
        raise AssertionError(f"{label}: {difference}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def public_structures(header):
    """Read flat public struct declarations, failing closed on unfamiliar C."""
    clean = re.sub(r"/\*.*?\*/|//[^\n]*", "", header, flags=re.S)
    definitions = re.findall(
        r"typedef\s+struct\s+(\w+)\s*\{([^{}]*)\}\s*(\w+)\s*;", clean
    )
    starts = re.findall(r"typedef\s+struct\s+\w+\s*\{", clean)
    if len(starts) != len(definitions):
        raise ValueError("Unrecognized public struct definition (possibly nested)")
    result = {}
    for tag, body, name in definitions:
        if tag != name:
            raise ValueError(f"Unexpected struct tag/typedef: {tag}/{name}")
        members = []
        for declaration in body.split(";"):
            declaration = declaration.strip()
            if not declaration:
                continue
            member = re.search(r"\b(\w+)\s*(?:\[[^\]]+\]\s*)*$", declaration)
            if not member or any(char in declaration for char in "{},():"):
                raise ValueError(f"Unrecognized member: {name}: {declaration}")
            members.append(member.group(1))
        if not members:
            raise ValueError(f"Empty public struct: {name}")
        result[name] = members
    if not result:
        raise ValueError("No public structs found")
    return result


def probe_source(structures):
    lines = [
        "#include <stddef.h>",
        "#include <stdio.h>",
        '#include "lacuna.h"',
        "int main(void) {",
        'printf("VERSION %u %u\\n", LC_ABI_VERSION, '
        "LC_COMPILED_GRAPH_IMAGE_VERSION);",
    ]
    for name, members in structures.items():
        lines.append(
            f'printf("STRUCT {name} %zu %zu\\n", '
            f"sizeof({name}), _Alignof({name}));"
        )
        for member in members:
            lines.append(
                f'printf("FIELD {name} {member} %zu %zu\\n", '
                f'offsetof({name}, {member}), sizeof((({name} *)0)->{member}));'
            )
    return "\n".join(lines + ["return 0;", "}", ""])


def cmake_settings(root):
    cache = root / "build" / "CMakeCache.txt"
    values = {}
    for line in cache.read_text().splitlines():
        match = re.match(r"([^:#/][^:=]*):[^=]+=(.*)", line)
        if match:
            values[match.group(1)] = match.group(2)
    keys = (
        "CMAKE_C_COMPILER",
        "CMAKE_BUILD_TYPE",
        "CMAKE_C_FLAGS",
        "CMAKE_C_FLAGS_DEBUG",
        "CMAKE_C_FLAGS_RELEASE",
        "CMAKE_C_FLAGS_RELWITHDEBINFO",
        "CMAKE_C_FLAGS_MINSIZEREL",
        "CMAKE_OSX_ARCHITECTURES",
        "CMAKE_OSX_DEPLOYMENT_TARGET",
        "CMAKE_OSX_SYSROOT",
        "CMAKE_INTERPROCEDURAL_OPTIMIZATION",
    )
    settings = {key: values.get(key) for key in keys}
    # Legacy builds have the same fixed64 types without explicit cache entries.
    for key in ("LACUNA_REAL_BITS", "LACUNA_TIME_BITS"):
        settings[key] = values.get(key, "64")
    return settings


def library_in(root):
    paths = [
        root / "build" / name
        for name in ("liblacuna_core.dylib", "liblacuna_core.so", "lacuna_core.dll")
    ]
    found = [path for path in paths if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one built C library in {root / 'build'}")
    return found[0]


def probe_flags(settings):
    """Use the build's relevant compiler settings, including the macOS SDK."""
    flags = shlex.split(settings.get("CMAKE_C_FLAGS") or "")
    build_type = (settings.get("CMAKE_BUILD_TYPE") or "").upper()
    flags += shlex.split(settings.get(f"CMAKE_C_FLAGS_{build_type}") or "")
    if settings.get("CMAKE_OSX_SYSROOT"):
        flags += ["-isysroot", settings["CMAKE_OSX_SYSROOT"]]
    if settings.get("CMAKE_OSX_DEPLOYMENT_TARGET"):
        flags += ["-mmacosx-version-min=" + settings["CMAKE_OSX_DEPLOYMENT_TARGET"]]
    for arch in (settings.get("CMAKE_OSX_ARCHITECTURES") or "").split(";"):
        if arch:
            flags += ["-arch", arch]
    return flags


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_inventory(root):
    """Identify archived source trees even when Git metadata is unavailable."""
    paths = [root / "CMakeLists.txt"]
    paths.extend((root / "c").rglob("*.c"))
    paths.extend((root / "c").rglob("*.h"))
    paths.extend((root / "src/lacuna").rglob("*.py"))
    paths.append(root / "tests/python/test_execution_plan_matrix.py")
    return {
        str(path.relative_to(root)): digest(path)
        for path in sorted(paths)
    }


def worker(args):
    baseline = args.baseline_root.resolve()
    sys.path.insert(0, str(baseline / "src"))
    import lacuna
    from lacuna import CoreEvaluator, MixedDriveUpdate, ModulationEvent
    from lacuna import RecordingConfig, StateInspectionRequest

    if not Path(lacuna.__file__).resolve().is_relative_to(baseline / "src"):
        raise RuntimeError("Worker did not import the unchanged baseline package")
    matrix_path = baseline / "tests/python/test_execution_plan_matrix.py"
    spec = importlib.util.spec_from_file_location(
        "float64_baseline_matrix", matrix_path
    )
    matrix = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = matrix
    spec.loader.exec_module(matrix)
    core = CoreEvaluator(args.library)
    args.output.mkdir(parents=True, exist_ok=True)
    image_dir = args.output / "images"
    image_dir.mkdir(exist_ok=True)
    cases = {}
    coverage = {}
    for name, factory in matrix.PLAN_CASES:
        print(f"{args.worker}: {name}", flush=True)
        resolved = factory().resolve()
        plan = resolved.execution_plan()
        inputs = matrix._matrix_inputs(resolved)
        modulations = tuple(
            ModulationEvent(t, index, value)
            for index in range(len(plan.modulators))
            for t, value in ((1.5, 0.75), (3.75, -0.4), (4.5, 0.6))
        )
        drives = (
            (
                MixedDriveUpdate(1.0, 0, 21.0, "drive"),
                MixedDriveUpdate(3.0, 1, 16.0, "drive"),
            )
            if name == "scalar-parameters-polarity-codecs"
            else ()
        )
        inspections = tuple(
            StateInspectionRequest(t, node.index)
            for t in (1.25, 4.25)
            for node in plan.nodes
        )
        recording = RecordingConfig(capacity=16384)
        kwargs = dict(
            inputs=inputs,
            modulations=modulations,
            drive_updates=drives,
            inspections=inspections,
            recording=recording,
            t_end=5.0,
        )
        if args.cross_images is not None:
            with core.load_compiled_graph_image(
                (args.cross_images / f"{name}.lcg").read_bytes(), execution_plan=plan
            ) as loaded:
                cases[name] = exact(loaded.run(resolved.initial_values, **kwargs))
            continue
        _, legacy = matrix._legacy_compile(core, factory())
        with legacy, core.compile_execution_plan(plan) as compiled:
            planned = compiled.run(resolved.initial_values, **kwargs)
            legacy_result = legacy.run(resolved.initial_values, **kwargs)
            image = compiled.to_bytes()
            (image_dir / f"{name}.lcg").write_bytes(image)
            with core.load_compiled_graph_image(image, execution_plan=plan) as loaded:
                loaded_result = loaded.run(resolved.initial_values, **kwargs)
            with compiled.create_run(resolved.initial_values) as run:
                first = run.execute(**kwargs)
                run.reset(resolved.initial_values)
                replay = run.execute(**kwargs)
            require_equal(exact(first), exact(replay), f"{name}: reset replay")
            require_equal(exact(planned), exact(loaded_result), f"{name}: local image")
            require_equal(
                exact(planned), exact(legacy_result), f"{name}: legacy compiler"
            )
            split = []
            with compiled.create_incremental_run(
                resolved.initial_values, t_end=5.0
            ) as run:
                frontier = 0.0
                for until in (0.5, 1.5, 2.75, 4.0):
                    part = run.advance_until(
                        until,
                        inputs=tuple(
                            item for item in inputs if frontier <= item.t < until
                        ),
                        drive_updates=tuple(
                            item for item in drives if frontier <= item.t < until
                        ),
                        modulations=tuple(
                            item for item in modulations
                            if frontier <= item.t < until
                        ),
                        inspections=tuple(
                            item for item in inspections
                            if frontier <= item.t < until
                        ),
                        recording=recording,
                    )
                    split.append(part)
                    frontier = until
                split.append(
                    run.finish(
                        inputs=tuple(
                            item for item in inputs if item.t >= frontier
                        ),
                        drive_updates=tuple(
                            item for item in drives if item.t >= frontier
                        ),
                        modulations=tuple(
                            item for item in modulations if item.t >= frontier
                        ),
                        inspections=tuple(
                            item for item in inspections if item.t >= frontier
                        ),
                        recording=recording,
                    )
                )
                cumulative_stats = run.cumulative_stats
            observations = dict(
                planned=planned,
                legacy=legacy_result,
                image=loaded_result,
                reusable_first=first,
                reusable_reset=replay,
                split=split,
                cumulative_stats=cumulative_stats,
            )
            if plan.modulators:
                with compiled.create_incremental_run(
                    resolved.initial_values, t_end=10.0
                ) as run:
                    before_reset = run.advance_until(
                        5.0, inputs=inputs, modulations=modulations, recording=recording
                    )
                    run.reset_episode(resolved.initial_values)
                    after_reset = run.advance_until(5.0, recording=recording)
                    continuation = run.finish(
                        inputs=tuple(
                            dataclasses.replace(item, t=item.t + 5.0)
                            for item in inputs
                        ),
                        modulations=tuple(
                            dataclasses.replace(item, t=item.t + 5.0)
                            for item in modulations
                        ),
                        recording=recording,
                    )
                require_equal(
                    exact(before_reset.weights), exact(after_reset.weights),
                    f"{name}: episode reset preserves weights",
                )
                observations["episode_reset"] = (
                    before_reset, after_reset, continuation
                )
            cases[name] = exact(observations)
            coverage[name] = {
                "nodes": len(planned.states),
                "state_values": sum(len(s.values) for s in planned.states),
                "spikes": len(planned.spikes),
                "weights": len(planned.weights),
                "changed_weights": sum(
                    a != b.weight
                    for a, b in zip(planned.weights, plan.connections)
                ),
                "plasticity_records": len(planned.plasticity),
                "trace_records": len(planned.trace),
                "inspections": len(planned.inspections),
                "image_bytes": len(image),
                "split_segments": len(split),
                "episode_reset": "episode_reset" in observations,
            }
    write_json(args.output / "fixtures.json", cases)
    write_json(args.output / "coverage.json", coverage)
    write_json(
        args.output / "worker.json",
        {
            "python_package": str(Path(lacuna.__file__).resolve()),
            "python_binding_sha256": digest(baseline / "src/lacuna/ffi.py"),
            "matrix_sha256": digest(matrix_path),
            "library": str(args.library),
            "library_sha256": digest(args.library),
            "abi_version": core.abi_version,
        },
    )


def validate(args):
    baseline, candidate = args.baseline_root.resolve(), args.candidate_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory must be new or empty. "
            "Previous evidence is never overwritten."
        )
    output.mkdir(parents=True, exist_ok=True)
    old_settings, new_settings = cmake_settings(baseline), cmake_settings(candidate)
    require_equal(old_settings, new_settings, "same-toolchain CMake settings")
    compiler = old_settings["CMAKE_C_COMPILER"]
    compiler_version = subprocess.check_output([compiler, "--version"], text=True)
    old_structs = public_structures((baseline / "c/include/lacuna.h").read_text())
    new_structs = public_structures((candidate / "c/include/lacuna.h").read_text())
    for name, members in old_structs.items():
        require_equal(
            members, new_structs.get(name), f"public structure members: {name}"
        )
    # Probe the entire old ABI against both headers, with the exact same compiler.
    source = output / "abi_probe.c"
    source.write_text(probe_source(old_structs))
    abi = {}
    for label, root in (("baseline", baseline), ("candidate", candidate)):
        executable = output / f"abi_probe_{label}"
        subprocess.run(
            [compiler, *probe_flags(old_settings), "-std=c11", "-I",
             str(root / "c/include"), str(source), "-o", str(executable)],
            check=True,
        )
        abi[label] = subprocess.check_output([str(executable)], text=True)
        (output / f"abi_{label}.txt").write_text(abi[label])
    require_equal(
        abi["baseline"], abi["candidate"], "public C ABI layouts and versions"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONHASHSEED"] = "0"
    script = str(Path(__file__).resolve())

    def launch(label, root, cross=None):
        command = [
            sys.executable, "-I", script, "--worker", label,
            "--baseline-root", str(baseline),
            "--library", str(library_in(root)),
            "--output", str(output / label),
        ]
        if cross is not None:
            command += ["--cross-images", str(output / cross / "images")]
        return subprocess.Popen(command, env=env)

    # Isolated processes avoid both Python module and native-library collisions.
    processes = [launch("baseline", baseline), launch("candidate", candidate)]
    statuses = [process.wait() for process in processes]
    if any(statuses):
        raise RuntimeError(f"Fixture child failed (exit codes {statuses})")
    old = json.loads((output / "baseline/fixtures.json").read_text())
    new = json.loads((output / "candidate/fixtures.json").read_text())
    require_equal(old, new, "old/new complete bit-exact fixture records")
    images = {}
    for name in old:
        a = output / "baseline/images" / f"{name}.lcg"
        b = output / "candidate/images" / f"{name}.lcg"
        if a.read_bytes() != b.read_bytes():
            raise AssertionError(f"Compiled graph image bytes differ: {name}")
        images[name] = {"bytes": a.stat().st_size, "sha256": digest(a)}
    processes = [
        launch("baseline_loads_candidate", baseline, "candidate"),
        launch("candidate_loads_baseline", candidate, "baseline"),
    ]
    statuses = [process.wait() for process in processes]
    if any(statuses):
        raise RuntimeError(f"Cross-load child failed (exit codes {statuses})")
    for label in ("baseline_loads_candidate", "candidate_loads_baseline"):
        crossed = json.loads((output / label / "fixtures.json").read_text())
        require_equal(
            {name: values["planned"] for name, values in old.items()},
            crossed,
            f"cross-load results: {label}",
        )
    report = {
        "status": "passed",
        "baseline_root": str(baseline),
        "candidate_root": str(candidate),
        "source_inventory_sha256": {
            "baseline": source_inventory(baseline),
            "candidate": source_inventory(candidate),
        },
        "toolchain": old_settings,
        "compiler_version": compiler_version,
        "public_structures": len(old_structs),
        "public_members": sum(map(len, old_structs.values())),
        "added_public_structures": sorted(new_structs.keys() - old_structs.keys()),
        "fixtures": len(old),
        "images": images,
        "coverage": json.loads((output / "baseline/coverage.json").read_text()),
        "comparison": (
            "Full JSON trees with exact binary64 bits "
            "and direct image-byte comparison"
        ),
        "excluded_fields": ["kernel_seconds"],
        "python_bindings": "Unchanged baseline package in all four child processes",
        "limitations": [
            "Same-host/toolchain evidence, not universal numerical equivalence",
            "Graph codec metadata is image-tested. Standalone codecs are not run.",
            "No stochastic fixture in the existing family matrix",
        ],
    }
    write_json(output / "report.json", report)
    print(
        f"PASS: {len(old)} fixtures; {len(old_structs)} public structs; "
        f"{sum(map(len, old_structs.values()))} members; "
        "byte-identical images; bidirectional cross-load"
    )
    print(f"Evidence: {output / 'report.json'}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--cross-images", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.library is None:
            parser.error("Worker requires --library")
        worker(args)
    else:
        if args.candidate_root is None:
            parser.error("--candidate-root is required")
        validate(args)


if __name__ == "__main__":
    main()
