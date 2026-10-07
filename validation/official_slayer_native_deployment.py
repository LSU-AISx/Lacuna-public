"""Compare trained JSON deployments against a standalone C image loader."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<8sIIIdQQI")
EVENT = struct.Struct("<dIIId")
SPIKE = struct.Struct("<dI")
OUT_HEADER = struct.Struct("<8sIII")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def compile_runner(directory, library):
    """Build a native loader with no frontend or training dependencies."""
    compiler = shutil.which("cc")
    if compiler is None:
        raise RuntimeError("a C compiler is required to build the replay utility")
    output = Path(directory) / "compiled_image_replay"
    source = ROOT / "validation/compiled_image_replay.c"
    command = [
        compiler,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT / "c/include"),
        str(source),
        str(library),
        "-Wl,-rpath," + str(Path(library).parent),
        "-lm",
        "-o",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    write_json(
        Path(directory) / "native_build.json",
        {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "source_sha256": sha256(source),
        },
    )
    if result.returncode:
        raise RuntimeError("native replay utility compilation failed")
    inspector = "otool" if platform.system() == "Darwin" else "ldd"
    dependency_command = (
        [inspector, "-L", str(output)]
        if inspector == "otool"
        else [inspector, str(output)]
    )
    dependencies = subprocess.run(
        dependency_command, capture_output=True, text=True, check=True
    )
    write_json(
        Path(directory) / "native_dependencies.json",
        {
            "command": dependency_command,
            "stdout": dependencies.stdout,
            "stderr": dependencies.stderr,
        },
    )
    return output


def input_header(stream, *, episodes, nodes, initial, duration, options):
    stream.write(
        HEADER.pack(
            b"LCRPLY01",
            episodes,
            nodes,
            len(initial),
            duration,
            options.queue_capacity,
            options.output_capacity,
            options.same_time_cascade_limit,
        )
    )
    stream.write(struct.pack(f"<{len(initial)}d", *initial))


def input_episode(stream, events):
    stream.write(struct.pack("<I", len(events)))
    for t, node in events:
        stream.write(EVENT.pack(t, node, 0, 0, 1.0))


def output_header(stream, *, episodes, nodes, states):
    stream.write(OUT_HEADER.pack(b"LCOUT001", episodes, nodes, states))


def output_episode(stream, result):
    """Retain scheduler order and raw binary64 state, without binning."""
    spikes = result.raw.core.spikes
    stream.write(struct.pack("<Q", len(spikes)))
    for spike in spikes:
        stream.write(SPIKE.pack(spike.t, spike.node))
    values = [value for state in result.final_states for value in state.values]
    last = [state.t for state in result.final_states]
    stream.write(struct.pack(f"<{len(values)}d", *values))
    stream.write(struct.pack(f"<{len(last)}d", *last))


def _read(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated native output")
    return value


def compare_outputs(expected, actual):
    """Require bitwise agreement for every spike, state, and update time."""
    rows = []
    with Path(expected).open("rb") as left, Path(actual).open("rb") as right:
        header = _read(left, OUT_HEADER.size)
        if header != _read(right, OUT_HEADER.size):
            raise ValueError("native output layout differs from reference")
        magic, episodes, nodes, states = OUT_HEADER.unpack(header)
        if magic != b"LCOUT001":
            raise ValueError("invalid output magic")
        for sample in range(episodes):
            expected_count = struct.unpack("<Q", _read(left, 8))[0]
            actual_count = struct.unpack("<Q", _read(right, 8))[0]
            expected_spikes = _read(left, expected_count * SPIKE.size)
            actual_spikes = _read(right, actual_count * SPIKE.size)
            states_equal = _read(left, states * 8) == _read(right, states * 8)
            times_equal = _read(left, nodes * 8) == _read(right, nodes * 8)
            rows.append(
                {
                    "sample": sample,
                    "spikes": actual_count,
                    "spikes_bitwise_equal": expected_spikes == actual_spikes,
                    "final_states_bitwise_equal": states_equal,
                    "last_update_times_bitwise_equal": times_equal,
                }
            )
        if left.read(1) or right.read(1):
            raise ValueError("unexpected trailing native output")
    return rows


def native_replay(executable, image, events, output, log):
    started = time.perf_counter()
    # The executed child is C. No Python environment is needed for replay.
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    result = subprocess.run(
        [str(executable), str(image), str(events), str(output)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    evidence = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "seconds": time.perf_counter() - started,
        "environment": environment,
    }
    write_json(log, evidence)
    if result.returncode:
        raise RuntimeError(f"native replay failed, see {log}")
    return evidence


def verify_artifact(directory, library):
    report = json.loads((directory / "report.json").read_text())
    verified = {}
    for name, expected in report["files_sha256"].items():
        if Path(name).name != name:
            raise ValueError("artifact manifest contains a nonlocal filename")
        actual = sha256(directory / name)
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {directory / name}")
        verified[name] = actual
    if sha256(library) != report["lacuna_library_sha256"]:
        raise ValueError("runtime library differs from the trained pilot report")
    return report, verified


def validate_artifact(directory, output, executable, library):
    import numpy as np
    import torch
    from lacuna import Engine, Network, RunOptions, SpikeTrain

    started = time.perf_counter()
    report, hashes = verify_artifact(directory, library)
    output.mkdir()
    network = Network.load(directory / "network.json")
    data = torch.load(directory / "validation_inputs.pt", weights_only=True)
    dt = report["import"]["timestep"]
    input_nodes = network.populations["input"].node_ids
    ports_by_node = {port.node: port.id for port in network.graph.input_ports}
    ports = [ports_by_node[node] for node in input_nodes]
    layer_nodes = [
        network.populations[f"layer_{index}"].node_ids
        for index in range(len(report["import"]["layers"]))
    ]
    node_layer = {
        node: (index, channel)
        for index, nodes in enumerate(layer_nodes)
        for channel, node in enumerate(nodes)
    }
    node_count = len(network.graph.nodes)
    bins = data["validation_inputs"].shape[-1]
    total_samples = sum(
        len(data[f"{split}_inputs"]) for split in ("validation", "test")
    )
    options = RunOptions(
        queue_capacity=max(4096, len(network.graph.edges) + node_count + 1),
        output_capacity=bins * node_count + 1,
        encoder_spike_capacity=max(4096, bins * len(ports) + 1),
    )
    initial = []
    for node in sorted(network.graph.nodes, key=lambda item: item.id):
        initial.extend(
            node.initial if isinstance(node.initial, tuple) else (node.initial,)
        )
    if len(initial) != node_count or any(initial):
        raise ValueError(
            "this trained-artifact protocol requires scalar zero-state LIF"
        )
    input_path = output / "episodes.bin"
    expected_path = output / "json_reference.bin"
    native_path = output / "native_loaded.bin"
    split_rows = []
    engine = Engine(library)
    with (
        engine.compile(network) as simulation,
        input_path.open("wb") as event_file,
        expected_path.open("wb") as expected_file,
    ):
        node_index = {
            node: index
            for index, node in enumerate(simulation.compiled.resolved.node_ids)
        }
        input_header(
            event_file,
            episodes=total_samples,
            nodes=node_count,
            initial=initial,
            duration=(bins - 1) * dt,
            options=options,
        )
        output_header(
            expected_file, episodes=total_samples, nodes=node_count, states=len(initial)
        )
        for split in ("validation", "test"):
            tensors = data[f"{split}_inputs"].reshape(-1, len(ports), bins)
            comparison = json.loads(
                (directory / f"{split}_comparison.json").read_text()
            )
            if len(tensors) != comparison["samples"]:
                raise ValueError("saved sample count differs from comparison")
            totals = [0] * len(layer_nodes)
            for sample, tensor in enumerate(tensors):
                if not bool(torch.all((tensor == 0) | (tensor == 1))):
                    raise ValueError("saved inputs are not binary")
                matrix = tensor.numpy()
                stimulus = {
                    port: SpikeTrain(
                        times=tuple(np.flatnonzero(matrix[channel]) * dt), values=1.0
                    )
                    for channel, port in enumerate(ports)
                }
                times, channels = np.nonzero(matrix.T)
                events = [
                    (float(t * dt), node_index[input_nodes[int(channel)]])
                    for t, channel in zip(times, channels)
                ]
                input_episode(event_file, events)
                result = simulation.run(
                    (bins - 1) * dt, inputs=stimulus, options=options
                )
                output_episode(expected_file, result)
                counts = [0] * len(layer_nodes[-1])
                for spike in result.spikes:
                    if spike.node in node_layer:
                        layer, channel = node_layer[spike.node]
                        totals[layer] += 1
                        if layer == len(layer_nodes) - 1:
                            counts[channel] += 1
                if counts != comparison["lacuna_output_counts"][sample]:
                    raise ValueError(
                        f"original counts not reproduced: {split} {sample}"
                    )
                prediction = max(range(len(counts)), key=counts.__getitem__)
                if prediction != comparison["lacuna_predictions"][sample]:
                    raise ValueError(
                        f"original prediction not reproduced: {split} {sample}"
                    )
                if (sample + 1) % 250 == 0:
                    print(
                        f"{directory.parent.name} {split}: {sample + 1}/{len(tensors)} JSON replay",
                        flush=True,
                    )
            if totals != [layer["lacuna_spikes"] for layer in comparison["layers"]]:
                raise ValueError(f"original layer totals not reproduced: {split}")
            split_rows.append(
                {
                    "split": split,
                    "samples": len(tensors),
                    "layer_spikes": totals,
                    "original_counts_and_predictions_match": True,
                }
            )
    source_seconds = time.perf_counter() - started
    native = native_replay(
        executable,
        directory / "network.lcbin",
        input_path,
        native_path,
        output / "native_process.json",
    )
    rows = compare_outputs(expected_path, native_path)
    failures = [
        row
        for row in rows
        if not all(
            row[key]
            for key in (
                "spikes_bitwise_equal",
                "final_states_bitwise_equal",
                "last_update_times_bitwise_equal",
            )
        )
    ]
    evidence = {
        "artifact": str(directory),
        "artifact_report_sha256": sha256(directory / "report.json"),
        "artifact_files_verified": hashes,
        "network_semantic_sha256": network.semantic_sha256,
        "samples": total_samples,
        "nodes": node_count,
        "states": len(initial),
        "splits": split_rows,
        "all_spikes_including_relays": sum(row["spikes"] for row in rows),
        "all_spikes_and_final_states_bitwise_equal": not failures,
        "first_failures": failures[:20],
        "failed_samples": len(failures),
        "json_replay_and_preparation_seconds": source_seconds,
        "native_process_seconds": native["seconds"],
        "precision_claim": "same runtime and host, not cross-platform or SLAYER bitwise equivalence",
        "output_sha256": {
            path.name: sha256(path) for path in (input_path, expected_path, native_path)
        },
    }
    write_json(output / "sample_comparison.json", rows)
    write_json(output / "report.json", evidence)
    if failures:
        raise RuntimeError(f"native deployment mismatch, see {output / 'report.json'}")
    verify_artifact(directory, library)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--library", type=Path, default=ROOT / "build/liblacuna_core.dylib"
    )
    args = parser.parse_args()
    output = (
        args.output
        or ROOT
        / "artifacts/official-slayer-native-deployment"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    library = args.library.resolve()
    write_json(
        output / "protocol.json",
        {
            "artifacts": [str(path.resolve()) for path in args.artifact],
            "library": str(library),
            "library_sha256": sha256(library),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "independence": "child executes only a C binary linked to Lacuna and system libraries",
            "comparison": "all ordered spikes, raw final states, and last update times, bitwise",
            "episodes": "zero-state independent episodes from saved validation and test tensors",
            "script_sha256": sha256(__file__),
        },
    )
    try:
        executable = compile_runner(output, library)
        evidence = [
            validate_artifact(path.resolve(), output / str(index), executable, library)
            for index, path in enumerate(args.artifact)
        ]
        write_json(
            output / "report.json",
            {
                "status": "complete",
                "results": evidence,
                "native_executable_sha256": sha256(executable),
            },
        )
    except Exception as exc:
        write_json(
            output / "failure.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    print(json.dumps({"output": str(output), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
