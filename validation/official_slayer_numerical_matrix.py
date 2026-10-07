"""Run a fixed numerical stress matrix against official SLAYER and Lacuna."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from lava.lib.dl import slayer

from lacuna import Engine
from lacuna.importers.slayer import (
    import_slayer_feedforward,
    validate_slayer_feedforward,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "d825ac506eb9bb91ea2b945116fb45a40e775451"
PATTERNS = ("silence", "bursts", "sparse", "dense", "cancellation", "near_threshold")


def case_grid():
    """Return the prescribed cases without consulting any observed result."""
    architectures = (
        ("dyadic_dense", [2048, 2048], 0.25, 256, [8]),
        ("slow_dense", [16, 16, 16], 0.1, 1024, [8]),
        ("deep_dense", [1, 32, 512, 2048, 3584, 4095], 1.0, 1024, [8]),
        ("grouped_conv", [4095, 512, 2048], 0.25, 256, [2, 5, 5]),
    )
    result = []
    for scale in (64, 4096):
        for architecture_index, (name, decays, timestep, bins, shape) in enumerate(
            architectures
        ):
            for pattern_index, pattern in enumerate(PATTERNS):
                exact = pattern in ("silence", "cancellation") or (
                    name == "dyadic_dense" and pattern != "near_threshold"
                )
                result.append(
                    {
                        "id": f"scale-{scale}-{name}-{pattern}",
                        "architecture": name,
                        "source_scale_argument": scale,
                        "source_state_scale": 64 * scale,
                        "source_state_quantum": 1 / (64 * scale),
                        "decay_integers": decays,
                        "timestep": timestep,
                        "bins": bins,
                        "input_shape": shape,
                        "samples": 4,
                        "source_batch_size": 4,
                        "pattern": pattern,
                        "seed": 701 + 100 * architecture_index + pattern_index,
                        "analytical_control": (
                            "silence"
                            if pattern in ("silence", "cancellation")
                            else "identity" if exact else None
                        ),
                    }
                )
    return result


def make_inputs(case):
    """Generate deterministic binary stimuli with no graded input values."""
    shape = (case["samples"], *case["input_shape"], case["bins"])
    generator = torch.Generator().manual_seed(case["seed"])
    pattern = case["pattern"]
    if pattern == "silence":
        return torch.zeros(shape)
    if pattern in ("sparse", "dense", "cancellation"):
        probability = 0.03 if pattern == "sparse" else 0.8
        inputs = (torch.rand(shape, generator=generator) < probability).float()
        if pattern == "cancellation":
            if len(case["input_shape"]) == 1:
                inputs[:, 1::2] = inputs[:, ::2]
            else:
                inputs[:] = inputs[:, :, :1, :1, :]
        return inputs
    inputs = torch.zeros(shape)
    for sample in range(case["samples"]):
        if pattern == "bursts":
            active = (torch.arange(case["bins"]) + sample) % 32 < 6
        elif pattern == "near_threshold":
            active = (torch.arange(case["bins"]) + sample) % 8 == 0
        else:
            raise ValueError(f"unknown input pattern: {pattern}")
        inputs[sample, ..., active] = 1
    return inputs


def build_model(case):
    """Construct only supported official CPU blocks with fixed signed weights."""

    def parameters(index):
        return {
            "threshold": 1.0,
            "current_decay": 1.0,
            "voltage_decay": case["decay_integers"][index] / 4096,
            "scale": case["source_scale_argument"],
            "persistent_state": False,
            "requires_grad": False,
        }

    options = {
        "pre_hook_fx": None,
        "weight_norm": False,
        "delay": False,
        "delay_shift": False,
    }
    if case["architecture"] == "grouped_conv":
        blocks = [
            slayer.block.cuba.Conv(parameters(0), 2, 2, 3, groups=2, **options),
            slayer.block.cuba.Conv(parameters(1), 2, 2, 2, groups=2, **options),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(parameters(2), 8, 8, **options),
        ]
    else:
        blocks = [
            slayer.block.cuba.Dense(parameters(i), 8, 8, **options)
            for i in range(len(case["decay_integers"]))
        ]
    model = torch.nn.Sequential(*blocks).eval()
    generator = torch.Generator().manual_seed(case["seed"] + 10_000)
    neural = [block for block in model if hasattr(block, "neuron")]
    with torch.no_grad():
        for index, block in enumerate(neural):
            weight = block.synapse.weight
            weight.copy_(torch.rand(weight.shape, generator=generator) * 1.1 - 0.45)
            if case["architecture"] == "dyadic_dense" or (
                case["pattern"] == "near_threshold"
            ):
                weight.zero_()
                if weight.shape[2:4] == (1, 1):
                    for channel in range(weight.shape[0]):
                        weight[channel, channel, 0, 0, 0] = 1.25
                else:
                    weight[:, 0, 0, 0, 0] = 1.25
                if index == 0 and case["pattern"] == "near_threshold":
                    weight[weight != 0] = 1 + case["source_state_quantum"] / 4
        if case["pattern"] == "cancellation":
            weight = neural[0].synapse.weight
            weight.zero_()
            if case["architecture"] == "grouped_conv":
                weight[:, 0, 0, 0, 0] = 1.25
                weight[:, 0, 0, 1, 0] = -1.25
            else:
                weight[:, ::2, 0, 0, 0] = 1.25
                weight[:, 1::2, 0, 0, 0] = -1.25
    return model


def check_analytical_control(case, inputs, comparison):
    """Gate exact controls, not the expected rounding-sensitive stress cases."""
    control = case["analytical_control"]
    if control is None:
        return None
    if not comparison["exact_spike_match_on_batch"]:
        raise AssertionError(f"analytical {control} control has differing spikes")
    expected = (
        torch.zeros(case["samples"], 8) if control == "silence" else inputs.sum(-1)
    ).tolist()
    if comparison["source_output_counts"] != expected:
        raise AssertionError("source spike counts differ from the analytical control")
    if comparison["lacuna_output_counts"] != expected:
        raise AssertionError("Lacuna spike counts differ from the analytical control")
    if control == "silence" and any(
        layer["source_spikes"] or layer["lacuna_spikes"]
        for layer in comparison["layers"]
    ):
        raise AssertionError("a silent analytical control emitted hidden spikes")
    return "passed"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(cases):
    completed = [case for case in cases if case["status"] == "completed"]
    comparisons = [case["comparison"] for case in completed]
    return {
        "completed_cases": len(completed),
        "error_cases": len(cases) - len(completed),
        "exact_cases": sum(c["exact_spike_match_on_batch"] for c in comparisons),
        "analytical_controls_passed": sum(
            c["analytical_control_result"] == "passed" for c in completed
        ),
        "paired_predictions": sum(c["samples"] for c in comparisons),
        "paired_predictions_matching": sum(
            sum(
                a == b for a, b in zip(c["source_predictions"], c["lacuna_predictions"])
            )
            for c in comparisons
        ),
        "mismatched_bins": sum(
            layer["mismatched_bins"] for c in comparisons for layer in c["layers"]
        ),
        "off_grid_spikes": sum(len(c["off_grid_spikes"]) for c in comparisons),
    }


def run(output_dir=None):
    torch.set_num_threads(1)
    source_dir = ROOT / "artifacts/lava-dl-official"
    commit = subprocess.check_output(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    clean = not subprocess.check_output(
        ["git", "-C", str(source_dir), "status", "--porcelain"], text=True
    ).strip()
    if commit != SOURCE_COMMIT or not clean:
        raise RuntimeError("the prescribed official source must be unchanged")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(output_dir)
        if output_dir
        else (ROOT / "artifacts/official-slayer-numerical-matrix" / stamp)
    )
    output.mkdir(parents=True, exist_ok=False)
    library = Path(Engine().core._lib._name).resolve()
    importer_hashes = {
        name: sha256(ROOT / name)
        for name in (
            "src/lacuna/importers/slayer.py",
            "src/lacuna/importers/dense_lif.py",
            "src/lacuna/importers/feedforward_lif.py",
        )
    }
    protocol = {
        "purpose": "bounded numerical transfer stress, not classification accuracy",
        "source_commit": commit,
        "source_tree_clean": clean,
        "script_sha256": sha256(__file__),
        "torch_version": torch.__version__,
        "torch_threads": 1,
        "source_dtype": "float32",
        "target_dtype": "binary64",
        "target_library": str(library),
        "target_library_sha256": sha256(library),
        "importer_sha256": importer_hashes,
        "exact_control_rationale": (
            "zero drive plus zero/cancelled deposits stays silent, and a 1.25 "
            "identity deposit always exceeds the source threshold and resets to "
            "zero. These controls have no retained subthreshold state."
        ),
        "stress_acceptance": (
            "Report every spike-bin and count disagreement without a numerical "
            "agreement threshold. Off-grid target spikes and analytical control "
            "failures are errors. Argmax comparisons have no task labels."
        ),
        "cases": case_grid(),
    }
    write_json(output / "protocol.json", protocol)
    report = {
        "protocol_sha256": sha256(output / "protocol.json"),
        "cases": [],
        "status": "running",
    }
    started = time.perf_counter()
    for index, case in enumerate(protocol["cases"], 1):
        directory = output / case["id"]
        directory.mkdir()
        row = {"id": case["id"], "status": "running"}
        case_start = time.perf_counter()
        try:
            model = build_model(case)
            inputs = make_inputs(case)
            imported = import_slayer_feedforward(
                model,
                input_shape=tuple(case["input_shape"]),
                timestep=case["timestep"],
                acknowledge_quantization=True,
                name=case["id"],
            )
            torch.save(inputs.to(torch.uint8), directory / "inputs.pt")
            torch.save(model.state_dict(), directory / "checkpoint.pt")
            write_json(directory / "source_metadata.json", imported.metadata)
            imported.deployment.network.save(directory / "network.json")
            comparison = validate_slayer_feedforward(
                model, imported, inputs, source_batch_size=case["source_batch_size"]
            )
            if comparison["off_grid_spikes"]:
                raise AssertionError("this zero-drive profile produced off-grid spikes")
            row.update(
                {
                    "status": "completed",
                    "comparison": comparison,
                    "analytical_control_result": check_analytical_control(
                        case, inputs, comparison
                    ),
                    "effective_layers": [
                        {k: v for k, v in layer.items() if k != "weights"}
                        for layer in imported.metadata["layers"]
                    ],
                    "source_files_sha256": imported.metadata["source_files_sha256"],
                    "artifact_sha256": {
                        name: sha256(directory / name)
                        for name in (
                            "inputs.pt",
                            "checkpoint.pt",
                            "source_metadata.json",
                            "network.json",
                        )
                    },
                }
            )
        except Exception as error:
            row.update(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        row["elapsed_seconds"] = time.perf_counter() - case_start
        report["cases"].append(row)
        report["summary"] = summarize(report["cases"])
        write_json(directory / "report.json", row)
        write_json(output / "report.json", report)
        print(
            f"[{index}/{len(protocol['cases'])}] {case['id']}: {row['status']}",
            flush=True,
        )
    report["elapsed_seconds"] = time.perf_counter() - started
    if sha256(library) != protocol["target_library_sha256"] or {
        name: sha256(ROOT / name) for name in importer_hashes
    } != importer_hashes:
        raise RuntimeError("target runtime or importer changed during the matrix")
    report["status"] = "failed" if report["summary"]["error_cases"] else "completed"
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output), **report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output_dir)["status"] == "completed" else 1)
