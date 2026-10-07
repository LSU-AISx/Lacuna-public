"""Run a fixed three-seed transfer study with the existing official pilots."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "dense": "examples/train_official_slayer_mnist.py",
    "convolutional": "examples/train_official_slayer_conv_mnist.py",
}
SOURCE_FILES = (
    *SCRIPTS.values(),
    "validation/official_slayer_transfer_campaign.py",
    "src/lacuna/importers/slayer.py",
    "src/lacuna/importers/dense_lif.py",
    "src/lacuna/importers/feedforward_lif.py",
    "src/lacuna/experiments/mnist_rstdp.py",
    "src/lacuna/ffi.py",
    "src/lacuna/simulation.py",
    "src/lacuna/network.py",
    "src/lacuna/graph.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixed_cases():
    """Vary only the model and encoding seed, not the model settings."""

    return [
        {"architecture": architecture, "seed": seed}
        for architecture in SCRIPTS
        for seed in (0, 1, 2)
    ]


def command(case, output, data_dir):
    return [
        sys.executable,
        str(ROOT / SCRIPTS[case["architecture"]]),
        "--output",
        str(output),
        "--data-dir",
        str(data_dir),
        "--train-samples",
        "10000",
        "--validation-samples",
        "1000",
        "--test-samples",
        "1000",
        "--epochs",
        "10",
        "--bins",
        "32",
        "--batch-size",
        "128",
        "--learning-rate",
        "0.003",
        "--split-seed",
        "137",
        "--seed",
        str(case["seed"]),
    ]


def paired_metrics(labels, source, target):
    if not labels or not (len(labels) == len(source) == len(target)):
        raise ValueError(
            "paired labels and predictions must have equal nonzero lengths"
        )
    n = len(labels)
    source_correct = sum(a == y for a, y in zip(source, labels))
    target_correct = sum(b == y for b, y in zip(target, labels))
    differing = [i for i, (a, b) in enumerate(zip(source, target)) if a != b]
    return {
        "samples": n,
        "source_correct": source_correct,
        "lacuna_correct": target_correct,
        "source_accuracy": source_correct / n,
        "lacuna_accuracy": target_correct / n,
        "accuracy_change_percentage_points": 100
        * (target_correct - source_correct)
        / n,
        "differing_predictions": len(differing),
        "prediction_agreement": (n - len(differing)) / n,
        "different_prediction_samples": differing,
    }


def audit_run(directory, case):
    import torch

    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["model_seed"] != case["seed"] or protocol["split_seed"] != 137:
        raise ValueError("run seed does not match the fixed campaign")
    for field, count in (
        ("train_samples", 10000),
        ("validation_samples", 1000),
        ("test_samples", 1000),
    ):
        if report[field] != count:
            raise ValueError("run data size does not match the fixed campaign")
    if (
        report["epochs_run"] != 10
        or report["learning_rate"] != 0.003
        or report["batch_size"] != 128
    ):
        raise ValueError("run training settings changed")
    if protocol["bins"] != 32 or protocol["timestep"] != 1.0:
        raise ValueError("run encoding settings changed")
    if protocol["test_used_for_selection"]:
        raise ValueError("test predictions must not select checkpoints")
    if set(protocol["train_indices"]) & set(protocol["validation_indices"]):
        raise ValueError("training and validation indices overlap")
    for filename, expected in report["files_sha256"].items():
        if digest(directory / filename) != expected:
            raise ValueError(f"artifact hash changed: {filename}")
    if report["official_source_dirty"]:
        raise ValueError("official source was modified")
    saved = torch.load(directory / "validation_inputs.pt", weights_only=True)
    split_results = {}
    for split in ("validation", "test"):
        comparison = json.loads(
            (directory / f"{split}_comparison.json").read_text()
        )
        labels = saved[f"{split}_labels"].tolist()
        if comparison["off_grid_spikes"]:
            raise ValueError(
                "unexpected off-grid spikes in zero-drive delta model"
            )
        metrics = paired_metrics(
            labels,
            comparison["source_predictions"],
            comparison["lacuna_predictions"],
        )
        if split == "test":
            if (
                metrics["source_correct"] != report["official_test"]["correct"]
                or metrics["lacuna_correct"]
                != report["lacuna_test"]["correct"]
            ):
                raise ValueError(
                    "stored accuracy disagrees with saved predictions and labels"
                )
        split_results[split] = {
            **metrics,
            "source_batch_size": comparison["source_batch_size"],
            "layers": comparison["layers"],
            "exact_spike_match": comparison["exact_spike_match_on_batch"],
            "off_grid_spikes": comparison["off_grid_spikes"],
        }
    return {
        **case,
        "directory": str(directory),
        "report_sha256": digest(directory / "report.json"),
        "selected_epoch": report["selected_epoch"],
        "training_seconds": report["training_seconds"],
        "weight_change_l2_by_layer": report["weight_change_l2_by_layer"],
        "source_commit": report["official_source_commit"],
        "dataset_sha256": protocol["dataset_sha256"],
        "split_indices_sha256": hashlib.sha256(
            json.dumps(
                {
                    key: protocol[key]
                    for key in (
                        "train_indices",
                        "validation_indices",
                        "canonical_test_indices",
                    )
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        **split_results,
    }


def aggregate(results):
    """Report descriptive seed variation, not independent-data confidence."""

    summaries = {}
    for architecture in SCRIPTS:
        selected = [
            row for row in results if row["architecture"] == architecture
        ]
        if not selected:
            continue
        tests = [row["test"] for row in selected]
        summaries[architecture] = {
            "runs": len(selected),
            "mean_source_accuracy": statistics.mean(
                row["source_accuracy"] for row in tests
            ),
            "mean_lacuna_accuracy": statistics.mean(
                row["lacuna_accuracy"] for row in tests
            ),
            "mean_accuracy_change_percentage_points": statistics.mean(
                row["accuracy_change_percentage_points"] for row in tests
            ),
            "source_accuracy_range": [
                min(row["source_accuracy"] for row in tests),
                max(row["source_accuracy"] for row in tests),
            ],
            "lacuna_accuracy_range": [
                min(row["lacuna_accuracy"] for row in tests),
                max(row["lacuna_accuracy"] for row in tests),
            ],
            "test_prediction_disagreements_by_seed": [
                row["differing_predictions"] for row in tests
            ],
            "validation_prediction_disagreements_by_seed": [
                row["validation"]["differing_predictions"] for row in selected
            ],
            "independent_test_sets": False,
        }
    return summaries


def run(args):
    import torch
    from lacuna import Engine
    from lava.lib.dl import slayer

    output = (
        Path(args.output).resolve()
        if args.output
        else ROOT
        / "artifacts/official-slayer-transfer-campaign"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(slayer.__file__).resolve().parents[5]
    if subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("official SLAYER source must be clean")
    library = Path(Engine().core._lib._name).resolve()
    hashes = {name: digest(ROOT / name) for name in SOURCE_FILES}
    manifest = {
        "purpose": "conversion validation, not model or hyperparameter selection",
        "cases": fixed_cases(),
        "fresh_runs": True,
        "training": {
            "train_samples": 10000,
            "validation_samples": 1000,
            "test_samples": 1000,
            "epochs": 10,
            "batch_size": 128,
            "learning_rate": 0.003,
            "bins": 32,
            "split_seed": 137,
        },
        "seed_scope": "model initialization, training order, and training/validation/test spike encoding",
        "same_image_indices_across_runs": True,
        "test_predictions_used_for_selection": False,
        "source_inference_batch_sizes": {"dense": 64, "convolutional": 128},
        "uncertainty": "three seeds on one shared image split, descriptive variation only, no population accuracy CI",
        "acceptance": "all runs complete with verified artifacts, no off-grid spikes; source/target differences are reported without a post-hoc accuracy cutoff",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_files_sha256": hashes,
        "library": str(library),
        "library_sha256": digest(library),
        "torch_version": str(torch.__version__),
        "python": sys.version,
    }
    write_json(output / "protocol.json", manifest)
    print(
        json.dumps(
            {"campaign": str(output), "planned_runs": len(manifest["cases"])}
        ),
        flush=True,
    )
    results = []
    try:
        for index, case in enumerate(manifest["cases"], start=1):
            if {
                name: digest(ROOT / name) for name in hashes
            } != hashes or digest(library) != manifest["library_sha256"]:
                raise ValueError("execution sources changed during campaign")
            label = f"{case['architecture']}-seed-{case['seed']}"
            destination = output / label
            print(
                json.dumps(
                    {
                        "run": index,
                        "of": len(manifest["cases"]),
                        "case": label,
                        "stage": "starting",
                    }
                ),
                flush=True,
            )
            with (output / f"{label}.log").open("w") as log:
                process = subprocess.Popen(
                    command(case, destination, Path(args.data_dir).resolve()),
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("{"):
                        print(f"[{label}] {line.rstrip()}", flush=True)
                if process.wait() != 0:
                    raise RuntimeError(
                        f"{label} failed, see its preserved log"
                    )
            result = audit_run(destination, case)
            if results and (
                result["split_indices_sha256"]
                != results[0]["split_indices_sha256"]
                or result["dataset_sha256"] != results[0]["dataset_sha256"]
            ):
                raise ValueError(
                    "dataset or shared image split changed between runs"
                )
            results.append(result)
            write_json(
                output / "progress.json",
                {"completed_runs": results, "aggregate": aggregate(results)},
            )
        if {name: digest(ROOT / name) for name in hashes} != hashes or digest(
            library
        ) != manifest["library_sha256"]:
            raise ValueError("execution sources changed during campaign")
        report = {
            "status": "complete",
            "protocol_sha256": digest(output / "protocol.json"),
            "runs": results,
            "aggregate": aggregate(results),
        }
        write_json(output / "report.json", report)
    except Exception as exc:
        write_json(
            output / "failure.json",
            {
                "status": "incomplete",
                "error": str(exc),
                "completed_runs": len(results),
            },
        )
        raise
    print(
        json.dumps(
            {"campaign": str(output), "aggregate": report["aggregate"]}
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--data-dir", default="artifacts/mnist-data")
    run(parser.parse_args())
