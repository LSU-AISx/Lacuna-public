#!/usr/bin/env python3
"""Train the selected neuron-only MNIST architecture on all training images."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from lacuna import CoreEvaluator
from lacuna.experiments import load_mnist

from mnist_neuron_search import CandidateConfig, Metrics, run_trial


DEFAULT_SCHEDULE = (0.01, 0.0075, 0.005, 0.004, 0.003, 0.002,
                    0.0015, 0.001, 0.00075, 0.0005)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="artifacts/optimization/mnist_pairwise_pool2_10000_disjoint.json",
    )
    parser.add_argument("--library", default="build/liblacuna_core.dylib")
    parser.add_argument("--data-dir", default="artifacts/mnist")
    parser.add_argument(
        "--learning-rate-schedule",
        type=float,
        nargs="+",
        default=DEFAULT_SCHEDULE,
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument(
        "--evaluation-threshold",
        type=float,
        help=(
            "fixed threshold used for the test split; defaults to the "
            "source architecture's selected output threshold"
        ),
    )
    parser.add_argument("--event-queue-capacity", type=int)
    parser.add_argument("--output-capacity", type=int)
    parser.add_argument(
        "--checkpoint",
        help="checkpoint path; defaults beside the output artifact",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from a compatible checkpoint when one exists",
    )
    parser.add_argument(
        "--output",
        default="artifacts/optimization/mnist_pairwise_pool2_full60000.json",
    )
    args = parser.parse_args()

    source_trial = json.loads(Path(args.source).read_text(encoding="utf-8"))[0]
    config = CandidateConfig(**source_trial["config"])
    config_overrides = {}
    if args.event_queue_capacity is not None:
        config_overrides["event_queue_capacity"] = args.event_queue_capacity
    if args.output_capacity is not None:
        config_overrides["output_capacity"] = args.output_capacity
    if config_overrides:
        config = replace(config, **config_overrides)
    evaluation_threshold = (
        config.output_threshold
        if args.evaluation_threshold is None
        else args.evaluation_threshold
    )
    train_images, train_labels, test_images, test_labels = load_mnist(
        args.data_dir, download=False
    )
    if args.train_limit is not None:
        train_images = train_images[: args.train_limit]
        train_labels = train_labels[: args.train_limit]
    if args.evaluation_limit is not None:
        test_images = test_images[: args.evaluation_limit]
        test_labels = test_labels[: args.evaluation_limit]

    schedule = tuple(args.learning_rate_schedule)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint is not None
        else Path(args.output).with_suffix(".checkpoint.json")
    )
    checkpoint_identity = {
        "source_architecture": args.source,
        "config": asdict(config),
        "learning_rate_schedule": list(schedule),
        "training_samples": int(len(train_labels)),
        "evaluation_samples": int(len(test_labels)),
        "evaluation_threshold": evaluation_threshold,
    }
    initial_weights = None
    initial_metrics = ()
    completed_training_blocks = 0
    elapsed_seconds = 0.0
    if args.resume and checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity") != checkpoint_identity:
            raise RuntimeError(
                f"incompatible full-training checkpoint: {checkpoint_path}"
            )
        initial_weights = checkpoint["learned_weights"]
        initial_metrics = tuple(
            Metrics(**metrics) for metrics in checkpoint["training_metrics"]
        )
        completed_training_blocks = int(checkpoint["completed_training_blocks"])
        elapsed_seconds = float(checkpoint["elapsed_seconds"])
        print(
            f"resuming after {completed_training_blocks} training blocks "
            f"({sum(item.samples for item in initial_metrics)} samples)",
            flush=True,
        )

    def save_checkpoint(weights, metrics, completed_blocks, elapsed):
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "identity": checkpoint_identity,
            "completed_training_blocks": completed_blocks,
            "elapsed_seconds": elapsed,
            "training_metrics": [asdict(item) for item in metrics],
            "learned_weights": [float(value) for value in weights],
        }
        temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(checkpoint_path)
        samples = sum(item.samples for item in metrics)
        correct = sum(item.accuracy * item.samples for item in metrics)
        print(
            f"checkpoint {completed_blocks}: samples={samples} "
            f"online_accuracy={100.0 * correct / samples:.2f}% "
            f"seconds={elapsed:.1f}",
            flush=True,
        )

    result = run_trial(
        CoreEvaluator(args.library),
        config,
        train_images,
        train_labels,
        test_images,
        test_labels,
        learning_rate_schedule=schedule,
        evaluation_thresholds=(evaluation_threshold,),
        initial_weights=initial_weights,
        initial_metrics=initial_metrics,
        completed_training_blocks=completed_training_blocks,
        elapsed_seconds=elapsed_seconds,
        checkpoint_callback=save_checkpoint,
    )
    serialized = asdict(result)
    serialized["protocol"] = {
        "training_split": "complete canonical MNIST training split",
        "training_samples": int(len(train_labels)),
        "evaluation_split": "complete canonical MNIST test split",
        "evaluation_samples": int(len(test_labels)),
        "evaluation_threshold_fixed_before_test": evaluation_threshold,
        "source_architecture": args.source,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([serialized], indent=2), encoding="utf-8")
    print(
        f"complete: train={100 * result.train.accuracy:.2f}% "
        f"test={100 * result.validation.accuracy:.2f}% "
        f"seconds={result.seconds:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
