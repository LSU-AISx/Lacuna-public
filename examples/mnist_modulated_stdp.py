#!/usr/bin/env python3
"""Train and evaluate Lacuna's direct reward-modulated STDP MNIST baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from lacuna import Engine
from lacuna.experiments.mnist_rstdp import (
    MNISTConfig,
    ProgressUpdate,
    build_mnist_classifier,
    evaluate_classifier,
    load_mnist,
    train_classifier,
)


def _default_library() -> str:
    candidates = sorted(Path("build").glob("liblacuna_core.*"))
    if not candidates:
        raise SystemExit(
            "Lacuna's C core is not built. Run: "
            "cmake -S . -B build && cmake --build build"
        )
    return str(candidates[0])


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a 784-LIF to 10-LIF delta-synapse classifier with "
            "class-scoped reward-modulated STDP."
        )
    )
    parser.add_argument("--library", default=None, help="path to liblacuna_core")
    parser.add_argument("--data-dir", default=None, help="MNIST cache directory")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--train-samples", type=int, default=1_000)
    parser.add_argument("--test-samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--max-rate", type=float, default=0.20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--save",
        default="artifacts/mnist_modulated_stdp.json",
        help="learned network destination; pass an empty string to disable",
    )
    args = parser.parse_args()
    if args.train_samples <= 0 or args.test_samples <= 0 or args.epochs <= 0:
        parser.error("sample counts and epochs must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main() -> None:
    args = _arguments()
    library = args.library or _default_library()
    print("Loading MNIST (the first run downloads about 12 MB)...")
    train_x, train_y, test_x, test_y = load_mnist(
        args.data_dir, download=not args.no_download
    )

    import numpy as np

    rng = np.random.default_rng(args.seed)
    train_count = min(args.train_samples, len(train_y))
    test_count = min(args.test_samples, len(test_y))
    test_indices = np.arange(test_count)
    config = MNISTConfig(
        learning_rate=args.learning_rate,
        pixel_max_rate=args.max_rate,
    )
    classifier = build_mnist_classifier(config, seed=args.seed)
    engine = Engine(library)
    print(
        f"Network: {config.feature_count + config.class_count} LIF neurons, "
        f"{config.feature_count * config.class_count} plastic delta edges"
    )

    def show_progress(update: ProgressUpdate) -> None:
        if update.sample % args.progress_every == 0 or update.sample == update.samples:
            print(
                f"  {update.sample:>6}/{update.samples}  "
                f"online accuracy={100.0 * update.running_accuracy:5.1f}%  "
                f"loss={update.running_loss:.3f}  "
                f"output spikes={update.output_spikes}"
            )

    for epoch in range(args.epochs):
        indices = rng.permutation(len(train_y))[:train_count]
        print(f"Epoch {epoch + 1}/{args.epochs}")
        training = train_classifier(
            classifier,
            train_x[indices],
            train_y[indices],
            engine=engine,
            seed=args.seed + epoch,
            progress=show_progress,
        )
        classifier = training.classifier
        metrics = training.metrics
        print(
            f"  online: {100.0 * metrics.accuracy:.2f}% accuracy, "
            f"{metrics.mean_output_spikes:.2f} output spikes/sample"
        )

    test = evaluate_classifier(
        classifier,
        test_x[test_indices],
        test_y[test_indices],
        engine=engine,
        seed=args.seed + 10_000,
    )
    print(
        f"Test: {100.0 * test.accuracy:.2f}% accuracy, "
        f"loss={test.mean_loss:.3f}, "
        f"{test.mean_output_spikes:.2f} output spikes/sample"
    )
    if args.save:
        destination = Path(args.save)
        classifier.network.save(destination)
        print(f"Saved learned network to {destination}")


if __name__ == "__main__":
    main()
