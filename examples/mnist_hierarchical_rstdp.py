#!/usr/bin/env python3
"""Train the local competitive hierarchy, reservoir, and R-STDP readout."""

from __future__ import annotations

import argparse
from pathlib import Path

from lacuna import Engine
from lacuna.experiments import (
    HierarchicalMNISTConfig,
    HiddenPretrainingProgress,
    ProgressUpdate,
    build_hierarchical_mnist_classifier,
    evaluate_classifier,
    freeze_hidden_plasticity,
    load_mnist,
    pretrain_hidden_stages,
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
            "Train a three-stage locally connected competitive LIF hierarchy, "
            "balanced reservoir, and modulated-STDP MNIST readout."
        )
    )
    parser.add_argument("--library", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--train-samples", type=int, default=200)
    parser.add_argument("--pretrain-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--hidden-learning-rate", type=float, default=0.001)
    parser.add_argument("--readout-learning-rate", type=float, default=0.002)
    parser.add_argument(
        "--save",
        default="artifacts/mnist_hierarchical_rstdp.json",
        help="learned network destination; pass an empty string to disable",
    )
    args = parser.parse_args()
    if (
        args.train_samples <= 0
        or args.pretrain_samples < 0
        or args.test_samples <= 0
        or args.epochs <= 0
    ):
        parser.error("sample counts and epochs must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def main() -> None:
    args = _arguments()
    print("Loading MNIST...")
    train_x, train_y, test_x, test_y = load_mnist(
        args.data_dir, download=not args.no_download
    )
    import numpy as np

    config = HierarchicalMNISTConfig(
        hidden_learning_rate=args.hidden_learning_rate,
        learning_rate=args.readout_learning_rate,
    )
    print("Building the local hierarchy (the first graph resolution is substantial)...")
    classifier = build_hierarchical_mnist_classifier(config, seed=args.seed)
    graph = classifier.network.graph
    plastic_edges = sum(edge.plasticity is not None for edge in graph.edges)
    print(
        f"Network: {len(graph.nodes)} LIF neurons, {len(graph.edges)} delta edges, "
        f"{plastic_edges} plastic"
    )
    print(
        "Spatial stages: "
        + " -> ".join(
            f"{shape.rows}x{shape.columns}x{shape.channels}"
            for shape in classifier.stage_shapes
        )
    )
    engine = Engine(args.library or _default_library())
    rng = np.random.default_rng(args.seed)
    train_count = min(args.train_samples, len(train_y))
    pretrain_count = min(args.pretrain_samples, len(train_y))
    test_count = min(args.test_samples, len(test_y))

    def show_hidden(update: HiddenPretrainingProgress) -> None:
        if update.sample % args.progress_every and update.sample != update.samples:
            return
        print(
            f"  stage {update.stage}/{update.stages}  "
            f"{update.sample:>5}/{update.samples}  spikes={update.spikes}"
        )

    if pretrain_count:
        pretrain_indices = rng.permutation(len(train_y))[:pretrain_count]
        print(f"Layer-wise hidden pretraining ({pretrain_count} images per stage)")
        classifier = pretrain_hidden_stages(
            classifier,
            train_x[pretrain_indices],
            engine=engine,
            seed=args.seed + 1_000,
            progress=show_hidden,
        )
    classifier = freeze_hidden_plasticity(classifier)

    def show(update: ProgressUpdate) -> None:
        if update.sample % args.progress_every and update.sample != update.samples:
            return
        counts = dict(update.layer_spikes)
        stages = "/".join(
            str(counts.get(f"stage{index}", 0))
            for index in range(1, len(config.hidden_channels) + 1)
        )
        reservoir = counts.get("reservoir_exc", 0) + counts.get("reservoir_inh", 0)
        print(
            f"  {update.sample:>5}/{update.samples}  "
            f"accuracy={100.0 * update.running_accuracy:5.1f}%  "
            f"hidden={stages}  reservoir={reservoir}  output={update.output_spikes}"
        )

    for epoch in range(args.epochs):
        indices = rng.permutation(len(train_y))[:train_count]
        print(f"Epoch {epoch + 1}/{args.epochs}")
        result = train_classifier(
            classifier,
            train_x[indices],
            train_y[indices],
            engine=engine,
            seed=args.seed + epoch,
            progress=show,
        )
        classifier = result.classifier
        print(
            f"  online accuracy={100.0 * result.metrics.accuracy:.2f}%, "
            f"output spikes/sample={result.metrics.mean_output_spikes:.2f}"
        )

    test = evaluate_classifier(
        classifier,
        test_x[:test_count],
        test_y[:test_count],
        engine=engine,
        seed=args.seed + 10_000,
    )
    print(
        f"Test: {100.0 * test.accuracy:.2f}% accuracy, "
        f"loss={test.mean_loss:.3f}, "
        f"output spikes/sample={test.mean_output_spikes:.2f}"
    )
    if args.save:
        destination = Path(args.save)
        classifier.network.save(destination)
        print(f"Saved learned network to {destination}")


if __name__ == "__main__":
    main()
