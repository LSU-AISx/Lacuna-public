#!/usr/bin/env python3
"""Evaluate persisted neuron-only MNIST search weights without retraining."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from lacuna import CoreEvaluator
from lacuna.experiments import load_mnist

from mnist_neuron_search import (
    CandidateConfig,
    build_candidate,
    evaluate_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--library", default="build/liblacuna_core.dylib")
    parser.add_argument("--data-dir", default="artifacts/mnist")
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--pairwise-decode", choices=("vote", "margin", "sigmoid", "hybrid")
    )
    parser.add_argument("--pairwise-decode-temperature", type=float)
    parser.add_argument("--pairwise-decode-alpha", type=float)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="artifacts/optimization/mnist_test.json")
    args = parser.parse_args()

    document = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    trial = document[0]
    config = CandidateConfig(**trial["config"])
    threshold = (
        float(trial["evaluation_threshold"])
        if args.threshold is None
        else args.threshold
    )
    config = replace(config, output_threshold=threshold)
    if args.pairwise_decode is not None:
        config = replace(config, pairwise_decode=args.pairwise_decode)
    if args.pairwise_decode_temperature is not None:
        config = replace(
            config,
            pairwise_decode_temperature=args.pairwise_decode_temperature,
        )
    if args.pairwise_decode_alpha is not None:
        config = replace(config, pairwise_decode_alpha=args.pairwise_decode_alpha)
    weights = tuple(float(value) for value in trial["learned_weights"])
    _, _, test_images, test_labels = load_mnist(args.data_dir, download=False)
    if args.limit is not None:
        test_images = test_images[: args.limit]
        test_labels = test_labels[: args.limit]

    candidate = build_candidate(config)
    metrics = evaluate_candidate(
        CoreEvaluator(args.library),
        candidate,
        config,
        weights,
        test_images,
        test_labels,
    )
    result = {
        "source": str(Path(args.artifact)),
        "threshold": threshold,
        "config": asdict(config),
        "metrics": asdict(metrics),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"test={100 * metrics.accuracy:.2f}% samples={metrics.samples} "
        f"spikes={metrics.mean_output_spikes:.2f} "
        f"silent={100 * metrics.silent_fraction:.2f}%"
    )


if __name__ == "__main__":
    main()
