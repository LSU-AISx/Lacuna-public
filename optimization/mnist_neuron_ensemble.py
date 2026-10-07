#!/usr/bin/env python3
"""Evaluate an ensemble of persisted neuron-only pairwise MNIST branches."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from lacuna import CoreEvaluator
from lacuna.experiments import load_mnist

from mnist_neuron_search import (
    CandidateConfig,
    _candidate_evidence,
    _presentations,
    balanced_indices,
    build_candidate,
)


def _load(
    path: str,
    pairwise_decode=None,
    pairwise_decode_temperature=None,
    pairwise_decode_alpha=None,
):
    trial = json.loads(Path(path).read_text(encoding="utf-8"))[0]
    config = CandidateConfig(**trial["config"])
    config = replace(config, output_threshold=float(trial["evaluation_threshold"]))
    if pairwise_decode is not None:
        config = replace(config, pairwise_decode=pairwise_decode)
    if pairwise_decode_temperature is not None:
        config = replace(
            config, pairwise_decode_temperature=pairwise_decode_temperature
        )
    if pairwise_decode_alpha is not None:
        config = replace(config, pairwise_decode_alpha=pairwise_decode_alpha)
    return config, tuple(float(value) for value in trial["learned_weights"])


def _collect(core, config, weights, images):
    candidate = build_candidate(config)
    frozen_graph = replace(
        candidate.network.graph,
        edges=tuple(
            replace(edge, weight=weights[edge.id], plasticity=None)
            for edge in candidate.network.graph.edges
        ),
        modulator_ports=(),
    )
    evidence_rows = []
    pair_difference_rows = []
    with frozen_graph.resolve().compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=config.sample_duration * len(images),
            encoder_seed=config.seed + 1_000_003,
            queue_capacity=16_384,
            output_capacity=8_192,
            encoder_spike_capacity=8_192,
        ) as run:
            for sample, image in enumerate(images):
                start = sample * config.sample_duration
                presentation_end = start + config.presentation
                sample_end = start + config.sample_duration
                result = (
                    run.finish(
                        scalar_inputs=_presentations(
                            candidate, image, start, presentation_end
                        )
                    )
                    if sample + 1 == len(images)
                    else run.advance_until(
                        sample_end,
                        scalar_inputs=_presentations(
                            candidate, image, start, presentation_end
                        ),
                    )
                )
                evidence, _, node_evidence = _candidate_evidence(
                    result.core.spikes,
                    candidate,
                    start,
                    config.presentation + config.settling,
                    config.first_spike_bonus,
                )
                evidence_rows.append(evidence)
                pair_difference_rows.append(
                    tuple(
                        node_evidence[first_node] - node_evidence[second_node]
                        for _, _, first_node, second_node in (
                            candidate.pairwise_output_nodes
                        )
                    )
                )
    return (
        np.asarray(evidence_rows, dtype=float),
        np.asarray(pair_difference_rows, dtype=float),
        tuple(
            (first, second)
            for first, second, _, _ in candidate.pairwise_output_nodes
        ),
    )


def _pair_reliability(differences, pairs, labels):
    values = []
    for pair_index, (first, second) in enumerate(pairs):
        selected = (labels == first) | (labels == second)
        pair_values = differences[selected, pair_index]
        pair_labels = labels[selected]
        correct = np.sum((pair_values > 0.0) & (pair_labels == first))
        correct += np.sum((pair_values < 0.0) & (pair_labels == second))
        correct += 0.5 * np.sum(pair_values == 0.0)
        probability = (float(correct) + 1.0) / (len(pair_values) + 2.0)
        probability = min(max(probability, 0.51), 0.99)
        values.append(np.log(probability / (1.0 - probability)))
    result = np.asarray(values, dtype=float)
    return result / np.mean(result)


def _weighted_pair_evidence(differences, pairs, reliability):
    result = np.zeros((len(differences), 10), dtype=float)
    rows = np.arange(len(differences))
    for pair_index, (first, second) in enumerate(pairs):
        values = differences[:, pair_index]
        weight = reliability[pair_index]
        result[rows[values > 0.0], first] += weight
        result[rows[values < 0.0], second] += weight
        ties = rows[values == 0.0]
        result[ties, first] += 0.5 * weight
        result[ties, second] += 0.5 * weight
    return result


def _calibrate_class_biases(evidence, labels):
    biases = np.zeros(10, dtype=float)
    grid = np.linspace(-1.0, 1.0, 41)
    for _ in range(4):
        changed = False
        for digit in range(10):
            current = biases[digit]
            best = current
            best_accuracy = float(np.mean(np.argmax(evidence + biases, axis=1) == labels))
            for value in grid:
                trial = biases.copy()
                trial[digit] = value
                accuracy = float(
                    np.mean(np.argmax(evidence + trial, axis=1) == labels)
                )
                if accuracy > best_accuracy:
                    best = value
                    best_accuracy = accuracy
            biases[digit] = best
            changed |= best != current
        if not changed:
            break
    return biases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+")
    parser.add_argument("--library", default="build/liblacuna_core.dylib")
    parser.add_argument("--data-dir", default="artifacts/mnist")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--validation-per-class", type=int, default=100)
    parser.add_argument("--validation-offset", type=int, default=500)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--branch-weights", type=float, nargs="*")
    parser.add_argument(
        "--pairwise-decode", choices=("vote", "margin", "sigmoid", "hybrid")
    )
    parser.add_argument("--pairwise-decode-temperature", type=float)
    parser.add_argument("--pairwise-decode-alpha", type=float)
    parser.add_argument("--calibration-from")
    parser.add_argument(
        "--output", default="artifacts/optimization/mnist_ensemble.json"
    )
    args = parser.parse_args()

    train_images, train_labels, test_images, test_labels = load_mnist(
        args.data_dir, download=False
    )
    if args.split == "validation":
        indices = balanced_indices(
            train_labels,
            args.validation_per_class,
            offset=args.validation_offset,
            seed=18,
        )
        images = train_images[indices]
        labels = train_labels[indices]
    else:
        images, labels = test_images, test_labels
    if args.limit is not None:
        images = images[: args.limit]
        labels = labels[: args.limit]

    core = CoreEvaluator(args.library)
    branches = []
    pair_differences = []
    branch_pairs = []
    branch_accuracy = []
    for artifact in args.artifacts:
        config, weights = _load(
            artifact,
            args.pairwise_decode,
            args.pairwise_decode_temperature,
            args.pairwise_decode_alpha,
        )
        evidence, differences, pairs = _collect(core, config, weights, images)
        branches.append(evidence)
        pair_differences.append(differences)
        branch_pairs.append(pairs)
        accuracy = float(np.mean(np.argmax(evidence, axis=1) == labels))
        branch_accuracy.append(accuracy)
        print(f"branch {Path(artifact).name}: {100 * accuracy:.2f}%", flush=True)
    calibration = None
    if args.calibration_from:
        calibration = json.loads(
            Path(args.calibration_from).read_text(encoding="utf-8")
        )
        if calibration["artifacts"] != args.artifacts:
            raise ValueError("calibration artifact order does not match")
        pair_reliability = tuple(
            np.asarray(values, dtype=float)
            for values in calibration["pair_reliability"]
        )
        branches = [
            _weighted_pair_evidence(differences, pairs, reliability)
            for differences, pairs, reliability in zip(
                pair_differences, branch_pairs, pair_reliability
            )
        ]
    elif args.split == "validation":
        pair_reliability = tuple(
            _pair_reliability(differences, pairs, labels)
            for differences, pairs in zip(pair_differences, branch_pairs)
        )
        calibrated = [
            _weighted_pair_evidence(differences, pairs, reliability)
            for differences, pairs, reliability in zip(
                pair_differences, branch_pairs, pair_reliability
            )
        ]
        raw_accuracy = max(
            np.mean(np.argmax(branch, axis=1) == labels) for branch in branches
        )
        calibrated_accuracy = max(
            np.mean(np.argmax(branch, axis=1) == labels) for branch in calibrated
        )
        if calibrated_accuracy >= raw_accuracy:
            branches = calibrated
        else:
            pair_reliability = tuple(
                np.ones(len(pairs), dtype=float) for pairs in branch_pairs
            )
    else:
        pair_reliability = tuple(
            np.ones(len(pairs), dtype=float) for pairs in branch_pairs
        )

    if calibration is not None:
        selected_weights = tuple(calibration["branch_weights"])
    elif args.branch_weights:
        if len(args.branch_weights) != len(branches):
            raise ValueError("one branch weight is required per artifact")
        selected_weights = tuple(args.branch_weights)
    elif args.split == "validation" and len(branches) <= 3:
        candidates = (
            weights
            for weights in itertools.product(range(5), repeat=len(branches))
            if any(weights)
        )
        selected_weights = max(
            candidates,
            key=lambda weights: np.mean(
                np.argmax(
                    sum(
                        weight * evidence
                        for weight, evidence in zip(weights, branches)
                    ),
                    axis=1,
                )
                == labels
            ),
        )
    else:
        selected_weights = (1.0,) * len(branches)
    combined = sum(
        weight * evidence
        for weight, evidence in zip(selected_weights, branches)
    )
    if calibration is not None:
        class_biases = np.asarray(calibration.get("class_biases", [0.0] * 10))
    elif args.split == "validation":
        class_biases = _calibrate_class_biases(combined, labels)
    else:
        class_biases = np.zeros(10, dtype=float)
    combined = combined + class_biases
    accuracy = float(np.mean(np.argmax(combined, axis=1) == labels))
    result = {
        "split": args.split,
        "samples": int(len(labels)),
        "artifacts": args.artifacts,
        "branch_accuracy": branch_accuracy,
        "branch_weights": selected_weights,
        "pair_reliability": [values.tolist() for values in pair_reliability],
        "class_biases": class_biases.tolist(),
        "accuracy": accuracy,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"ensemble: {100 * accuracy:.2f}% weights={selected_weights}",
        flush=True,
    )


if __name__ == "__main__":
    main()
