"""Checks for the frozen multi-seed transfer protocol and paired summaries."""

from pathlib import Path

import pytest

from validation.official_slayer_transfer_campaign import (
    aggregate,
    command,
    fixed_cases,
    paired_metrics,
)


def test_campaign_contains_three_fixed_seeds_for_each_architecture():
    cases = fixed_cases()
    assert len(cases) == 6
    for name in ("dense", "convolutional"):
        assert [
            case["seed"] for case in cases if case["architecture"] == name
        ] == [0, 1, 2]


def test_campaign_commands_freeze_training_and_split_options():
    for case in fixed_cases():
        invocation = command(case, Path("output"), Path("data"))
        options = dict(zip(invocation[2::2], invocation[3::2]))
        assert options == {
            "--output": "output",
            "--data-dir": "data",
            "--train-samples": "10000",
            "--validation-samples": "1000",
            "--test-samples": "1000",
            "--epochs": "10",
            "--bins": "32",
            "--batch-size": "128",
            "--learning-rate": "0.003",
            "--split-seed": "137",
            "--seed": str(case["seed"]),
        }


def test_paired_metrics_distinguish_prediction_change_from_accuracy_change():
    metrics = paired_metrics([0, 1, 2, 3], [0, 1, 0, 0], [0, 0, 2, 1])
    assert metrics["source_correct"] == metrics["lacuna_correct"] == 2
    assert metrics["accuracy_change_percentage_points"] == 0
    assert metrics["differing_predictions"] == 3
    assert metrics["prediction_agreement"] == 0.25
    assert metrics["different_prediction_samples"] == [1, 2, 3]


@pytest.mark.parametrize(
    "labels,source,target",
    [([], [], []), ([1], [], [1]), ([1, 2], [1], [1, 2])],
)
def test_paired_metrics_reject_incomplete_sequences(labels, source, target):
    with pytest.raises(ValueError):
        paired_metrics(labels, source, target)


def test_aggregate_reports_seed_variation_without_treating_images_as_independent():
    rows = []
    for seed in range(3):
        metrics = paired_metrics([0, 1], [0, 1], [0, 1 if seed == 0 else 0])
        rows.append(
            {
                "architecture": "dense",
                "seed": seed,
                "test": metrics,
                "validation": metrics,
            }
        )
    summary = aggregate(rows)["dense"]
    assert summary["runs"] == 3
    assert summary["mean_source_accuracy"] == 1.0
    assert summary["mean_lacuna_accuracy"] == pytest.approx(2 / 3)
    assert summary["mean_accuracy_change_percentage_points"] == pytest.approx(
        -100 / 3
    )
    assert summary["test_prediction_disagreements_by_seed"] == [0, 1, 1]
    assert summary["independent_test_sets"] is False
    assert aggregate([]) == {}
