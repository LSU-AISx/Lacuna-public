"""Optional checks for the official SLAYER WDBC pilot protocol."""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lava.lib.dl.slayer")

from examples.train_official_slayer_wdbc import (  # noqa: E402
    encode_features,
    metrics,
    scale_features,
    selection_key,
    stratified_split,
)


def test_encoder_matches_seeded_interleaved_complementary_rates():
    features = np.array([[0.0, 0.25, 1.0], [0.75, 0.5, 0.0]])
    probabilities = torch.tensor(
        [[0.02, 0.45, 0.1275, 0.3425, 0.45, 0.02],
         [0.3425, 0.1275, 0.235, 0.235, 0.02, 0.45]],
        dtype=torch.float32,
    )
    generator = torch.Generator().manual_seed(17)
    expected = (
        torch.rand(2, 6, 64, generator=generator)
        < probabilities[..., None]
    ).float()

    encoded = encode_features(features, bins=64, seed=17)

    assert encoded.shape == (2, 6, 64)
    assert encoded.dtype == torch.float32
    assert torch.equal(encoded, expected)
    assert set(encoded.flatten().tolist()) == {0.0, 1.0}
    assert torch.equal(encoded, encode_features(features, bins=64, seed=17))
    assert not torch.equal(
        encoded, encode_features(features, bins=64, seed=18)
    )


def test_encoder_does_not_consume_global_torch_random_state():
    before = torch.random.get_rng_state().clone()
    encode_features([[0.2, 0.8]], bins=16, seed=7)
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "features",
    [
        [0.5],
        [[[0.5]]],
        [[-0.01]],
        [[1.01]],
        [[float("nan")]],
        [[float("inf")]],
    ],
)
def test_encoder_rejects_malformed_features(features):
    with pytest.raises(ValueError, match="matrix in"):
        encode_features(features, bins=16, seed=7)


@pytest.mark.parametrize("bins", [0, -1])
def test_encoder_rejects_nonpositive_bin_count(bins):
    with pytest.raises(ValueError, match="bins must be positive"):
        encode_features([[0.5]], bins=bins, seed=7)


def test_metrics_use_truth_rows_and_malignant_positive():
    result = metrics([0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1])

    assert result["confusion_matrix"] == [[3, 1], [1, 1]]
    assert result["samples"] == 6
    assert result["correct"] == 4
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["benign_specificity"] == pytest.approx(3 / 4)
    assert result["malignant_sensitivity"] == pytest.approx(1 / 2)
    assert result["balanced_accuracy"] == pytest.approx(5 / 8)


@pytest.mark.parametrize(
    "labels,predictions",
    [
        ([0, 1], [0]),
        ([[0, 1]], [[0, 1]]),
        ([0, 2], [0, 1]),
        ([0, 1], [-1, 1]),
        ([0, 0], [0, 1]),
        ([], []),
    ],
)
def test_metrics_reject_invalid_class_vectors(labels, predictions):
    with pytest.raises(ValueError):
        metrics(labels, predictions)


def test_selection_prioritizes_balanced_accuracy_before_accuracy_and_loss():
    baseline = {"balanced_accuracy": 0.8, "accuracy": 0.8, "loss": 0.1}
    better_balance = {
        "balanced_accuracy": 0.81, "accuracy": 0.75, "loss": 0.2
    }
    better_accuracy = {**baseline, "accuracy": 0.81, "loss": 0.2}
    lower_loss = {**baseline, "loss": 0.09}

    assert selection_key(1, better_balance) > selection_key(1, baseline)
    assert selection_key(1, better_accuracy) > selection_key(1, baseline)
    assert selection_key(1, lower_loss) > selection_key(1, baseline)


@pytest.mark.parametrize("invalid", ([0.9, 1.0], [0.0, 1.1]))
def test_metrics_rejects_fractional_class_ids(invalid):
    with pytest.raises(ValueError, match="binary"):
        metrics(invalid, [0, 1])
    with pytest.raises(ValueError, match="binary"):
        metrics([0, 1], invalid)


def test_selection_ignores_test_metrics_and_prefers_earlier_epoch_on_tie():
    validation = {"balanced_accuracy": 0.8, "accuracy": 0.8, "loss": 0.1}
    test_loser = {**validation, "test_accuracy": 0.0}
    test_winner = {**validation, "test_accuracy": 1.0}

    assert selection_key(10, test_loser) == selection_key(10, test_winner)
    assert selection_key(10, validation) > selection_key(11, validation)


def test_scaler_uses_training_only_and_clips_held_out_values():
    train = np.array([[2.0, 4.0, 7.0], [6.0, 12.0, 7.0]])
    validation = np.array([[4.0, 8.0, 7.0]])
    test = np.array([[-2.0, 20.0, 8.0]])
    transformed, minimum, maximum = scale_features(train, validation, test)

    np.testing.assert_array_equal(minimum, [2.0, 4.0, 7.0])
    np.testing.assert_array_equal(maximum, [6.0, 12.0, 7.0])
    np.testing.assert_array_equal(transformed[0], [[0, 0, 0], [1, 1, 0]])
    np.testing.assert_array_equal(transformed[1], [[0.5, 0.5, 0.0]])
    np.testing.assert_array_equal(transformed[2], [[0.0, 1.0, 1.0]])

    altered, altered_minimum, altered_maximum = scale_features(
        train, validation * 1000, test * -1000
    )
    np.testing.assert_array_equal(altered[0], transformed[0])
    np.testing.assert_array_equal(altered_minimum, minimum)
    np.testing.assert_array_equal(altered_maximum, maximum)


def test_stratified_patient_split_is_disjoint_complete_and_reproducible():
    labels = np.array([0] * 357 + [1] * 212, dtype=np.int64)
    splits = stratified_split(labels, 137)
    repeated = stratified_split(labels, 137)

    assert [len(indices) for indices in splits] == [343, 113, 113]
    assert [np.bincount(labels[indices]).tolist() for indices in splits] == [
        [215, 128], [71, 42], [71, 42]
    ]
    joined = np.concatenate(splits)
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(labels)))
    for actual, expected in zip(splits, repeated):
        np.testing.assert_array_equal(actual, expected)
