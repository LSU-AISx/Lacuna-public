"""Optional protocol and deployment checks for the official SLAYER MNIST pilot."""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("lava.lib.dl.slayer")

from examples.train_official_slayer_mnist import (  # noqa: E402
    build_model,
    classification_metrics,
    encode_images,
    stratified_groups,
)
from lacuna.importers.slayer import (  # noqa: E402
    import_slayer_dense,
    validate_slayer_dense,
)


def test_stratified_groups_are_balanced_disjoint_and_reproducible():
    labels = np.repeat(np.arange(10), np.arange(40, 50))
    first = stratified_groups(labels, (200, 100), seed=137)
    repeated = stratified_groups(labels, (200, 100), seed=137)
    changed = stratified_groups(labels, (200, 100), seed=138)

    assert [len(group) for group in first] == [200, 100]
    assert not np.intersect1d(*first).size
    for group, expected, other, count in zip(
        first, repeated, changed, (20, 10)
    ):
        assert group.dtype == np.int64
        assert len(np.unique(group)) == len(group)
        assert group.min() >= 0 and group.max() < len(labels)
        np.testing.assert_array_equal(group, expected)
        np.testing.assert_array_equal(np.bincount(labels[group]), [count] * 10)
        assert not np.array_equal(group, other)


def test_train_validation_selection_is_independent_of_test_pool():
    training_labels = np.repeat(np.arange(10), 40)
    baseline = stratified_groups(training_labels, (200, 100), seed=137)
    for test_labels in (
        np.repeat(np.arange(10), 15),
        np.repeat(np.arange(10), 27)[::-1],
    ):
        stratified_groups(test_labels, (100,), seed=138)
        actual = stratified_groups(training_labels, (200, 100), seed=137)
        for selected, expected in zip(actual, baseline):
            np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize("sizes", [(), (0,), (-10,), (15,), (True,), (10.0,)])
def test_stratified_groups_reject_invalid_sizes(sizes):
    with pytest.raises(ValueError, match="positive multiples of ten"):
        stratified_groups(np.tile(np.arange(10), 3), sizes, seed=0)


@pytest.mark.parametrize("labels", [[[0, 1]], [-1, 0], [0, 10], [0, 1.5]])
def test_stratified_groups_reject_invalid_labels(labels):
    with pytest.raises(ValueError, match="digit labels"):
        stratified_groups(labels, (10,), seed=0)


def test_stratified_groups_reject_insufficient_class_support():
    with pytest.raises(ValueError, match="not enough examples per digit"):
        stratified_groups(np.tile(np.arange(10), 2), (20, 10), seed=0)


def test_encoder_uses_exact_seeded_pixel_normalization():
    pixels = np.tile(np.array([0, 64, 128, 255], dtype=np.uint8), (2, 196))
    probabilities = 0.4 * torch.tensor(pixels, dtype=torch.float32) / 255.0
    generator = torch.Generator().manual_seed(17)
    expected = (
        torch.rand(2, 784, 32, generator=generator)
        < probabilities[..., None]
    ).float()

    actual = encode_images(pixels, bins=32, seed=17)

    assert actual.shape == (2, 784, 32)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(actual, encode_images(pixels, bins=32, seed=17))
    assert not torch.equal(actual, encode_images(pixels, bins=32, seed=18))
    assert bool(torch.all((actual == 0) | (actual == 1)))
    assert not bool(torch.any(actual[:, ::4]))


def test_encoder_keeps_black_pixels_silent_and_preserves_global_rng():
    before = torch.random.get_rng_state().clone()
    encoded = encode_images(np.zeros((3, 784)), bins=8, seed=3)

    assert not bool(torch.any(encoded))
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "images",
    [
        np.zeros(784),
        np.zeros((1, 28, 28)),
        np.zeros((0, 784)),
        np.zeros((2, 783)),
        np.full((1, 784), -1),
        np.full((1, 784), 256),
        np.full((1, 784), np.nan),
        np.full((1, 784), np.inf),
    ],
)
def test_encoder_rejects_invalid_images(images):
    with pytest.raises(ValueError, match="matrix in"):
        encode_images(images, bins=8, seed=0)


@pytest.mark.parametrize("bins", [0, -1, True, 1.5])
def test_encoder_rejects_invalid_bin_count(bins):
    with pytest.raises(ValueError, match="positive integer"):
        encode_images(np.zeros((1, 784)), bins=bins, seed=0)


def test_metrics_use_truth_rows_and_return_absent_class_recalls():
    result = classification_metrics([0, 0, 1, 2, 9], [0, 1, 1, 9, 9])
    confusion = np.zeros((10, 10), dtype=int)
    confusion[0, 0] = confusion[0, 1] = 1
    confusion[1, 1] = confusion[2, 9] = confusion[9, 9] = 1

    assert result["samples"] == 5
    assert result["correct"] == 3
    assert result["accuracy"] == pytest.approx(0.6)
    assert result["per_digit_recall"] == [
        0.5, 1.0, 0.0, None, None, None, None, None, None, 1.0
    ]
    np.testing.assert_array_equal(result["confusion_matrix"], confusion)


@pytest.mark.parametrize(
    "labels,predictions",
    [
        ([], []),
        ([0, 1], [0]),
        ([[0]], [[0]]),
        ([-1], [0]),
        ([10], [0]),
        ([1.5], [1]),
        ([1], [1.5]),
        ([0], [10]),
        ([0], [np.nan]),
    ],
)
def test_metrics_reject_invalid_digit_vectors(labels, predictions):
    with pytest.raises(ValueError, match="digit labels"):
        classification_metrics(labels, predictions)


def test_wide_dense_validation_exceeds_default_queue_and_output_bounds(core):
    model, _ = build_model(8)
    model = torch.nn.Sequential(model[0]).eval()
    with torch.no_grad():
        model[0].synapse.weight.fill_(1 / 512)
    imported = import_slayer_dense(model, acknowledge_quantization=True)
    inputs = torch.ones(1, 784, 8)

    assert len(imported.deployment.network.graph.edges) == 6272
    assert inputs.numel() > 4096
    report = validate_slayer_dense(
        model, imported, inputs, library=core._lib._name
    )

    assert report["exact_spike_match_on_batch"]
    assert report["prediction_agreement"] == 1.0
    assert report["off_grid_spikes"] == []
    assert report["source_output_counts"] == [[8.0] * 8]
    assert report["lacuna_output_counts"] == [[8.0] * 8]
    assert report["layers"][0]["source_spikes"] == 64
    assert report["layers"][0]["lacuna_spikes"] == 64
