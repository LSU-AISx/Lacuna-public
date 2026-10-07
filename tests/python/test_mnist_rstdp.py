import pytest

from lacuna import Engine, ModulatedSTDP
from lacuna.experiments.mnist_rstdp import (
    MNISTConfig,
    build_mnist_classifier,
    centered_class_rewards,
    evaluate_classifier,
    freeze_classifier,
    train_classifier,
)
from lacuna.ffi import CoreEvaluator


def test_centered_reward_is_confidence_sensitive_and_zero_sum() -> None:
    probabilities, rewards = centered_class_rewards((0.0, 3.0, 1.0), 0)

    assert sum(probabilities) == pytest.approx(1.0)
    assert sum(rewards) == pytest.approx(0.0, abs=1.0e-15)
    assert rewards[0] > 0.0
    assert rewards[1] < rewards[2] < 0.0

    confident, smaller = centered_class_rewards((5.0, 0.0, 0.0), 0)
    assert confident[0] > probabilities[0]
    assert smaller[0] < rewards[0]


def test_mnist_classifier_uses_lif_delta_edges_and_one_modulator_per_class() -> None:
    classifier = build_mnist_classifier(
        MNISTConfig(feature_count=4, class_count=3), seed=5
    )
    graph = classifier.network.graph

    assert len(graph.nodes) == 7
    assert len(graph.edges) == 12
    assert all(edge.synapse is None for edge in graph.edges)
    assert all(isinstance(edge.plasticity, ModulatedSTDP) for edge in graph.edges)
    assert tuple(len(port.edges) for port in graph.modulator_ports) == (4, 4, 4)
    assert {
        edge for port in graph.modulator_ports for edge in port.edges
    } == set(range(12))
    assert all(model.source.startswith("neuron StandardLIF") for model in graph.models)


def test_probe_then_centered_reward_potentiates_target_and_depresses_competitor(
    core: CoreEvaluator,
) -> None:
    config = MNISTConfig(
        feature_count=4,
        class_count=2,
        presentation_duration=5.0,
        probe_delay=3.0,
        reward_delay=1.0,
        sample_duration=20.0,
        pixel_max_rate=2.0,
        weight_low=0.04,
        weight_high=0.04,
        weight_bounds=(0.001, 0.15),
        learning_rate=0.01,
    )
    classifier = build_mnist_classifier(config, seed=2)
    before = tuple(edge.weight for edge in classifier.network.graph.edges)

    training = train_classifier(
        classifier,
        ((255, 255, 0, 0),),
        (0,),
        engine=Engine(core._lib._name),
        seed=3,
    )
    after = tuple(edge.weight for edge in training.classifier.network.graph.edges)

    assert after[0] > before[0]
    assert after[1] > before[1]
    assert after[2:4] == before[2:4]
    assert after[4] < before[4]
    assert after[5] < before[5]
    assert after[6:] == before[6:]
    frozen = freeze_classifier(training.classifier)
    assert not frozen.network.graph.modulator_ports
    assert all(edge.plasticity is None for edge in frozen.network.graph.edges)
    metrics = evaluate_classifier(
        training.classifier,
        ((255, 255, 0, 0),),
        (0,),
        engine=Engine(core._lib._name),
        seed=4,
    )
    assert metrics.samples == 1
