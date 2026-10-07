"""Reproducible experiments built on Lacuna's public authoring API."""

from .mnist_rstdp import (
    ClassificationMetrics,
    MNISTClassifier,
    MNISTConfig,
    ProgressUpdate,
    TrainingResult,
    build_mnist_classifier,
    centered_class_rewards,
    evaluate_classifier,
    freeze_classifier,
    load_mnist,
    train_classifier,
)
from .hierarchical_mnist_rstdp import (
    HierarchicalMNISTClassifier,
    HierarchicalMNISTConfig,
    HiddenPretrainingProgress,
    SpatialShape,
    build_hierarchical_mnist_classifier,
    hierarchy_shapes,
    local_many_to_one_pairs,
    pretrain_hidden_stages,
    freeze_hidden_plasticity,
)

__all__ = [
    "ClassificationMetrics",
    "MNISTClassifier",
    "MNISTConfig",
    "ProgressUpdate",
    "TrainingResult",
    "HierarchicalMNISTClassifier",
    "HierarchicalMNISTConfig",
    "HiddenPretrainingProgress",
    "SpatialShape",
    "build_mnist_classifier",
    "centered_class_rewards",
    "build_hierarchical_mnist_classifier",
    "evaluate_classifier",
    "freeze_classifier",
    "load_mnist",
    "hierarchy_shapes",
    "local_many_to_one_pairs",
    "pretrain_hidden_stages",
    "freeze_hidden_plasticity",
    "train_classifier",
]
