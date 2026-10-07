"""Online reward-modulated STDP baseline for MNIST.

The experiment intentionally uses only standard LIF neurons and delta
synapses. Python presents samples and computes a class-scoped third factor.
all neuron, synapse, trace, and weight-update execution remains in Lacuna's C
evaluator.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import os
import struct
import tempfile
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from ..codec import NativeEventEncoder, PoissonRateEncoder
from ..models import LIF
from ..network import Network, NetworkBuilder, Uniform
from ..plasticity import ModulatedSTDP
from ..recording import RecordingPlan, RunOptions, SpikeRecording
from ..simulation import (
    Engine,
    ModulationSeries,
    ScalarPresentation,
    SimulationResult,
    SpikeTrain,
)
from ..synapses import Delta


@dataclass(frozen=True)
class MNISTConfig:
    """Timing, encoding, and learning parameters for the direct classifier."""

    feature_count: int = 28 * 28
    class_count: int = 10
    presentation_duration: float = 25.0
    settling_duration: float = 0.0
    probe_delay: float = 3.0
    reward_delay: float = 1.0
    sample_duration: float = 80.0
    pixel_max_rate: float = 0.20
    pixel_spike_amplitude: float = 20.0
    probe_spike_amplitude: float = 20.0
    output_threshold: float = -55.0
    weight_low: float = 0.04
    weight_high: float = 0.08
    weight_bounds: tuple[float, float] = (0.001, 0.15)
    learning_rate: float = 0.002
    trace_tau: float = 20.0
    eligibility_tau: float = 100.0
    softmax_temperature: float = 1.0
    first_spike_bonus: float = 0.5
    reward_scale: float = 1.0
    probe_all_classes: bool = True

    def __post_init__(self) -> None:
        if self.feature_count <= 0 or self.class_count <= 1:
            raise ValueError("feature_count must be positive and class_count must exceed one")
        positive = (
            "presentation_duration",
            "probe_delay",
            "reward_delay",
            "sample_duration",
            "pixel_max_rate",
            "pixel_spike_amplitude",
            "probe_spike_amplitude",
            "learning_rate",
            "trace_tau",
            "eligibility_tau",
            "softmax_temperature",
            "reward_scale",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.first_spike_bonus < 0.0 or not math.isfinite(self.first_spike_bonus):
            raise ValueError("first_spike_bonus must be nonnegative and finite")
        if not math.isfinite(self.settling_duration) or self.settling_duration < 0.0:
            raise ValueError("settling_duration must be nonnegative and finite")
        if not math.isfinite(self.output_threshold) or self.output_threshold <= -65.0:
            raise ValueError("output_threshold must be finite and exceed -65")
        if not isinstance(self.probe_all_classes, bool):
            raise ValueError("probe_all_classes must be boolean")
        reward_time = (
            self.presentation_duration
            + self.settling_duration
            + self.probe_delay
            + self.reward_delay
        )
        if self.sample_duration <= reward_time:
            raise ValueError("sample_duration must leave an idle interval after reward")
        lower, upper = self.weight_bounds
        if not 0.0 <= lower < upper:
            raise ValueError("weight_bounds must satisfy 0 <= lower < upper")
        if not lower <= self.weight_low <= self.weight_high <= upper:
            raise ValueError("initial weights must lie inside weight_bounds")


@dataclass(frozen=True)
class MNISTClassifier:
    """Network plus the stable ports and edge groups used by the trainer."""

    network: Network
    config: MNISTConfig
    pixel_ports: tuple[str, ...]
    probe_ports: tuple[str, ...]
    reward_ports: tuple[str, ...]
    output_nodes: tuple[int, ...]
    class_edges: tuple[tuple[int, ...], ...]
    activity_populations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClassificationMetrics:
    """Classification counts and mean reward for one evaluation pass."""

    samples: int
    correct: int
    mean_loss: float
    mean_output_spikes: float

    @property
    def accuracy(self) -> float:
        """Return correct predictions divided by evaluated samples."""

        return 0.0 if self.samples == 0 else self.correct / self.samples


@dataclass(frozen=True)
class ProgressUpdate:
    """Periodic training progress emitted by an experiment runner."""

    sample: int
    samples: int
    target: int
    prediction: int
    running_accuracy: float
    running_loss: float
    output_spikes: int
    layer_spikes: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class TrainingResult:
    """Trained classifier, epoch metrics, and learned output weights."""

    classifier: MNISTClassifier
    metrics: ClassificationMetrics


def build_mnist_classifier(
    config: MNISTConfig = MNISTConfig(), *, seed: int = 0
) -> MNISTClassifier:
    """Build a dense pixel-to-class R-STDP network.

    Each class has its own modulator port targeting exactly the incoming edges
    of that class.  Every edge still owns only one plasticity rule.
    """

    builder = NetworkBuilder(
        "mnist_modulated_stdp",
        metadata={
            "benchmark": "MNIST",
            "architecture": "dense_lif_delta_rstdp",
            "reward": "centered_class_advantage",
        },
    )
    pixels = builder.population(
        "pixels",
        config.feature_count,
        LIF(
            name="mnist_pixel_lif",
            tau_m=10.0,
            refractory=1.0,
        ),
    )
    outputs = builder.population(
        "outputs",
        config.class_count,
        LIF(
            name="mnist_output_lif",
            tau_m=20.0,
            v_threshold=config.output_threshold,
            refractory=2.0,
        ),
    )
    pixel_ports = tuple(
        builder.inputs(
            "pixel",
            pixels,
            encoder=PoissonRateEncoder(
                0.0,
                config.pixel_max_rate,
                amplitude=config.pixel_spike_amplitude,
            ),
        )
    )
    probe_ports = tuple(
        builder.inputs("eligibility_probe", outputs, encoder=NativeEventEncoder())
    )
    builder.outputs("digit", outputs)

    rule = ModulatedSTDP(
        tau_pre=config.trace_tau,
        tau_post=config.trace_tau,
        tau_eligibility_plus=config.eligibility_tau,
        tau_eligibility_minus=config.eligibility_tau,
        learning_rate=config.learning_rate,
        bounds=config.weight_bounds,
        consume_on_modulation=True,
    )
    reward_ports = []
    class_edges = []
    for digit in range(config.class_count):
        edges = builder.connect(
            pixels,
            outputs[digit],
            synapse=Delta(),
            weight=Uniform(
                config.weight_low,
                config.weight_high,
                seed=seed + 104_729 * (digit + 1),
            ),
            delay=0.0,
            plasticity=rule,
        )
        class_edges.append(tuple(edges))
        reward_ports.append(builder.modulator(f"reward[{digit}]", targets=edges))

    network = builder.build()
    return MNISTClassifier(
        network=network,
        config=config,
        pixel_ports=pixel_ports,
        probe_ports=probe_ports,
        reward_ports=tuple(reward_ports),
        output_nodes=outputs.node_ids,
        class_edges=tuple(class_edges),
    )


def centered_class_rewards(
    evidence: Sequence[float],
    target: int,
    *,
    temperature: float = 1.0,
    scale: float = 1.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return softmax confidence and the centered signal ``scale * (y - p)``.

    The rewards sum to zero.  Correct, uncertain decisions produce a large
    positive signal, while confident wrong alternatives receive proportionally
    stronger negative feedback. Once the network is confidently correct, all updates
    naturally become small.
    """

    values = tuple(float(value) for value in evidence)
    if len(values) < 2 or not 0 <= target < len(values):
        raise ValueError("evidence and target do not describe a classification task")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("evidence must be finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive and finite")
    shifted = tuple(value / temperature for value in values)
    maximum = max(shifted)
    exponentials = tuple(math.exp(value - maximum) for value in shifted)
    total = sum(exponentials)
    probabilities = tuple(value / total for value in exponentials)
    rewards = tuple(
        scale * ((1.0 if index == target else 0.0) - probability)
        for index, probability in enumerate(probabilities)
    )
    return probabilities, rewards


def _image_inputs(
    classifier: MNISTClassifier,
    image: Sequence[object],
    start: float,
    end: float,
    input_scale: float,
) -> dict[str, ScalarPresentation]:
    if len(image) != classifier.config.feature_count:
        raise ValueError(
            f"image has {len(image)} features; expected {classifier.config.feature_count}"
        )
    inputs = {}
    for port, raw in zip(classifier.pixel_ports, image):
        value = float(raw) / input_scale
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("image values must lie between zero and input_scale")
        if value > 0.0:
            inputs[port] = ScalarPresentation(start, end, value)
    return inputs


def _output_evidence(
    result: SimulationResult,
    output_nodes: Sequence[int],
    start: float,
    duration: float,
    first_spike_bonus: float,
) -> tuple[tuple[float, ...], int]:
    positions = {node: index for index, node in enumerate(output_nodes)}
    counts = [0] * len(output_nodes)
    first: list[float | None] = [None] * len(output_nodes)
    for event in result.spikes:
        index = positions.get(event.node)
        if index is None:
            continue
        counts[index] += 1
        if first[index] is None:
            first[index] = event.t
    evidence = []
    for count, time in zip(counts, first):
        bonus = 0.0
        if time is not None and first_spike_bonus:
            relative = min(max((time - start) / duration, 0.0), 1.0)
            bonus = first_spike_bonus * (1.0 - relative)
        evidence.append(float(count) + bonus)
    return tuple(evidence), sum(counts)


def _metrics(samples: int, correct: int, loss: float, spikes: int):
    return ClassificationMetrics(
        samples=samples,
        correct=correct,
        mean_loss=loss / samples,
        mean_output_spikes=spikes / samples,
    )


def _layer_spike_counts(
    classifier: MNISTClassifier, result: SimulationResult
) -> tuple[tuple[str, int], ...]:
    if not classifier.activity_populations:
        return ()
    membership = {
        name: frozenset(classifier.network.node_ids(name))
        for name in classifier.activity_populations
    }
    counts = {name: 0 for name in classifier.activity_populations}
    for event in result.spikes:
        for name, nodes in membership.items():
            if event.node in nodes:
                counts[name] += 1
                break
    return tuple((name, counts[name]) for name in classifier.activity_populations)


def _default_options() -> RunOptions:
    return RunOptions(
        queue_capacity=65_536,
        output_capacity=65_536,
        encoder_spike_capacity=8_192,
        encoder_drive_capacity=4_096,
        decoder_event_capacity=4_096,
    )


def train_classifier(
    classifier: MNISTClassifier,
    images: Sequence[Sequence[object]],
    labels: Sequence[object],
    *,
    engine: Engine | None = None,
    seed: int = 0,
    input_scale: float = 255.0,
    options: RunOptions | None = None,
    progress: Callable[[ProgressUpdate], None] | None = None,
) -> TrainingResult:
    """Train online in one persistent incremental C run."""

    sample_count = len(labels)
    if sample_count == 0 or len(images) != sample_count:
        raise ValueError("images and labels must have the same nonzero length")
    if not math.isfinite(input_scale) or input_scale <= 0.0:
        raise ValueError("input_scale must be positive and finite")
    config = classifier.config
    duration = config.sample_duration * sample_count
    correct = 0
    total_loss = 0.0
    total_spikes = 0
    final_result = None
    runtime = engine or Engine()
    spike_targets: object = (
        ("outputs", *classifier.activity_populations)
        if classifier.activity_populations
        else "outputs"
    )
    recording = RecordingPlan(spikes=SpikeRecording(targets=spike_targets))

    with runtime.compile(classifier.network) as compiled:
        with compiled.start_run(
            duration,
            recording=recording,
            seed=seed,
            options=options or _default_options(),
        ) as run:
            for sample, (image, raw_target) in enumerate(zip(images, labels)):
                target = int(raw_target)
                if not 0 <= target < config.class_count:
                    raise ValueError(f"label {target} lies outside the configured classes")
                start = sample * config.sample_duration
                presentation_end = start + config.presentation_duration
                decision_end = presentation_end + config.settling_duration
                probe_time = decision_end + config.probe_delay
                reward_time = probe_time + config.reward_delay
                sample_end = start + config.sample_duration

                decision = run.advance(
                    decision_end,
                    inputs=_image_inputs(
                        classifier,
                        image,
                        start,
                        presentation_end,
                        input_scale,
                    ),
                )
                evidence, spike_count = _output_evidence(
                    decision,
                    classifier.output_nodes,
                    start,
                    config.presentation_duration + config.settling_duration,
                    config.first_spike_bonus,
                )
                probabilities, rewards = centered_class_rewards(
                    evidence,
                    target,
                    temperature=config.softmax_temperature,
                    scale=config.reward_scale,
                )
                prediction = max(range(config.class_count), key=evidence.__getitem__)
                correct += int(prediction == target)
                total_loss -= math.log(max(probabilities[target], 1.0e-300))
                total_spikes += spike_count
                layer_spikes = _layer_spike_counts(classifier, decision)

                # A common post-decision probe gives every class projection a
                # comparable causal eligibility trace.  The label is carried
                # only by y-p: correct edges potentiate and competing edges
                # depress.  Keeping the probe outside the decision window means
                # it cannot leak into the prediction.
                probe_ports = (
                    classifier.probe_ports
                    if config.probe_all_classes
                    else (classifier.probe_ports[target],)
                )
                run.advance(
                    reward_time,
                    inputs={
                        port: SpikeTrain(
                            (probe_time,), config.probe_spike_amplitude
                        )
                        for port in probe_ports
                    },
                )
                reward_inputs = {
                    port: ModulationSeries((reward_time,), reward)
                    for port, reward in zip(classifier.reward_ports, rewards)
                }
                final_result = (
                    run.finish(inputs=reward_inputs)
                    if sample + 1 == sample_count
                    else run.advance(sample_end, inputs=reward_inputs)
                )
                if progress is not None:
                    progress(
                        ProgressUpdate(
                            sample=sample + 1,
                            samples=sample_count,
                            target=target,
                            prediction=prediction,
                            running_accuracy=correct / (sample + 1),
                            running_loss=total_loss / (sample + 1),
                            output_spikes=spike_count,
                            layer_spikes=layer_spikes,
                        )
                    )

    assert final_result is not None
    learned = final_result.learned_network()
    return TrainingResult(
        classifier=replace(classifier, network=learned),
        metrics=_metrics(sample_count, correct, total_loss, total_spikes),
    )


def freeze_classifier(classifier: MNISTClassifier) -> MNISTClassifier:
    """Return an inference-only copy with learning state removed."""

    graph = classifier.network.graph
    frozen_graph = replace(
        graph,
        edges=tuple(replace(edge, plasticity=None) for edge in graph.edges),
        modulator_ports=(),
    )
    frozen_network = replace(classifier.network, graph=frozen_graph, _owner=object())
    return replace(classifier, network=frozen_network)


def evaluate_classifier(
    classifier: MNISTClassifier,
    images: Sequence[Sequence[object]],
    labels: Sequence[object],
    *,
    engine: Engine | None = None,
    seed: int = 1,
    input_scale: float = 255.0,
    options: RunOptions | None = None,
) -> ClassificationMetrics:
    """Evaluate learned weights with all plasticity disabled."""

    sample_count = len(labels)
    if sample_count == 0 or len(images) != sample_count:
        raise ValueError("images and labels must have the same nonzero length")
    frozen = freeze_classifier(classifier)
    config = frozen.config
    correct = 0
    total_loss = 0.0
    total_spikes = 0
    runtime = engine or Engine()
    recording = RecordingPlan(spikes=SpikeRecording(targets="outputs"))

    with runtime.compile(frozen.network) as compiled:
        with compiled.start_run(
            config.sample_duration * sample_count,
            recording=recording,
            seed=seed,
            options=options or _default_options(),
        ) as run:
            for sample, (image, raw_target) in enumerate(zip(images, labels)):
                target = int(raw_target)
                if not 0 <= target < config.class_count:
                    raise ValueError(f"label {target} lies outside the configured classes")
                start = sample * config.sample_duration
                presentation_end = start + config.presentation_duration
                decision_end = presentation_end + config.settling_duration
                sample_end = start + config.sample_duration
                decision = run.advance(
                    decision_end,
                    inputs=_image_inputs(
                        frozen, image, start, presentation_end, input_scale
                    ),
                )
                evidence, spike_count = _output_evidence(
                    decision,
                    frozen.output_nodes,
                    start,
                    config.presentation_duration + config.settling_duration,
                    config.first_spike_bonus,
                )
                probabilities, _ = centered_class_rewards(
                    evidence,
                    target,
                    temperature=config.softmax_temperature,
                    scale=config.reward_scale,
                )
                prediction = max(range(config.class_count), key=evidence.__getitem__)
                correct += int(prediction == target)
                total_loss -= math.log(max(probabilities[target], 1.0e-300))
                total_spikes += spike_count
                if sample + 1 == sample_count:
                    run.finish()
                else:
                    run.advance(sample_end)

    return _metrics(sample_count, correct, total_loss, total_spikes)


_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
_MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(destination: Path, expected_md5: str) -> None:
    if destination.exists() and _md5(destination) == expected_md5:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    failure = None
    for base in _MNIST_MIRRORS:
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".download",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                with urllib.request.urlopen(base + destination.name, timeout=60) as source:
                    while chunk := source.read(1024 * 1024):
                        temporary.write(chunk)
            temporary_path = Path(temporary_name)
            if _md5(temporary_path) != expected_md5:
                raise RuntimeError(f"checksum mismatch for {destination.name}")
            os.replace(temporary_path, destination)
            return
        except Exception as exc:  # try the next stable mirror
            failure = exc
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
    raise RuntimeError(f"could not download {destination.name}: {failure}")


def _read_images(path: Path):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional experiment dependency
        raise RuntimeError("MNIST loading requires NumPy") from exc
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < 16:
        raise RuntimeError(f"truncated MNIST image file: {path}")
    magic, count, rows, columns = struct.unpack(">IIII", payload[:16])
    if magic != 2051 or len(payload) != 16 + count * rows * columns:
        raise RuntimeError(f"invalid MNIST image file: {path}")
    return np.frombuffer(payload, dtype=np.uint8, offset=16).reshape(
        count, rows * columns
    )


def _read_labels(path: Path):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional experiment dependency
        raise RuntimeError("MNIST loading requires NumPy") from exc
    with gzip.open(path, "rb") as stream:
        payload = stream.read()
    if len(payload) < 8:
        raise RuntimeError(f"truncated MNIST label file: {path}")
    magic, count = struct.unpack(">II", payload[:8])
    if magic != 2049 or len(payload) != 8 + count:
        raise RuntimeError(f"invalid MNIST label file: {path}")
    return np.frombuffer(payload, dtype=np.uint8, offset=8)


def load_mnist(
    data_dir: str | os.PathLike[str] | None = None, *, download: bool = True
):
    """Load the canonical train/test split as compact uint8 NumPy arrays."""

    root = (
        Path(data_dir).expanduser()
        if data_dir is not None
        else Path.home() / ".cache" / "lacuna" / "mnist"
    )
    for name, checksum in _MNIST_FILES.items():
        path = root / name
        if download:
            _download_file(path, checksum)
        elif not path.exists() or _md5(path) != checksum:
            raise FileNotFoundError(f"missing or invalid cached MNIST file: {path}")
    train_images = _read_images(root / "train-images-idx3-ubyte.gz")
    train_labels = _read_labels(root / "train-labels-idx1-ubyte.gz")
    test_images = _read_images(root / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_labels(root / "t10k-labels-idx1-ubyte.gz")
    return train_images, train_labels, test_images, test_labels


__all__ = [
    "ClassificationMetrics",
    "MNISTClassifier",
    "MNISTConfig",
    "ProgressUpdate",
    "TrainingResult",
    "build_mnist_classifier",
    "centered_class_rewards",
    "evaluate_classifier",
    "freeze_classifier",
    "load_mnist",
    "train_classifier",
]
