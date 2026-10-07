"""Locally connected competitive hierarchy and reservoir MNIST experiment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from ..codec import NativeEventEncoder, PoissonRateEncoder
from ..graph import NeuronPolarity
from ..models import LIF
from ..network import (
    ExplicitConnections,
    FixedProbability,
    NetworkBuilder,
    Uniform,
)
from ..plasticity import ModulatedSTDP, PairSTDP
from ..recording import RecordingPlan, RunOptions, SpikeRecording
from ..simulation import Engine
from ..synapses import Delta
from .mnist_rstdp import (
    MNISTClassifier,
    MNISTConfig,
    _default_options,
    _image_inputs,
)


@dataclass(frozen=True)
class SpatialShape:
    """Rows, columns, and channels of one flattened spatial layer."""

    rows: int
    columns: int
    channels: int

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.columns <= 0 or self.channels <= 0:
            raise ValueError("spatial dimensions and channels must be positive")

    @property
    def size(self) -> int:
        """Return the flattened number of units in this spatial field."""

        return self.rows * self.columns * self.channels

    def index(self, row: int, column: int, channel: int) -> int:
        """Return the flattened row-major index of one unit."""

        if not (
            0 <= row < self.rows
            and 0 <= column < self.columns
            and 0 <= channel < self.channels
        ):
            raise IndexError("spatial coordinate lies outside the layer")
        return (row * self.columns + column) * self.channels + channel


@dataclass(frozen=True)
class HierarchicalMNISTConfig(MNISTConfig):
    """Four local competitive stages followed by a balanced reservoir."""

    presentation_duration: float = 40.0
    settling_duration: float = 10.0
    probe_delay: float = 3.0
    reward_delay: float = 1.0
    sample_duration: float = 120.0
    weight_low: float = 0.06
    weight_high: float = 0.16
    weight_bounds: tuple[float, float] = (0.001, 0.40)
    learning_rate: float = 0.002
    output_threshold: float = -63.5
    input_rows: int = 28
    input_columns: int = 28
    hidden_channels: tuple[int, ...] = (4, 8, 12)
    kernels: tuple[int, ...] = (5, 3, 3)
    strides: tuple[int, ...] = (2, 2, 1)
    hidden_weight_ranges: tuple[tuple[float, float], ...] = (
        (0.50, 0.80),
        (0.75, 1.05),
        (0.75, 1.05),
    )
    hidden_thresholds: tuple[float, ...] = (-55.0, -60.0, -60.0)
    hidden_learning_rate: float = 0.001
    hidden_weight_bounds: tuple[float, float] = (0.001, 1.5)
    feedforward_delay: float = 0.2
    channel_delay_jitter: float = 0.4
    inhibition_delay: float = 0.0
    inhibition_trigger_weight: float = 12.0
    inhibition_weight: float = 12.0
    reservoir_excitatory: int = 48
    reservoir_inhibitory: int = 16
    reservoir_probability: float = 0.08
    reservoir_input_probability: float = 0.06
    reservoir_threshold: float = -63.0
    reservoir_input_range: tuple[float, float] = (2.2, 3.0)
    reservoir_input_delay_jitter: float = 1.0
    reservoir_excitatory_weight: float = 0.4
    reservoir_inhibitory_weight: float = 0.8
    reservoir_delay: float = 1.0
    readout_inhibitory: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.input_rows * self.input_columns != self.feature_count:
            raise ValueError("input_rows * input_columns must equal feature_count")
        count = len(self.hidden_channels)
        if count not in (3, 4):
            raise ValueError("the hierarchy requires three or four hidden stages")
        if not (
            len(self.kernels)
            == len(self.strides)
            == len(self.hidden_weight_ranges)
            == len(self.hidden_thresholds)
            == count
        ):
            raise ValueError("hidden stage parameter tuples must have equal lengths")
        if any(value <= 0 for value in (*self.hidden_channels, *self.kernels, *self.strides)):
            raise ValueError("hidden channels, kernels, and strides must be positive")
        if any(not math.isfinite(value) or value <= -65.0 for value in self.hidden_thresholds):
            raise ValueError("hidden thresholds must be finite and exceed -65")
        if not math.isfinite(self.hidden_learning_rate) or self.hidden_learning_rate <= 0:
            raise ValueError("hidden_learning_rate must be positive and finite")
        for name in (
            "feedforward_delay",
            "channel_delay_jitter",
            "inhibition_delay",
            "inhibition_trigger_weight",
            "inhibition_weight",
            "reservoir_excitatory_weight",
            "reservoir_inhibitory_weight",
            "reservoir_delay",
            "reservoir_input_delay_jitter",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.inhibition_trigger_weight == 0.0 or self.inhibition_weight == 0.0:
            raise ValueError("lateral inhibition weights must be positive")
        if self.reservoir_excitatory <= 0 or self.reservoir_inhibitory <= 0:
            raise ValueError("both reservoir populations must be nonempty")
        if not math.isfinite(self.reservoir_threshold) or self.reservoir_threshold <= -65.0:
            raise ValueError("reservoir_threshold must be finite and exceed -65")
        if not 0.0 <= self.reservoir_probability <= 1.0:
            raise ValueError("reservoir_probability must lie in [0, 1]")
        if not 0.0 < self.reservoir_input_probability <= 1.0:
            raise ValueError("reservoir_input_probability must lie in (0, 1]")
        if not isinstance(self.readout_inhibitory, bool):
            raise ValueError("readout_inhibitory must be boolean")
        hierarchy_shapes(self)


@dataclass(frozen=True)
class HierarchicalMNISTClassifier(MNISTClassifier):
    """MNIST classifier with local hidden stages and a reservoir readout."""

    stage_shapes: tuple[SpatialShape, ...] = ()
    stage_edge_ids: tuple[tuple[int, ...], ...] = ()
    inhibition_edge_ids: tuple[tuple[int, ...], ...] = ()
    reservoir_edge_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class HiddenPretrainingProgress:
    """Epoch summary for unsupervised hidden-layer pretraining."""

    stage: int
    stages: int
    sample: int
    samples: int
    spikes: int


def hierarchy_shapes(config: HierarchicalMNISTConfig) -> tuple[SpatialShape, ...]:
    """Derive every spatial stage shape from the experiment configuration."""

    shapes = [SpatialShape(config.input_rows, config.input_columns, 1)]
    for channels, kernel, stride in zip(
        config.hidden_channels, config.kernels, config.strides
    ):
        previous = shapes[-1]
        rows = (previous.rows - kernel) // stride + 1
        columns = (previous.columns - kernel) // stride + 1
        if rows <= 0 or columns <= 0:
            raise ValueError("a local receptive field is larger than its input stage")
        shapes.append(SpatialShape(rows, columns, channels))
    return tuple(shapes)


def local_many_to_one_pairs(
    source: SpatialShape,
    target: SpatialShape,
    *,
    kernel: int,
    stride: int,
) -> tuple[tuple[int, int], ...]:
    """Connect every target channel to its complete local source volume."""

    expected_rows = (source.rows - kernel) // stride + 1
    expected_columns = (source.columns - kernel) // stride + 1
    if (
        kernel <= 0
        or stride <= 0
        or target.rows != expected_rows
        or target.columns != expected_columns
    ):
        raise ValueError("target geometry does not match kernel and stride")
    pairs = []
    for target_row in range(target.rows):
        for target_column in range(target.columns):
            for target_channel in range(target.channels):
                target_index = target.index(
                    target_row, target_column, target_channel
                )
                source_row = target_row * stride
                source_column = target_column * stride
                for row_offset in range(kernel):
                    for column_offset in range(kernel):
                        for source_channel in range(source.channels):
                            pairs.append(
                                (
                                    source.index(
                                        source_row + row_offset,
                                        source_column + column_offset,
                                        source_channel,
                                    ),
                                    target_index,
                                )
                            )
    return tuple(pairs)


def _lateral_pairs(shape: SpatialShape):
    feature_to_inhibitor = []
    inhibitor_to_feature = []
    for row in range(shape.rows):
        for column in range(shape.columns):
            inhibitor = row * shape.columns + column
            for channel in range(shape.channels):
                feature = shape.index(row, column, channel)
                feature_to_inhibitor.append((feature, inhibitor))
                inhibitor_to_feature.append((inhibitor, feature))
    return tuple(feature_to_inhibitor), tuple(inhibitor_to_feature)


def _readout_rule(config: HierarchicalMNISTConfig, *, inhibitory: bool):
    rule = ModulatedSTDP(
        tau_pre=config.trace_tau,
        tau_post=config.trace_tau,
        tau_eligibility_plus=config.eligibility_tau,
        tau_eligibility_minus=config.eligibility_tau,
        learning_rate=config.learning_rate,
        bounds=config.weight_bounds,
        consume_on_modulation=True,
    )
    if not inhibitory:
        return rule
    # Magnitude changes on Dale-inhibitory edges must be inverted: positive
    # evidence weakens inhibition of the correct output, while negative evidence
    # strengthens inhibition of a competing output.
    return replace(
        rule,
        positive_plus=-rule.positive_plus,
        positive_minus=-rule.positive_minus,
        negative_plus=-rule.negative_plus,
        negative_minus=-rule.negative_minus,
    )


def build_hierarchical_mnist_classifier(
    config: HierarchicalMNISTConfig = HierarchicalMNISTConfig(), *, seed: int = 0
) -> HierarchicalMNISTClassifier:
    """Construct the full local hierarchy, reservoir, and readout network."""

    shapes = hierarchy_shapes(config)
    builder = NetworkBuilder(
        "hierarchical_mnist_modulated_stdp",
        metadata={
            "benchmark": "MNIST",
            "architecture": "local_competitive_hierarchy_reservoir_rstdp",
            "stage_shapes": [
                [shape.rows, shape.columns, shape.channels] for shape in shapes
            ],
            "reward": "centered_class_advantage",
        },
    )
    pixels = builder.population(
        "pixels",
        config.feature_count,
        LIF(name="hierarchy_pixel_lif", tau_m=10.0, refractory=1.0),
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
    hidden_rule = PairSTDP(
        tau_pre=config.trace_tau,
        tau_post=config.trace_tau,
        a_plus=0.5,
        a_minus=0.5,
        learning_rate=config.hidden_learning_rate,
        bounds=config.hidden_weight_bounds,
    )
    current = pixels
    stage_edge_ids = []
    inhibition_edge_ids = []
    activity_populations = []
    for index, (
        source_shape,
        target_shape,
        kernel,
        stride,
        weight_range,
        threshold,
    ) in enumerate(
        zip(
            shapes[:-1],
            shapes[1:],
            config.kernels,
            config.strides,
            config.hidden_weight_ranges,
            config.hidden_thresholds,
        ),
        start=1,
    ):
        stage_name = f"stage{index}"
        stage = builder.population(
            stage_name,
            target_shape.size,
            LIF(
                name=f"hierarchy_stage{index}_lif",
                tau_m=20.0,
                v_threshold=threshold,
                refractory=2.0,
            ),
        )
        pairs = local_many_to_one_pairs(
            source_shape, target_shape, kernel=kernel, stride=stride
        )
        delay_generator = random.Random(seed + index * 20_011)
        edge_delays = tuple(
            config.feedforward_delay
            + delay_generator.uniform(0.0, config.channel_delay_jitter)
            for _ in pairs
        )
        stage_edges = builder.connect(
            current,
            stage,
            synapse=Delta(),
            pattern=ExplicitConnections(pairs),
            weight=Uniform(*weight_range, seed=seed + index * 10_007),
            delay=edge_delays,
            plasticity=hidden_rule,
        )
        inhibitor_name = f"stage{index}_inhibition"
        inhibitors = builder.population(
            inhibitor_name,
            target_shape.rows * target_shape.columns,
            LIF(
                name=f"hierarchy_stage{index}_inhibitory_lif",
                tau_m=10.0,
                v_threshold=-55.0,
                refractory=1.0,
            ),
            polarity=NeuronPolarity.INHIBITORY,
        )
        forward_pairs, feedback_pairs = _lateral_pairs(target_shape)
        trigger_edges = builder.connect(
            stage,
            inhibitors,
            synapse=Delta(),
            pattern=ExplicitConnections(forward_pairs),
            weight=config.inhibition_trigger_weight,
            delay=0.0,
        )
        feedback_edges = builder.connect(
            inhibitors,
            stage,
            synapse=Delta(),
            pattern=ExplicitConnections(feedback_pairs),
            weight=config.inhibition_weight,
            delay=config.inhibition_delay,
        )
        stage_edge_ids.append(tuple(stage_edges))
        inhibition_edge_ids.append(tuple((*trigger_edges, *feedback_edges)))
        activity_populations.extend((stage_name, inhibitor_name))
        current = stage

    reservoir_exc = builder.population(
        "reservoir_exc",
        config.reservoir_excitatory,
        LIF(
            name="hierarchy_reservoir_exc_lif",
            v_threshold=config.reservoir_threshold,
        ),
    )
    reservoir_inh = builder.population(
        "reservoir_inh",
        config.reservoir_inhibitory,
        LIF(
            name="hierarchy_reservoir_inh_lif",
            v_threshold=config.reservoir_threshold,
        ),
        polarity=NeuronPolarity.INHIBITORY,
    )
    reservoir_edges = []
    for offset, target in enumerate((reservoir_exc, reservoir_inh)):
        reservoir_edges.extend(
            builder.connect(
                current,
                target,
                synapse=Delta(),
                pattern=FixedProbability(
                    config.reservoir_input_probability,
                    seed=seed + 175_003 + offset,
                ),
                weight=Uniform(
                    *config.reservoir_input_range,
                    seed=seed + 200_003 + offset,
                ),
                delay=Uniform(
                    config.feedforward_delay,
                    config.feedforward_delay + config.reservoir_input_delay_jitter,
                    seed=seed + 250_007 + offset,
                ),
            )
        )
    recurrent_specs = (
        (reservoir_exc, reservoir_exc, config.reservoir_excitatory_weight, True),
        (reservoir_exc, reservoir_inh, config.reservoir_excitatory_weight, False),
        (reservoir_inh, reservoir_exc, config.reservoir_inhibitory_weight, False),
        (reservoir_inh, reservoir_inh, config.reservoir_inhibitory_weight, True),
    )
    for offset, (source, target, weight, exclude_self) in enumerate(recurrent_specs):
        reservoir_edges.extend(
            builder.connect(
                source,
                target,
                synapse=Delta(),
                pattern=FixedProbability(
                    config.reservoir_probability,
                    seed=seed + 300_007 + offset,
                    exclude_self=exclude_self,
                ),
                weight=weight,
                delay=Uniform(
                    max(0.0, config.reservoir_delay * 0.5),
                    config.reservoir_delay * 1.5,
                    seed=seed + 350_011 + offset,
                ),
            )
        )
    activity_populations.extend(("reservoir_exc", "reservoir_inh"))

    outputs = builder.population(
        "outputs",
        config.class_count,
        LIF(
            name="hierarchy_output_lif",
            tau_m=20.0,
            v_threshold=config.output_threshold,
            refractory=2.0,
        ),
    )
    probe_ports = tuple(
        builder.inputs("eligibility_probe", outputs, encoder=NativeEventEncoder())
    )
    builder.outputs("digit", outputs)
    excitatory_rule = _readout_rule(config, inhibitory=False)
    inhibitory_rule = _readout_rule(config, inhibitory=True)
    reward_ports = []
    class_edges = []
    inhibitory_low = max(config.weight_bounds[0], config.weight_low * 0.5)
    inhibitory_high = config.weight_high * 0.5
    for digit in range(config.class_count):
        excitatory_edges = builder.connect(
            reservoir_exc,
            outputs[digit],
            synapse=Delta(),
            weight=Uniform(
                config.weight_low,
                config.weight_high,
                seed=seed + 400_009 + digit,
            ),
            delay=0.0,
            plasticity=excitatory_rule,
        )
        inhibitory_edges = (
            builder.connect(
                reservoir_inh,
                outputs[digit],
                synapse=Delta(),
                weight=Uniform(
                    inhibitory_low,
                    inhibitory_high,
                    seed=seed + 500_009 + digit,
                ),
                delay=0.0,
                plasticity=inhibitory_rule,
            )
            if config.readout_inhibitory
            else ()
        )
        edges = tuple((*excitatory_edges, *inhibitory_edges))
        class_edges.append(edges)
        reward_ports.append(builder.modulator(f"reward[{digit}]", targets=edges))

    network = builder.build()
    return HierarchicalMNISTClassifier(
        network=network,
        config=config,
        pixel_ports=pixel_ports,
        probe_ports=probe_ports,
        reward_ports=tuple(reward_ports),
        output_nodes=outputs.node_ids,
        class_edges=tuple(class_edges),
        activity_populations=tuple(activity_populations),
        stage_shapes=shapes[1:],
        stage_edge_ids=tuple(stage_edge_ids),
        inhibition_edge_ids=tuple(inhibition_edge_ids),
        reservoir_edge_ids=tuple(reservoir_edges),
    )


def _transfer_weights(template, learned):
    learned_edges = {edge.id: edge for edge in learned.graph.edges}
    graph = replace(
        template.graph,
        edges=tuple(
            replace(edge, weight=learned_edges[edge.id].weight)
            for edge in template.graph.edges
        ),
    )
    return replace(template, graph=graph, _owner=object())


def pretrain_hidden_stages(
    classifier: HierarchicalMNISTClassifier,
    images: Sequence[Sequence[object]],
    *,
    engine: Engine | None = None,
    seed: int = 0,
    input_scale: float = 255.0,
    stage_indices: Sequence[int] | None = None,
    options: RunOptions | None = None,
    progress: Callable[[HiddenPretrainingProgress], None] | None = None,
) -> HierarchicalMNISTClassifier:
    """Train each local Pair-STDP projection in a separate frozen-input phase."""

    if len(images) == 0:
        raise ValueError("hidden pretraining requires at least one image")
    stages = (
        tuple(range(len(classifier.stage_edge_ids)))
        if stage_indices is None
        else tuple(stage_indices)
    )
    if len(stages) != len(set(stages)) or any(
        not isinstance(stage, int)
        or isinstance(stage, bool)
        or not 0 <= stage < len(classifier.stage_edge_ids)
        for stage in stages
    ):
        raise ValueError("stage_indices contain an invalid or repeated stage")
    current = classifier
    runtime = engine or Engine()
    limits = options or _default_options()
    for phase, stage in enumerate(stages):
        active = frozenset(current.stage_edge_ids[stage])
        phase_graph = replace(
            current.network.graph,
            edges=tuple(
                replace(
                    edge,
                    plasticity=edge.plasticity if edge.id in active else None,
                )
                for edge in current.network.graph.edges
            ),
            modulator_ports=(),
        )
        phase_network = replace(
            current.network,
            graph=phase_graph,
            _owner=object(),
        )
        phase_classifier = replace(current, network=phase_network)
        duration = current.config.sample_duration * len(images)
        final_result = None
        with runtime.compile(phase_network) as compiled:
            with compiled.start_run(
                duration,
                recording=RecordingPlan(
                    spikes=SpikeRecording(targets=f"stage{stage + 1}")
                ),
                seed=seed + phase,
                options=limits,
            ) as run:
                for sample, image in enumerate(images):
                    start = sample * current.config.sample_duration
                    presentation_end = start + current.config.presentation_duration
                    decision_end = presentation_end + current.config.settling_duration
                    decision = run.advance(
                        decision_end,
                        inputs=_image_inputs(
                            phase_classifier,
                            image,
                            start,
                            presentation_end,
                            input_scale,
                        ),
                    )
                    final_result = (
                        run.finish()
                        if sample + 1 == len(images)
                        else run.advance(start + current.config.sample_duration)
                    )
                    if progress is not None:
                        progress(
                            HiddenPretrainingProgress(
                                stage=stage + 1,
                                stages=len(current.stage_edge_ids),
                                sample=sample + 1,
                                samples=len(images),
                                spikes=len(decision.spikes),
                            )
                        )
        assert final_result is not None
        learned_phase = final_result.learned_network()
        current = replace(
            current,
            network=_transfer_weights(current.network, learned_phase),
        )
    return current


def freeze_hidden_plasticity(
    classifier: HierarchicalMNISTClassifier,
) -> HierarchicalMNISTClassifier:
    """Freeze local feature weights while retaining modulated readout learning."""

    graph = classifier.network.graph
    frozen_graph = replace(
        graph,
        edges=tuple(
            replace(edge, plasticity=None)
            if isinstance(edge.plasticity, PairSTDP)
            else edge
            for edge in graph.edges
        ),
    )
    return replace(
        classifier,
        network=replace(
            classifier.network,
            graph=frozen_graph,
            _owner=object(),
        ),
    )


__all__ = [
    "HierarchicalMNISTClassifier",
    "HierarchicalMNISTConfig",
    "HiddenPretrainingProgress",
    "SpatialShape",
    "build_hierarchical_mnist_classifier",
    "hierarchy_shapes",
    "local_many_to_one_pairs",
    "pretrain_hidden_stages",
    "freeze_hidden_plasticity",
]
