#!/usr/bin/env python3
"""Online-plasticity search for a fully trainable neuronal MNIST hierarchy.

Python constructs candidates, presents images, and supplies reward modulation.
All neuron dynamics and every synaptic weight update execute in Lacuna's C
evaluator.  No gradients, offline fitted weights, image filters, or surrogate
models are used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from lacuna import (
    AdaptiveLIF,
    Convolution2D,
    CoreEvaluator,
    ExplicitConnections,
    LIF,
    ModulatedSTDP,
    ModulationInput,
    Network,
    NetworkBuilder,
    NeuronPolarity,
    PairSTDP,
    RegularRateEncoder,
    ScalarInput,
    SpikeInput,
    TripletSTDP,
    Uniform,
)
from lacuna.experiments import load_mnist

from mnist_neuron_search import balanced_indices


@dataclass(frozen=True)
class SearchConfig:
    architecture: str = "local2"
    shared_convolution: bool = False
    hidden_strategy: str = "modulated"
    adaptive_hidden: bool = False
    channels1: int = 4
    channels2: int = 8
    channels3: int = 24
    threshold1: float = -58.0
    threshold2: float = -60.0
    inhibition_weight: float = 10.0
    hidden1_low: float = 2.0
    hidden1_high: float = 4.0
    hidden1_bound: float = 8.0
    hidden2_low: float = 0.30
    hidden2_high: float = 0.90
    hidden2_bound: float = 3.0
    hidden3_low: float = 0.8
    hidden3_high: float = 1.6
    hidden3_bound: float = 4.0
    reservoir_size: int = 128
    reservoir_input_low: float = 0.10
    reservoir_input_high: float = 0.30
    reservoir_input_bound: float = 1.0
    reservoir_recurrent_degree: int = 8
    reservoir_exc_weight: float = 0.8
    reservoir_inh_weight: float = 1.6
    hidden_learning_rate: float = 0.001
    pair_a_plus: float = 0.6
    pair_a_minus: float = 0.3
    output_low: float = 0.10
    output_high: float = 0.22
    output_bound: float = 0.60
    output_learning_rate: float = 0.005
    output_threshold: float = -63.5
    trace_tau: float = 20.0
    eligibility_tau: float = 120.0
    hidden_reward_scale: float = 0.5
    hidden_reward_mode: str = "correctness"
    presentation: float = 40.0
    settling: float = 10.0
    sample_duration: float = 100.0
    max_rate: float = 0.20
    first_spike_bonus: float = 1.0
    seed: int = 41

    def __post_init__(self) -> None:
        if self.architecture not in {"local2", "local3_reservoir"}:
            raise ValueError("unknown architecture")
        if self.hidden_strategy not in {"modulated", "pair", "triplet", "frozen"}:
            raise ValueError("unknown hidden plasticity strategy")
        if self.hidden_reward_mode not in {"correctness", "advantage"}:
            raise ValueError("unknown hidden reward mode")
        if self.channels1 <= 0 or self.channels2 <= 0:
            raise ValueError("hidden channel counts must be positive")
        if self.channels3 <= 0 or self.reservoir_size < 5:
            raise ValueError("third-stage channels and reservoir size must be positive")
        if self.reservoir_recurrent_degree <= 0:
            raise ValueError("reservoir recurrent degree must be positive")
        if self.sample_duration <= self.presentation + self.settling + 5.0:
            raise ValueError("sample duration leaves no post-reward recovery")


@dataclass(frozen=True)
class Candidate:
    network: Network
    pixel_ports: tuple[str, ...]
    output_nodes: tuple[int, ...]
    pair_nodes: tuple[tuple[int, int, int, int], ...]
    probe_port: str
    output_reward_ports: tuple[str, ...]
    hidden_reward_ports: tuple[str, ...]
    hidden_node_groups: tuple[tuple[int, ...], ...]
    hidden_edge_groups: tuple[tuple[int, ...], ...]
    output_edges: tuple[int, ...]


@dataclass(frozen=True)
class Metrics:
    samples: int
    accuracy: float
    mean_output_spikes: float
    mean_hidden_spikes: tuple[float, ...]
    silent_fraction: float
    mean_loss: float


def _local_pairs(
    input_side: int,
    input_channels: int,
    output_side: int,
    output_channels: int,
    *,
    kernel: int = 2,
    stride: int = 2,
) -> tuple[tuple[int, int], ...]:
    expected = (input_side - kernel) // stride + 1
    if output_side != expected:
        raise ValueError("local target geometry is inconsistent")
    pairs = []
    for output_row in range(output_side):
        for output_column in range(output_side):
            for output_channel in range(output_channels):
                target = (
                    output_row * output_side + output_column
                ) * output_channels + output_channel
                for row_offset in range(kernel):
                    for column_offset in range(kernel):
                        input_row = output_row * stride + row_offset
                        input_column = output_column * stride + column_offset
                        for input_channel in range(input_channels):
                            source = (
                                input_row * input_side + input_column
                            ) * input_channels + input_channel
                            pairs.append((source, target))
    return tuple(pairs)


def _competition_pairs(side: int, channels: int):
    forward, feedback = [], []
    for position in range(side * side):
        for channel in range(channels):
            feature = position * channels + channel
            forward.append((feature, position))
            feedback.append((position, feature))
    return tuple(forward), tuple(feedback)


def _fixed_degree_pairs(source_count: int, target_count: int, degree: int, seed: int):
    generator = random.Random(seed)
    degree = min(degree, target_count)
    return tuple(
        (source, target)
        for source in range(source_count)
        for target in generator.sample(range(target_count), degree)
    )


def _hidden_rule(config: SearchConfig):
    if config.hidden_strategy == "frozen":
        return None
    if config.hidden_strategy == "pair":
        return PairSTDP(
            tau_pre=config.trace_tau,
            tau_post=config.trace_tau,
            a_plus=config.pair_a_plus,
            a_minus=config.pair_a_minus,
            learning_rate=config.hidden_learning_rate,
            bounds=(0.001, config.hidden1_bound),
        )
    if config.hidden_strategy == "triplet":
        return TripletSTDP.visual_cortex(
            learning_rate=config.hidden_learning_rate,
            bounds=(0.001, config.hidden1_bound),
        )
    return ModulatedSTDP(
        tau_pre=config.trace_tau,
        tau_post=config.trace_tau,
        tau_eligibility_plus=config.eligibility_tau,
        tau_eligibility_minus=config.eligibility_tau,
        learning_rate=config.hidden_learning_rate,
        bounds=(0.001, config.hidden1_bound),
        consume_on_modulation=True,
    )


def _stage_model(config: SearchConfig, *, index: int):
    threshold = config.threshold1 if index == 1 else config.threshold2
    if config.adaptive_hidden:
        return AdaptiveLIF(
            name=f"whole_stage{index}_adaptive_lif",
            tau_m=20.0,
            tau_adaptation=120.0,
            adaptation_increment=0.5,
            v_threshold=threshold,
            refractory=2.0,
        )
    return LIF(
        name=f"whole_stage{index}_lif",
        tau_m=20.0,
        v_threshold=threshold,
        refractory=2.0,
    )


def build_candidate(
    config: SearchConfig,
    *,
    plastic_stages: frozenset[str] | None = None,
) -> Candidate:
    if plastic_stages is None:
        plastic_stages = frozenset(
            ("hidden1", "hidden2", "output")
            if config.architecture == "local2"
            else ("hidden1", "hidden2", "hidden3", "reservoir", "output")
        )
    builder = NetworkBuilder(
        "mnist_whole_network_search",
        metadata={"experiment": "online_whole_network_search", "config": asdict(config)},
    )
    pixels = builder.population(
        "pixels", 784, LIF(name="whole_pixel_lif", tau_m=10.0, refractory=1.0)
    )
    pixel_ports = tuple(
        builder.inputs(
            "pixel",
            pixels,
            encoder=RegularRateEncoder(0.0, config.max_rate, amplitude=20.0),
        )
    )

    hidden_reward_ports = []
    hidden_node_groups = []
    hidden_edge_groups = []
    current = pixels
    input_side, input_channels = 28, 1
    stage_specs = [
            (14, config.channels1, config.hidden1_low, config.hidden1_high,
             config.hidden1_bound),
            (7, config.channels2, config.hidden2_low, config.hidden2_high,
             config.hidden2_bound),
    ]
    if config.architecture == "local3_reservoir":
        stage_specs.append(
            (3, config.channels3, config.hidden3_low, config.hidden3_high,
             config.hidden3_bound)
        )
    for index, (side, channels, low, high, bound) in enumerate(
        stage_specs,
        start=1,
    ):
        stage = builder.population(
            f"stage{index}", side * side * channels, _stage_model(config, index=index)
        )
        hidden_node_groups.append(stage.node_ids)
        pairs = _local_pairs(input_side, input_channels, side, channels)
        delay_rng = random.Random(config.seed + index * 104_729)
        delays = tuple(0.1 + delay_rng.uniform(0.0, 0.3) for _ in pairs)
        rule = _hidden_rule(config) if f"hidden{index}" in plastic_stages else None
        if index > 1 and rule is not None:
            rule = replace(rule, bounds=(0.001, bound))
        connection_pattern = (
            Convolution2D(
                input_shape=(input_side, input_side, input_channels),
                output_channels=channels,
                kernel_size=2,
                stride=2,
            )
            if config.shared_convolution
            else ExplicitConnections(pairs)
        )
        edges = tuple(
            builder.connect(
                current,
                stage,
                pattern=connection_pattern,
                weight=Uniform(low, high, seed=config.seed + index * 65_537),
                delay=delays,
                plasticity=rule,
            )
        )
        hidden_edge_groups.append(tuple(edges))
        if config.hidden_strategy == "modulated" and rule is not None:
            hidden_reward_ports.append(
                builder.modulator(f"hidden_reward[{index}]", targets=edges)
            )

        inhibitors = builder.population(
            f"stage{index}_inhibitors",
            side * side,
            LIF(
                name=f"whole_stage{index}_inhibitory_lif",
                tau_m=8.0,
                v_threshold=-55.0,
                refractory=1.0,
            ),
            polarity=NeuronPolarity.INHIBITORY,
        )
        forward, feedback = _competition_pairs(side, channels)
        builder.connect(
            stage,
            inhibitors,
            pattern=ExplicitConnections(forward),
            weight=12.0,
            delay=0.0,
        )
        builder.connect(
            inhibitors,
            stage,
            pattern=ExplicitConnections(feedback),
            weight=config.inhibition_weight,
            delay=0.1,
        )
        current = stage
        input_side, input_channels = side, channels

    if config.architecture == "local3_reservoir":
        excitatory_count = round(config.reservoir_size * 0.8)
        inhibitory_count = config.reservoir_size - excitatory_count
        reservoir_exc = builder.population(
            "reservoir_exc",
            excitatory_count,
            LIF(
                name="whole_reservoir_exc_lif",
                tau_m=20.0,
                v_threshold=-60.0,
                refractory=2.0,
            ),
            polarity=NeuronPolarity.EXCITATORY,
        )
        reservoir_inh = builder.population(
            "reservoir_inh",
            inhibitory_count,
            LIF(
                name="whole_reservoir_inh_lif",
                tau_m=10.0,
                v_threshold=-58.0,
                refractory=1.0,
            ),
            polarity=NeuronPolarity.INHIBITORY,
        )
        reservoir_rule = (
            _hidden_rule(config) if "reservoir" in plastic_stages else None
        )
        if reservoir_rule is not None:
            reservoir_rule = replace(
                reservoir_rule, bounds=(0.001, config.reservoir_input_bound)
            )
        reservoir_edges = []
        for target_index, target in enumerate((reservoir_exc, reservoir_inh)):
            reservoir_edges.extend(
                builder.connect(
                    current,
                    target,
                    weight=Uniform(
                        config.reservoir_input_low,
                        config.reservoir_input_high,
                        seed=config.seed + 20_000_033 + target_index,
                    ),
                    delay=0.1,
                    plasticity=reservoir_rule,
                )
            )
        hidden_edge_groups.append(tuple(reservoir_edges))
        if config.hidden_strategy == "modulated" and reservoir_rule is not None:
            hidden_reward_ports.append(
                builder.modulator("reservoir_reward", targets=reservoir_edges)
            )
        hidden_node_groups.append(
            tuple((*reservoir_exc.node_ids, *reservoir_inh.node_ids))
        )
        degree = config.reservoir_recurrent_degree
        recurrent_specs = (
            (reservoir_exc, excitatory_count, reservoir_exc, excitatory_count,
             degree, config.reservoir_exc_weight, 30_000_041),
            (reservoir_exc, excitatory_count, reservoir_inh, inhibitory_count,
             max(1, degree // 2), config.reservoir_exc_weight, 30_000_043),
            (reservoir_inh, inhibitory_count, reservoir_exc, excitatory_count,
             degree, config.reservoir_inh_weight, 30_000_047),
            (reservoir_inh, inhibitory_count, reservoir_inh, inhibitory_count,
             max(1, degree // 2), config.reservoir_inh_weight, 30_000_049),
        )
        for source, source_count, target, target_count, count, weight, seed in recurrent_specs:
            builder.connect(
                source,
                target,
                pattern=ExplicitConnections(
                    _fixed_degree_pairs(
                        source_count, target_count, count, config.seed + seed
                    )
                ),
                weight=weight,
                delay=1.0,
            )
        current = reservoir_exc

    digit_pairs = tuple(
        (first, second) for first in range(10) for second in range(first + 1, 10)
    )
    outputs = builder.population(
        "outputs",
        90,
        LIF(
            name="whole_output_lif",
            tau_m=20.0,
            v_threshold=config.output_threshold,
            refractory=2.0,
        ),
    )
    builder.outputs("digit", outputs)
    probe = builder.neuron(
        "eligibility_probe", LIF(name="whole_probe_lif", tau_m=5.0, refractory=5.0)
    )
    probe_port = builder.input("probe_trigger", probe)
    builder.connect(probe, outputs, weight=20.0, delay=0.0)

    output_rule = None
    if "output" in plastic_stages:
        output_rule = ModulatedSTDP(
            tau_pre=config.trace_tau,
            tau_post=config.trace_tau,
            tau_eligibility_plus=config.eligibility_tau,
            tau_eligibility_minus=config.eligibility_tau,
            learning_rate=config.output_learning_rate,
            bounds=(0.001, config.output_bound),
            consume_on_modulation=True,
        )
    output_reward_ports, output_edge_ids, pair_nodes = [], [], []
    for pair_index, (first, second) in enumerate(digit_pairs):
        first_node = outputs.node_ids[2 * pair_index]
        second_node = outputs.node_ids[2 * pair_index + 1]
        pair_nodes.append((first, second, first_node, second_node))
        for side_index, target in enumerate(
            (outputs[2 * pair_index], outputs[2 * pair_index + 1])
        ):
            edges = tuple(
                builder.connect(
                    current,
                    target,
                    weight=Uniform(
                        config.output_low,
                        config.output_high,
                        seed=config.seed + pair_index * 131_071 + side_index,
                    ),
                    delay=0.0,
                    plasticity=output_rule,
                )
            )
            output_edge_ids.extend(edges)
            if output_rule is not None:
                output_reward_ports.append(
                    builder.modulator(
                        f"pair_reward[{first},{second}][{side_index}]", targets=edges
                    )
                )

    return Candidate(
        network=builder.build(),
        pixel_ports=pixel_ports,
        output_nodes=outputs.node_ids,
        pair_nodes=tuple(pair_nodes),
        probe_port=probe_port,
        output_reward_ports=tuple(output_reward_ports),
        hidden_reward_ports=tuple(hidden_reward_ports),
        hidden_node_groups=tuple(hidden_node_groups),
        hidden_edge_groups=tuple(hidden_edge_groups),
        output_edges=tuple(output_edge_ids),
    )


def _inputs(candidate: Candidate, image, start: float, end: float):
    encoded_start = start + 1.0e-9
    return tuple(
        ScalarInput(encoded_start, end, port, float(value) / 255.0)
        for port, value in zip(candidate.pixel_ports, image)
        if value > 0
    )


def _evidence(spikes, candidate: Candidate, start: float, duration: float, bonus: float):
    positions = {node: index for index, node in enumerate(candidate.output_nodes)}
    counts = [0] * len(candidate.output_nodes)
    first = [None] * len(candidate.output_nodes)
    for event in spikes:
        index = positions.get(event.node)
        if index is None:
            continue
        counts[index] += 1
        if first[index] is None:
            first[index] = event.t
    individual = []
    for count, spike_time in zip(counts, first):
        latency = 0.0
        if spike_time is not None:
            phase = min(max((spike_time - start) / duration, 0.0), 1.0)
            latency = bonus * (1.0 - phase)
        individual.append(count + latency)
    by_node = dict(zip(candidate.output_nodes, individual))
    votes = [0.0] * 10
    for first_digit, second_digit, first_node, second_node in candidate.pair_nodes:
        first_value, second_value = by_node[first_node], by_node[second_node]
        if first_value > second_value:
            votes[first_digit] += 1.0
        elif second_value > first_value:
            votes[second_digit] += 1.0
        else:
            votes[first_digit] += 0.5
            votes[second_digit] += 0.5
    return tuple(votes), by_node, sum(counts)


def _pair_rewards(candidate: Candidate, by_node, label: int):
    rewards = []
    for first, second, first_node, second_node in candidate.pair_nodes:
        if label not in (first, second):
            rewards.extend((0.0, 0.0))
            continue
        values = (by_node[first_node], by_node[second_node])
        target = 0 if label == first else 1
        rival = 1 - target
        if values[target] < values[rival] + 1.0:
            rewards.extend((1.0, -1.0) if target == 0 else (-1.0, 1.0))
        else:
            rewards.extend((0.0, 0.0))
    return tuple(rewards)


def _hidden_reward(config: SearchConfig, votes, label: int):
    prediction = int(np.argmax(votes))
    if config.hidden_reward_mode == "correctness":
        value = 1.0 if prediction == label else -1.0
    else:
        rival = max(votes[digit] for digit in range(10) if digit != label)
        value = max(-1.0, min(1.0, (votes[label] - rival) / 2.0))
    return config.hidden_reward_scale * value


def _probabilities(votes):
    maximum = max(votes)
    values = [math.exp(value - maximum) for value in votes]
    total = sum(values)
    return tuple(value / total for value in values)


def _metrics(correct, loss, spikes, hidden_spikes, silent, samples):
    return Metrics(
        samples=samples,
        accuracy=correct / samples,
        mean_output_spikes=spikes / samples,
        mean_hidden_spikes=tuple(value / samples for value in hidden_spikes),
        silent_fraction=silent / samples,
        mean_loss=loss / samples,
    )


def _combine(parts):
    samples = sum(part.samples for part in parts)
    return Metrics(
        samples=samples,
        accuracy=sum(part.accuracy * part.samples for part in parts) / samples,
        mean_output_spikes=sum(
            part.mean_output_spikes * part.samples for part in parts
        ) / samples,
        mean_hidden_spikes=tuple(
            sum(part.mean_hidden_spikes[index] * part.samples for part in parts)
            / samples
            for index in range(len(parts[0].mean_hidden_spikes))
        ),
        silent_fraction=sum(part.silent_fraction * part.samples for part in parts)
        / samples,
        mean_loss=sum(part.mean_loss * part.samples for part in parts) / samples,
    )


def _with_weights(candidate: Candidate, weights):
    graph = replace(
        candidate.network.graph,
        edges=tuple(
            replace(edge, weight=float(weights[edge.id]))
            for edge in candidate.network.graph.edges
        ),
    )
    return replace(candidate, network=replace(candidate.network, graph=graph))


def train(core, candidate: Candidate, config: SearchConfig, images, labels):
    correct = silent = output_spikes = 0
    hidden_spikes = [0] * len(candidate.hidden_node_groups)
    hidden_sets = tuple(set(group) for group in candidate.hidden_node_groups)
    loss = 0.0
    final = None
    with candidate.network.graph.resolve().compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=config.sample_duration * len(labels),
            encoder_seed=config.seed,
            queue_capacity=262_144,
            output_capacity=65_536,
            encoder_spike_capacity=16_384,
        ) as run:
            for sample, (image, raw_label) in enumerate(zip(images, labels)):
                label = int(raw_label)
                start = sample * config.sample_duration
                presentation_end = start + config.presentation
                decision_end = presentation_end + config.settling
                probe_time = decision_end + 3.0
                reward_time = probe_time + 1.0
                sample_end = start + config.sample_duration
                decision = run.advance_until(
                    decision_end,
                    scalar_inputs=_inputs(candidate, image, start, presentation_end),
                )
                votes, by_node, spike_count = _evidence(
                    decision.core.spikes,
                    candidate,
                    start,
                    decision_end - start,
                    config.first_spike_bonus,
                )
                probabilities = _probabilities(votes)
                prediction = int(np.argmax(votes))
                correct += int(prediction == label)
                loss -= math.log(max(probabilities[label], 1.0e-300))
                output_spikes += spike_count
                for event in decision.core.spikes:
                    for index, nodes in enumerate(hidden_sets):
                        hidden_spikes[index] += int(event.node in nodes)
                silent += int(spike_count == 0)
                output_rewards = _pair_rewards(candidate, by_node, label)
                hidden_reward = _hidden_reward(config, votes, label)
                modulation = [
                    ModulationInput(reward_time, port, reward)
                    for port, reward in zip(
                        candidate.output_reward_ports, output_rewards
                    )
                ]
                modulation.extend(
                    ModulationInput(reward_time, port, hidden_reward)
                    for port in candidate.hidden_reward_ports
                )
                kwargs = {
                    "spike_inputs": (
                        SpikeInput(probe_time, candidate.probe_port, 20.0),
                    ),
                    "modulation_inputs": tuple(modulation),
                }
                final = (
                    run.finish(**kwargs)
                    if sample + 1 == len(labels)
                    else run.advance_until(sample_end, **kwargs)
                )
    if final is None:
        raise ValueError("training set must be nonempty")
    return final.core.weights, _metrics(
        correct, loss, output_spikes, hidden_spikes, silent, len(labels)
    )


def evaluate(core, candidate: Candidate, config: SearchConfig, weights, images, labels):
    frozen = replace(
        candidate.network.graph,
        edges=tuple(
            replace(edge, weight=float(weights[edge.id]), plasticity=None)
            for edge in candidate.network.graph.edges
        ),
        modulator_ports=(),
    )
    candidate = replace(candidate, network=replace(candidate.network, graph=frozen))
    correct = silent = output_spikes = 0
    hidden_spikes = [0] * len(candidate.hidden_node_groups)
    hidden_sets = tuple(set(group) for group in candidate.hidden_node_groups)
    loss = 0.0
    with frozen.resolve().compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=config.sample_duration * len(labels),
            encoder_seed=config.seed + 1_000_003,
            queue_capacity=262_144,
            output_capacity=65_536,
            encoder_spike_capacity=16_384,
        ) as run:
            for sample, (image, raw_label) in enumerate(zip(images, labels)):
                label = int(raw_label)
                start = sample * config.sample_duration
                end = start + config.presentation
                decision_end = end + config.settling
                sample_end = start + config.sample_duration
                decision = run.advance_until(
                    decision_end,
                    scalar_inputs=_inputs(candidate, image, start, end),
                )
                votes, _, spike_count = _evidence(
                    decision.core.spikes,
                    candidate,
                    start,
                    config.presentation + config.settling,
                    config.first_spike_bonus,
                )
                probabilities = _probabilities(votes)
                correct += int(int(np.argmax(votes)) == label)
                loss -= math.log(max(probabilities[label], 1.0e-300))
                output_spikes += spike_count
                for event in decision.core.spikes:
                    for index, nodes in enumerate(hidden_sets):
                        hidden_spikes[index] += int(event.node in nodes)
                silent += int(spike_count == 0)
                if sample + 1 == len(labels):
                    run.finish()
                else:
                    run.advance_until(sample_end)
    return _metrics(
        correct, loss, output_spikes, hidden_spikes, silent, len(labels)
    )


def run_trial(
    core,
    config,
    train_x,
    train_y,
    validation_x,
    validation_y,
    schedule,
    initial_weights=None,
):
    started = time.perf_counter()
    generator = np.random.default_rng(config.seed)
    chunks = np.array_split(generator.permutation(len(train_y)), len(schedule))
    weights = initial_weights
    train_parts = []
    candidate = None
    for phase, (factor, indices) in enumerate(zip(schedule, chunks)):
        phase_config = replace(
            config,
            hidden_learning_rate=config.hidden_learning_rate * factor,
            output_learning_rate=config.output_learning_rate * factor,
            seed=config.seed + phase * 1_000_003,
        )
        candidate = build_candidate(phase_config)
        if weights is not None:
            candidate = _with_weights(candidate, weights)
        weights, metrics = train(
            core, candidate, phase_config, train_x[indices], train_y[indices]
        )
        train_parts.append(metrics)
    assert candidate is not None and weights is not None
    evaluation_candidate = _with_weights(build_candidate(config), weights)
    validation = evaluate(
        core,
        evaluation_candidate,
        config,
        weights,
        validation_x,
        validation_y,
    )
    group_means = [
        float(np.mean([weights[edge] for edge in group]))
        for group in (*candidate.hidden_edge_groups, candidate.output_edges)
    ]
    return {
        "config": asdict(config),
        "schedule_factors": list(schedule),
        "train": asdict(_combine(train_parts)),
        "validation": asdict(validation),
        "plastic_weight_means": group_means,
        "learned_weights": [float(value) for value in weights],
        "weight_parameter_count": candidate.network.validate().weight_parameter_count,
        "shared_weight_group_count": candidate.network.validate().shared_weight_group_count,
        "seconds": time.perf_counter() - started,
    }


def run_layerwise_trial(
    core,
    config,
    train_x,
    train_y,
    validation_x,
    validation_y,
    schedule,
    initial_weights=None,
):
    """Train each projection online while all other projections are frozen."""
    started = time.perf_counter()
    weights = initial_weights
    stage_results = []
    candidate = None
    scopes = (
        ("hidden1", "hidden2", "output")
        if config.architecture == "local2"
        else ("hidden1", "hidden2", "hidden3", "reservoir", "output")
    )
    for scope_index, scope in enumerate(scopes):
        generator = np.random.default_rng(config.seed + scope_index * 10_000_019)
        chunks = np.array_split(generator.permutation(len(train_y)), len(schedule))
        parts = []
        for phase, (factor, indices) in enumerate(zip(schedule, chunks)):
            phase_config = replace(
                config,
                hidden_learning_rate=config.hidden_learning_rate * factor,
                output_learning_rate=config.output_learning_rate * factor,
                seed=(
                    config.seed
                    + scope_index * 10_000_019
                    + phase * 1_000_003
                ),
            )
            candidate = build_candidate(
                phase_config, plastic_stages=frozenset((scope,))
            )
            if weights is not None:
                candidate = _with_weights(candidate, weights)
            weights, metrics = train(
                core, candidate, phase_config, train_x[indices], train_y[indices]
            )
            parts.append(metrics)
        stage_results.append({"scope": scope, "train": asdict(_combine(parts))})

    assert candidate is not None and weights is not None
    evaluation_candidate = _with_weights(
        build_candidate(config, plastic_stages=frozenset()), weights
    )
    validation = evaluate(
        core,
        evaluation_candidate,
        config,
        weights,
        validation_x,
        validation_y,
    )
    group_means = [
        float(np.mean([weights[edge] for edge in group]))
        for group in (*candidate.hidden_edge_groups, candidate.output_edges)
    ]
    return {
        "config": asdict(config),
        "training_mode": "layerwise",
        "schedule_factors": list(schedule),
        "stages": stage_results,
        "train": stage_results[-1]["train"],
        "validation": asdict(validation),
        "plastic_weight_means": group_means,
        "learned_weights": [float(value) for value in weights],
        "seconds": time.perf_counter() - started,
    }


def candidates(seed: int):
    base = SearchConfig(seed=seed)
    result = [
        base,
        replace(base, hidden_strategy="pair", seed=seed + 1),
        replace(base, hidden_strategy="triplet", seed=seed + 2),
        replace(base, adaptive_hidden=True, seed=seed + 3),
        replace(base, channels1=8, channels2=12, seed=seed + 4),
        replace(
            base,
            hidden_reward_mode="advantage",
            hidden_reward_scale=1.0,
            seed=seed + 5,
        ),
    ]
    rng = random.Random(seed)
    while len(result) < 24:
        result.append(
            replace(
                base,
                hidden_strategy=rng.choice(("modulated", "pair", "triplet")),
                adaptive_hidden=rng.choice((False, True)),
                channels1=rng.choice((4, 6, 8)),
                channels2=rng.choice((8, 12, 16)),
                threshold1=rng.choice((-59.0, -58.0, -57.0)),
                threshold2=rng.choice((-61.0, -60.0, -59.0)),
                inhibition_weight=rng.choice((6.0, 10.0, 14.0)),
                hidden_learning_rate=rng.choice((0.0005, 0.001, 0.002)),
                output_learning_rate=rng.choice((0.003, 0.005, 0.008)),
                output_threshold=rng.choice((-64.0, -63.5, -63.0)),
                output_low=rng.choice((0.08, 0.10, 0.12)),
                output_high=rng.choice((0.18, 0.22, 0.26)),
                output_bound=rng.choice((0.50, 0.60, 0.75)),
                hidden_reward_scale=rng.choice((0.25, 0.5, 1.0)),
                hidden_reward_mode=rng.choice(("correctness", "advantage")),
                trace_tau=rng.choice((10.0, 20.0, 30.0)),
                eligibility_tau=rng.choice((80.0, 120.0, 200.0)),
                seed=seed + len(result),
            )
        )
    return result


def neighborhood(base: SearchConfig):
    """Controlled one-factor refinements around a promoted configuration."""
    return [
        base,
        replace(base, pair_a_minus=0.45, seed=base.seed + 101),
        replace(base, pair_a_minus=0.60, seed=base.seed + 102),
        replace(base, pair_a_plus=0.40, pair_a_minus=0.40, seed=base.seed + 103),
        replace(base, hidden_learning_rate=0.001, seed=base.seed + 104),
        replace(base, hidden_learning_rate=0.003, seed=base.seed + 105),
        replace(base, output_learning_rate=0.005, seed=base.seed + 106),
        replace(base, output_learning_rate=0.012, seed=base.seed + 107),
        replace(base, threshold2=-60.0, seed=base.seed + 108),
        replace(base, threshold2=-62.0, seed=base.seed + 109),
        replace(base, inhibition_weight=14.0, seed=base.seed + 110),
        replace(base, channels2=12, seed=base.seed + 111),
        replace(base, channels2=20, seed=base.seed + 112),
        replace(base, output_threshold=-62.5, seed=base.seed + 113),
    ]


def hierarchy_neighborhood(base: SearchConfig):
    """Activity and capacity screen for the three-stage reservoir hierarchy."""
    base = replace(
        base,
        architecture="local3_reservoir",
        output_low=0.20,
        output_high=0.40,
        output_bound=0.80,
        output_threshold=-63.0,
    )
    return [
        base,
        replace(base, hidden3_low=0.5, hidden3_high=1.2, seed=base.seed + 201),
        replace(base, hidden3_low=1.2, hidden3_high=2.0, seed=base.seed + 202),
        replace(
            base,
            reservoir_input_low=0.20,
            reservoir_input_high=0.40,
            seed=base.seed + 203,
        ),
        replace(base, output_low=0.30, output_high=0.50, seed=base.seed + 204),
        replace(
            base,
            reservoir_exc_weight=1.2,
            reservoir_inh_weight=2.4,
            seed=base.seed + 205,
        ),
        replace(base, reservoir_size=192, seed=base.seed + 206),
    ]


def convolution_neighborhood(base: SearchConfig):
    """Learning-rate screen for spatially shared Pair-STDP kernels."""
    base = replace(
        base,
        architecture="local2",
        shared_convolution=True,
        hidden_strategy="pair",
        adaptive_hidden=False,
    )
    return [
        replace(base, hidden_learning_rate=0.0005),
        replace(base, hidden_learning_rate=0.002),
        replace(base, hidden_learning_rate=0.01),
        replace(base, hidden_learning_rate=0.02),
        replace(base, hidden_learning_rate=0.05),
        replace(base, hidden_learning_rate=0.10),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build/liblacuna_core.dylib")
    parser.add_argument("--data-dir", default="artifacts/mnist")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--validation-per-class", type=int, default=20)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--validation-offset", type=int)
    parser.add_argument(
        "--strategy",
        choices=("all", "modulated", "pair", "triplet", "frozen"),
        default="all",
    )
    parser.add_argument("--schedule", type=float, nargs="+", default=(1.0, 0.5, 0.25))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--configs-from",
        nargs="+",
        help="rerun configurations, in ranked order, from prior search artifacts",
    )
    parser.add_argument(
        "--candidate-indices",
        type=int,
        nargs="+",
        help="one-based candidate indices to run after loading/generating candidates",
    )
    parser.add_argument(
        "--refine-neighborhood",
        action="store_true",
        help="replace the first loaded/generated candidate with controlled refinements",
    )
    parser.add_argument(
        "--resume-from",
        help="continue online learning from the top trial's persisted edge weights",
    )
    parser.add_argument(
        "--freeze-hidden",
        action="store_true",
        help="freeze previously learned hidden projections during online readout consolidation",
    )
    parser.add_argument(
        "--layerwise",
        action="store_true",
        help="train hidden stage 1, hidden stage 2, and output in separate online passes",
    )
    parser.add_argument(
        "--true-hierarchy",
        action="store_true",
        help="use three learned local stages followed by a recurrent E/I reservoir",
    )
    parser.add_argument(
        "--refine-hierarchy",
        action="store_true",
        help="screen gain and capacity controls around the three-stage reservoir hierarchy",
    )
    parser.add_argument(
        "--shared-convolution",
        action="store_true",
        help="replace local feedforward projections with spatially shared kernels",
    )
    parser.add_argument(
        "--refine-convolution",
        action="store_true",
        help="screen Pair-STDP learning rates for shared convolutional kernels",
    )
    parser.add_argument(
        "--output", default="artifacts/optimization/mnist_whole_network_search.json"
    )
    args = parser.parse_args()
    if args.trials <= 0 or args.train_per_class <= 0 or args.validation_per_class <= 0:
        raise ValueError("trial and sample counts must be positive")
    if any(factor < 0.0 for factor in args.schedule):
        raise ValueError("schedule factors must be nonnegative")

    train_images, train_labels, _, _ = load_mnist(args.data_dir, download=False)
    validation_offset = (
        args.train_offset + args.train_per_class
        if args.validation_offset is None
        else args.validation_offset
    )
    train_indices = balanced_indices(
        train_labels,
        args.train_per_class,
        offset=args.train_offset,
        seed=args.seed,
    )
    validation_indices = balanced_indices(
        train_labels,
        args.validation_per_class,
        offset=validation_offset,
        seed=args.seed + 1,
    )
    if args.configs_from:
        configs = []
        for source in args.configs_from:
            prior = json.loads(Path(source).read_text(encoding="utf-8"))
            configs.extend(SearchConfig(**item["config"]) for item in prior)
    else:
        configs = candidates(args.seed)
    if args.refine_neighborhood:
        if not configs:
            raise ValueError("a base candidate is required for neighborhood refinement")
        configs = neighborhood(configs[0])
    if args.freeze_hidden:
        configs = [replace(config, hidden_strategy="frozen") for config in configs]
    if args.true_hierarchy:
        configs = [replace(config, architecture="local3_reservoir") for config in configs]
    if args.refine_hierarchy:
        if not configs:
            raise ValueError("a base candidate is required for hierarchy refinement")
        configs = hierarchy_neighborhood(configs[0])
    if args.shared_convolution:
        configs = [replace(config, shared_convolution=True) for config in configs]
    if args.refine_convolution:
        if not configs:
            raise ValueError("a base candidate is required for convolution refinement")
        configs = convolution_neighborhood(configs[0])
    if args.strategy != "all":
        configs = [c for c in configs if c.hidden_strategy == args.strategy]
    if args.candidate_indices:
        invalid = [index for index in args.candidate_indices if not 1 <= index <= len(configs)]
        if invalid:
            raise ValueError(f"candidate indices out of range: {invalid}")
        configs = [configs[index - 1] for index in args.candidate_indices]
    configs = configs[: args.trials]
    if not configs:
        raise ValueError("strategy filter produced no candidates")

    core = CoreEvaluator(args.library)
    initial_weights = None
    if args.resume_from:
        if len(configs) != 1:
            raise ValueError("resumed training requires exactly one candidate")
        resumed = json.loads(Path(args.resume_from).read_text(encoding="utf-8"))[0]
        initial_weights = tuple(float(value) for value in resumed["learned_weights"])
    results = []
    for index, config in enumerate(configs, start=1):
        print(
            f"trial {index}/{len(configs)} strategy={config.hidden_strategy} "
            f"channels={config.channels1},{config.channels2} "
            f"adaptive={config.adaptive_hidden}",
            flush=True,
        )
        trial_runner = run_layerwise_trial if args.layerwise else run_trial
        result = trial_runner(
            core,
            config,
            train_images[train_indices],
            train_labels[train_indices],
            train_images[validation_indices],
            train_labels[validation_indices],
            tuple(args.schedule),
            initial_weights=initial_weights,
        )
        results.append(result)
        validation = result["validation"]
        print(
            f"  validation={100 * validation['accuracy']:.2f}% "
            f"spikes={validation['mean_output_spikes']:.2f} "
            f"hidden={validation['mean_hidden_spikes']} "
            f"silent={100 * validation['silent_fraction']:.1f}% "
            f"weights={result['plastic_weight_means']} "
            f"seconds={result['seconds']:.1f}",
            flush=True,
        )
    results.sort(key=lambda value: value["validation"]["accuracy"], reverse=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ranking:", flush=True)
    for rank, result in enumerate(results, start=1):
        config = result["config"]
        print(
            f"  {rank:2d}. {100 * result['validation']['accuracy']:6.2f}% "
            f"{config['hidden_strategy']} "
            f"channels={config['channels1']},{config['channels2']} "
            f"adaptive={config['adaptive_hidden']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
