#!/usr/bin/env python3
"""Experiment-only whole-network Pair-STDP/R-STDP alternating spiking CNN."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from lacuna import (
    Convolution2D,
    CoreEvaluator,
    ExplicitConnections,
    LIF,
    ModulatedSTDP,
    NetworkBuilder,
    NeuronPolarity,
    OneToOne,
    PairSTDP,
    Uniform,
)
from lacuna.experiments import load_mnist

from optimization.mnist_neuron_search import (
    CandidateConfig,
    CandidateNetwork,
    Metrics,
    _combine_metrics,
    _encoder,
    _presentations,
    _project_kernel_sum,
    balanced_indices,
    candidate_with_weights,
    evaluate_candidate,
    train_candidate,
)


@dataclass(frozen=True)
class AlternatingCNNConfig(CandidateConfig):
    plasticity_mode: str = "pair"
    cnn_channels: tuple[int, ...] = (16, 32, 48, 64)
    cnn_kernels: tuple[int, ...] = (4, 3, 3, 3)
    cnn_strides: tuple[int, ...] = (2, 2, 1, 1)
    cnn_paddings: tuple[int, ...] = (0, 0, 1, 0)
    cnn_thresholds: tuple[float, ...] = (-60.0, -60.0, -59.0, -58.0)
    pair_conv_learning_rate: float = 0.001
    pair_readout_learning_rate: float = 0.0005
    reward_conv_learning_rate: float = 0.05
    reward_readout_learning_rate: float = 0.04
    pair_trace_tau: float = 5.0
    convolution_kernel_sum_per_input_channel: float = 4.0
    convolution_inhibition: float = 10.0
    convolution_inhibition_trigger: float = 12.0
    record_stage_spikes: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.plasticity_mode not in {"pair", "modulated"}:
            raise ValueError("plasticity_mode must be pair or modulated")
        lengths = {
            len(self.cnn_channels),
            len(self.cnn_kernels),
            len(self.cnn_strides),
            len(self.cnn_paddings),
            len(self.cnn_thresholds),
        }
        if lengths != {len(self.cnn_channels)} or not self.cnn_channels:
            raise ValueError("CNN stage specifications must have equal nonzero lengths")
        if any(value <= 0 for value in self.cnn_channels):
            raise ValueError("CNN channel counts must be positive")
        if any(value <= 0 for value in self.cnn_kernels + self.cnn_strides):
            raise ValueError("CNN kernels and strides must be positive")
        if any(value < 0 for value in self.cnn_paddings):
            raise ValueError("CNN paddings cannot be negative")
        if min(
            self.pair_conv_learning_rate,
            self.pair_readout_learning_rate,
            self.reward_conv_learning_rate,
            self.reward_readout_learning_rate,
            self.pair_trace_tau,
            self.convolution_kernel_sum_per_input_channel,
        ) <= 0.0:
            raise ValueError("alternating learning parameters must be positive")


@dataclass(frozen=True)
class AlternatingCNNNetwork(CandidateNetwork):
    stage_edges: tuple[tuple[int, ...], ...]
    stage_shapes: tuple[tuple[int, int, int], ...]
    stage_nodes: tuple[tuple[int, ...], ...]
    feedforward_edges: tuple[int, ...]


def default_config(seed: int = 23) -> AlternatingCNNConfig:
    return AlternatingCNNConfig(
        architecture="pool2",
        encoder="regular",
        presentation=25.0,
        settling=5.0,
        sample_duration=80.0,
        max_rate=0.20,
        ttfs_max_latency=20.0,
        silence_threshold=0.05,
        pool_weight=6.0,
        pool_threshold=-60.0,
        output_threshold=-62.0,
        readout_low=0.04,
        readout_high=0.12,
        readout_bound=0.30,
        learning_rate=0.04,
        trace_tau=20.0,
        temperature=1.0,
        first_spike_bonus=1.0,
        seed=seed,
        reward_mode="margin",
        reward_margin=1.0,
        readout_mode="pairwise",
        pairwise_decode="vote",
        shared_pool=True,
        shared_pool_channels=64,
        shared_pool_plasticity="modulated",
        shared_pool_learning_rate=0.05,
        shared_pool_trace_tau=5.0,
        shared_pool_eligibility_tau_plus=7_210.0,
        shared_pool_eligibility_tau_minus=3_610.0,
        shared_pool_reward_mode="correctness",
        shared_pool_correct_reward=1.0,
        shared_pool_incorrect_reward=-1.0,
        shared_pool_weight_low=0.1,
        shared_pool_weight_high=0.9,
        shared_pool_weight_bound=8.0,
        shared_pool_competition="winner_specific",
        shared_pool_inhibition_delay=0.1,
        shared_pool_normalization_interval=500,
        shared_pool_readout_scale=3.0,
        event_queue_capacity=262_144,
        output_capacity=65_536,
        encoder_spike_capacity=8_192,
        readout_eligibility_tau_plus=100.0,
        readout_eligibility_tau_minus=100.0,
    )


def _kernel_weights(
    config: AlternatingCNNConfig,
    *,
    stage: int,
    input_channels: int,
    output_channels: int,
    kernel: int,
) -> tuple[float, ...]:
    generator = random.Random(config.seed + (stage + 1) * 786_433)
    coefficients = kernel * kernel * input_channels
    target = config.convolution_kernel_sum_per_input_channel * input_channels
    result = []
    for _ in range(output_channels):
        values = [generator.uniform(0.1, 0.9) for _ in range(coefficients)]
        result.extend(
            _project_kernel_sum(
                values,
                target,
                lower=0.001,
                upper=config.shared_pool_weight_bound,
            )
        )
    return tuple(result)


def _stage_rule(config: AlternatingCNNConfig):
    if config.plasticity_mode == "pair":
        return PairSTDP(
            tau_pre=config.pair_trace_tau,
            tau_post=config.pair_trace_tau,
            a_plus=0.6,
            a_minus=0.3,
            learning_rate=config.pair_conv_learning_rate,
            bounds=(0.001, config.shared_pool_weight_bound),
        )
    return ModulatedSTDP(
        tau_pre=config.shared_pool_trace_tau,
        tau_post=config.shared_pool_trace_tau,
        tau_eligibility_plus=config.shared_pool_eligibility_tau_plus,
        tau_eligibility_minus=config.shared_pool_eligibility_tau_minus,
        learning_rate=config.reward_conv_learning_rate,
        bounds=(0.001, config.shared_pool_weight_bound),
        consume_on_modulation=True,
    )


def _readout_rule(config: AlternatingCNNConfig):
    scale = config.shared_pool_readout_scale or 1.0
    if config.plasticity_mode == "pair":
        return PairSTDP(
            tau_pre=config.pair_trace_tau,
            tau_post=config.pair_trace_tau,
            a_plus=0.6,
            a_minus=0.3,
            learning_rate=config.pair_readout_learning_rate,
            bounds=(0.001, config.readout_bound * scale),
        )
    return ModulatedSTDP(
        tau_pre=config.trace_tau,
        tau_post=config.trace_tau,
        tau_eligibility_plus=config.readout_eligibility_tau_plus,
        tau_eligibility_minus=config.readout_eligibility_tau_minus,
        learning_rate=config.reward_readout_learning_rate,
        bounds=(0.001, config.readout_bound * scale),
        consume_on_modulation=True,
    )


def _add_competition(
    builder: NetworkBuilder,
    stage,
    *,
    name: str,
    side: int,
    channels: int,
    magnitude: float,
    trigger: float,
    delay: float,
) -> tuple[int, ...]:
    if magnitude == 0.0 or channels <= 1:
        return ()
    inhibitors = builder.population(
        f"{name}_competition",
        side * side * channels,
        LIF(
            name=f"{name}_competition_lif",
            tau_m=5.0,
            v_threshold=-55.0,
            refractory=1.0,
        ),
        polarity=NeuronPolarity.INHIBITORY,
    )
    builder.connect(
        stage,
        inhibitors,
        pattern=OneToOne(),
        weight=trigger,
        delay=delay,
    )
    feedback = tuple(
        (position * channels + winner, position * channels + competitor)
        for position in range(side * side)
        for winner in range(channels)
        for competitor in range(channels)
        if competitor != winner
    )
    builder.connect(
        inhibitors,
        stage,
        pattern=ExplicitConnections(feedback),
        weight=magnitude,
        delay=delay,
    )
    return inhibitors.node_ids


def build_alternating_cnn(config: AlternatingCNNConfig) -> AlternatingCNNNetwork:
    builder = NetworkBuilder(
        f"mnist_alternating_cnn_{config.plasticity_mode}",
        metadata={"optimizer": "mnist_alternating_cnn", "config": asdict(config)},
    )
    pixels = builder.population(
        "pixels",
        784,
        LIF(name="alternating_pixel_lif", tau_m=10.0, refractory=1.0),
    )
    pixel_ports = tuple(builder.inputs("pixel", pixels, encoder=_encoder(config)))
    source = pixels
    input_shape = (28, 28, 1)
    stage_edges = []
    stage_shapes = []
    stage_nodes = []
    competition_nodes = []
    for stage_index, (channels, kernel, stride, padding, threshold) in enumerate(
        zip(
            config.cnn_channels,
            config.cnn_kernels,
            config.cnn_strides,
            config.cnn_paddings,
            config.cnn_thresholds,
        )
    ):
        pattern = Convolution2D(
            input_shape=input_shape,
            output_channels=channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        output_shape = pattern.output_shape
        stage = builder.population(
            f"conv{stage_index + 1}",
            output_shape[0] * output_shape[1] * output_shape[2],
            LIF(
                name=f"alternating_conv{stage_index + 1}_lif",
                tau_m=20.0,
                v_threshold=threshold,
                refractory=2.0,
            ),
        )
        edges = builder.connect(
            source,
            stage,
            pattern=pattern,
            weight=_kernel_weights(
                config,
                stage=stage_index,
                input_channels=input_shape[2],
                output_channels=channels,
                kernel=kernel,
            ),
            delay=0.1,
            plasticity=_stage_rule(config),
        )
        stage_edges.append(tuple(edges))
        stage_shapes.append(output_shape)
        stage_nodes.append(stage.node_ids)
        if config.record_stage_spikes:
            builder.outputs(f"conv{stage_index + 1}_activity", stage)
        competition_nodes.extend(
            _add_competition(
                builder,
                stage,
                name=f"conv{stage_index + 1}",
                side=output_shape[0],
                channels=channels,
                magnitude=config.convolution_inhibition,
                trigger=config.convolution_inhibition_trigger,
                delay=config.shared_pool_inhibition_delay,
            )
        )
        source = stage
        input_shape = output_shape

    all_stage_edges = tuple(edge for edges in stage_edges for edge in edges)
    shared_reward_port = (
        builder.modulator("feature_reward", targets=all_stage_edges)
        if config.plasticity_mode == "modulated"
        else None
    )
    digit_pairs = tuple(
        (first, second)
        for first in range(10)
        for second in range(first + 1, 10)
    )
    outputs = builder.population(
        "outputs",
        2 * len(digit_pairs),
        LIF(
            name="alternating_output_lif",
            tau_m=config.output_tau_m,
            v_threshold=config.output_threshold,
            refractory=2.0,
        ),
    )
    probe = builder.neuron(
        "eligibility_probe",
        LIF(name="alternating_probe_lif", tau_m=5.0, refractory=5.0),
    )
    probe_port = builder.input("probe_trigger", probe)
    builder.connect(probe, outputs, weight=20.0, delay=0.0)
    builder.outputs("digit", outputs)
    readout_rule = _readout_rule(config)
    readout_scale = config.shared_pool_readout_scale or 1.0
    reward_ports = []
    pairwise = []
    class_nodes = [[] for _ in range(10)]
    readout_edges = []
    for pair_index, (first, second) in enumerate(digit_pairs):
        first_node = outputs.node_ids[2 * pair_index]
        second_node = outputs.node_ids[2 * pair_index + 1]
        class_nodes[first].append(first_node)
        class_nodes[second].append(second_node)
        pairwise.append((first, second, first_node, second_node))
        for side, (digit, target) in enumerate(
            ((first, outputs[2 * pair_index]), (second, outputs[2 * pair_index + 1]))
        ):
            edges = tuple(
                builder.connect(
                    source,
                    target,
                    weight=Uniform(
                        config.readout_low * readout_scale,
                        config.readout_high * readout_scale,
                        seed=config.seed + pair_index * 131_071 + side,
                    ),
                    delay=0.0,
                    plasticity=readout_rule,
                )
            )
            readout_edges.extend(edges)
            if config.plasticity_mode == "modulated":
                reward_ports.append(
                    builder.modulator(
                        f"reward[{first},{second}][{digit}]", targets=edges
                    )
                )
    return AlternatingCNNNetwork(
        network=builder.build(),
        pixel_ports=pixel_ports,
        off_pixel_ports=(),
        output_nodes=outputs.node_ids,
        class_output_nodes=tuple(tuple(nodes) for nodes in class_nodes),
        pairwise_output_nodes=tuple(pairwise),
        ecoc_output_nodes=(),
        ecoc_codes=(),
        pairwise_decode=config.pairwise_decode,
        pairwise_decode_temperature=config.pairwise_decode_temperature,
        pairwise_decode_alpha=config.pairwise_decode_alpha,
        probe_port=probe_port,
        reward_ports=tuple(reward_ports),
        shared_pool_reward_port=shared_reward_port,
        shared_pool_edges=all_stage_edges,
        readout_weight_scale=readout_scale,
        feature_nodes=source.node_ids,
        competition_nodes=tuple(competition_nodes),
        stage_edges=tuple(stage_edges),
        stage_shapes=tuple(stage_shapes),
        stage_nodes=tuple(stage_nodes),
        feedforward_edges=all_stage_edges + tuple(readout_edges),
    )


def measure_stage_activity(
    core,
    config: AlternatingCNNConfig,
    learned_weights,
    images,
) -> tuple[dict[str, float], ...]:
    diagnostic_config = replace(
        config,
        plasticity_mode="modulated",
        record_stage_spikes=True,
    )
    candidate = candidate_with_weights(
        build_alternating_cnn(diagnostic_config), learned_weights
    )
    frozen_graph = replace(
        candidate.network.graph,
        edges=tuple(
            replace(edge, weight=learned_weights[edge.id], plasticity=None)
            for edge in candidate.network.graph.edges
        ),
        modulator_ports=(),
    )
    node_to_stage = {
        node: stage
        for stage, nodes in enumerate(candidate.stage_nodes)
        for node in nodes
    }
    output_nodes = set(candidate.output_nodes)
    counts = [0] * (len(candidate.stage_nodes) + 1)
    active_samples = [0] * len(counts)
    with frozen_graph.resolve().compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=diagnostic_config.sample_duration * len(images),
            encoder_seed=diagnostic_config.seed + 1_000_003,
            queue_capacity=diagnostic_config.event_queue_capacity,
            output_capacity=diagnostic_config.output_capacity,
            encoder_spike_capacity=diagnostic_config.encoder_spike_capacity,
        ) as run:
            for sample, image in enumerate(images):
                start = sample * diagnostic_config.sample_duration
                end = start + diagnostic_config.presentation
                sample_end = start + diagnostic_config.sample_duration
                result = (
                    run.finish(scalar_inputs=_presentations(candidate, image, start, end))
                    if sample + 1 == len(images)
                    else run.advance_until(
                        sample_end,
                        scalar_inputs=_presentations(candidate, image, start, end),
                    )
                )
                sample_counts = [0] * len(counts)
                for event in result.core.spikes:
                    if event.node in output_nodes:
                        sample_counts[-1] += 1
                    else:
                        stage = node_to_stage.get(event.node)
                        if stage is not None:
                            sample_counts[stage] += 1
                for index, value in enumerate(sample_counts):
                    counts[index] += value
                    active_samples[index] += int(value > 0)
    names = [f"conv{index + 1}" for index in range(len(candidate.stage_nodes))]
    names.append("readout")
    return tuple(
        {
            "name": name,
            "mean_spikes": count / len(images),
            "active_fraction": active / len(images),
        }
        for name, count, active in zip(names, counts, active_samples)
    )


def normalize_convolution_kernels(
    candidate: AlternatingCNNNetwork,
    config: AlternatingCNNConfig,
    learned_weights,
) -> tuple[float, ...]:
    normalized = [float(value) for value in learned_weights]
    input_channels = 1
    for edges, output_channels in zip(candidate.stage_edges, config.cnn_channels):
        groups: dict[int, list[int]] = {}
        for edge_id in edges:
            group = candidate.network.graph.edges[edge_id].weight_group
            if group is None:
                raise ValueError("convolution edge is missing a shared weight group")
            groups.setdefault(group, []).append(edge_id)
        ordered_groups = sorted(groups)
        coefficients = len(ordered_groups) // output_channels
        target = config.convolution_kernel_sum_per_input_channel * input_channels
        for channel in range(output_channels):
            kernel_groups = ordered_groups[
                channel * coefficients : (channel + 1) * coefficients
            ]
            values = [normalized[groups[group][0]] for group in kernel_groups]
            projected = _project_kernel_sum(
                values,
                target,
                lower=0.001,
                upper=config.shared_pool_weight_bound,
            )
            for group, value in zip(kernel_groups, projected):
                for edge_id in groups[group]:
                    normalized[edge_id] = value
        input_channels = output_channels
    return tuple(normalized)


def train_pass(
    core,
    config: AlternatingCNNConfig,
    mode: str,
    images,
    labels,
    *,
    weights=None,
    block_size=500,
    block_callback=None,
):
    mode_config = replace(config, plasticity_mode=mode)
    metrics = []
    blocks = range(0, len(labels), block_size)
    for block_index, start in enumerate(blocks):
        stop = min(start + block_size, len(labels))
        block_config = replace(
            mode_config,
            seed=mode_config.seed + block_index * 104_729,
        )
        candidate = build_alternating_cnn(block_config)
        if weights is not None:
            candidate = candidate_with_weights(candidate, weights)
        weights, block_metrics = train_candidate(
            core,
            candidate,
            block_config,
            images[start:stop],
            labels[start:stop],
        )
        weights = normalize_convolution_kernels(candidate, block_config, weights)
        metrics.append(block_metrics)
        if block_callback is not None:
            block_callback(mode, block_index + 1, stop, weights, tuple(metrics))
    return tuple(weights), _combine_metrics(metrics)


def run_alternating(
    core,
    config: AlternatingCNNConfig,
    train_images,
    train_labels,
    validation_images,
    validation_labels,
    *,
    epochs: int,
    block_size: int,
    checkpoint_path: Path | None = None,
):
    started = time.perf_counter()
    generator = np.random.default_rng(config.seed)
    weights = None
    history = []

    def checkpoint(mode, block, samples, current_weights, metrics):
        if checkpoint_path is None:
            return
        payload = {
            "config": asdict(config),
            "history": history,
            "active_pass": mode,
            "active_block": block,
            "active_samples": samples,
            "active_metrics": [asdict(item) for item in metrics],
            "learned_weights": [float(value) for value in current_weights],
            "elapsed_seconds": time.perf_counter() - started,
        }
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(checkpoint_path)

    for epoch in range(epochs):
        pair_order = generator.permutation(len(train_labels))
        weights, pair_metrics = train_pass(
            core,
            replace(config, seed=config.seed + epoch * 2_000_003),
            "pair",
            train_images[pair_order],
            train_labels[pair_order],
            weights=weights,
            block_size=block_size,
            block_callback=checkpoint,
        )
        reward_order = generator.permutation(len(train_labels))
        weights, reward_metrics = train_pass(
            core,
            replace(config, seed=config.seed + epoch * 2_000_003 + 1_000_003),
            "modulated",
            train_images[reward_order],
            train_labels[reward_order],
            weights=weights,
            block_size=block_size,
            block_callback=checkpoint,
        )
        evaluation_config = replace(config, plasticity_mode="modulated")
        evaluation_candidate = candidate_with_weights(
            build_alternating_cnn(evaluation_config), weights
        )
        validation = evaluate_candidate(
            core,
            evaluation_candidate,
            evaluation_config,
            weights,
            validation_images,
            validation_labels,
        )
        history.append(
            {
                "epoch": epoch + 1,
                "pair": asdict(pair_metrics),
                "reward": asdict(reward_metrics),
                "validation": asdict(validation),
            }
        )
        print(
            f"epoch {epoch + 1}: pair={100 * pair_metrics.accuracy:.2f}% "
            f"reward={100 * reward_metrics.accuracy:.2f}% "
            f"validation={100 * validation.accuracy:.2f}%",
            flush=True,
        )
    return {
        "config": asdict(config),
        "history": history,
        "learned_weights": [float(value) for value in weights],
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build/liblacuna_core.dylib")
    parser.add_argument("--data-dir", default="artifacts/mnist")
    parser.add_argument("--train-per-class", type=int, default=5)
    parser.add_argument("--validation-per-class", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=50)
    parser.add_argument("--channels", type=int, nargs=4, default=(16, 32, 48, 64))
    parser.add_argument("--thresholds", type=float, nargs=4)
    parser.add_argument("--pair-ratio", type=float, default=0.02)
    parser.add_argument("--reward-conv-learning-rate", type=float, default=0.05)
    parser.add_argument("--feature-incorrect-reward", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--output",
        default="artifacts/optimization/mnist_alternating_cnn.json",
    )
    args = parser.parse_args()
    if not 0.0 < args.pair_ratio < 1.0:
        raise ValueError("pair-ratio must lie strictly between zero and one")
    if args.reward_conv_learning_rate <= 0.0:
        raise ValueError("reward-conv-learning-rate must be positive")
    if args.feature_incorrect_reward >= 0.0:
        raise ValueError("feature-incorrect-reward must be negative")
    config = replace(
        default_config(args.seed),
        cnn_channels=tuple(args.channels),
        cnn_thresholds=(
            tuple(args.thresholds)
            if args.thresholds is not None
            else default_config(args.seed).cnn_thresholds
        ),
        shared_pool_channels=args.channels[-1],
        pair_conv_learning_rate=0.05 * args.pair_ratio,
        pair_readout_learning_rate=0.04 * args.pair_ratio,
        reward_conv_learning_rate=args.reward_conv_learning_rate,
        shared_pool_incorrect_reward=args.feature_incorrect_reward,
    )
    train_images, train_labels, _, _ = load_mnist(args.data_dir, download=False)
    train_indices = balanced_indices(
        train_labels, args.train_per_class, offset=0, seed=args.seed
    )
    validation_indices = balanced_indices(
        train_labels,
        args.validation_per_class,
        offset=args.train_per_class,
        seed=args.seed + 1,
    )
    destination = Path(args.output)
    result = run_alternating(
        CoreEvaluator(args.library),
        config,
        train_images[train_indices],
        train_labels[train_indices],
        train_images[validation_indices],
        train_labels[validation_indices],
        epochs=args.epochs,
        block_size=args.block_size,
        checkpoint_path=destination.with_suffix(".checkpoint.json"),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
