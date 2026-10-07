#!/usr/bin/env python3
"""Standalone neuron-only MNIST architecture and hyperparameter search.

This file intentionally uses Lacuna's existing public authoring and execution
interfaces without modifying the engine.  Every operation inside a candidate
network is a neuron, delta synapse, or existing plastic synapse rule.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from lacuna import (
    AdaptiveLIF,
    BurstEncoder,
    Convolution2D,
    CoreEvaluator,
    ExplicitConnections,
    LIF,
    ModulatedSTDP,
    ModulationInput,
    Network,
    NetworkBuilder,
    OneToOne,
    PairSTDP,
    PoissonRateEncoder,
    RegularRateEncoder,
    ScalarInput,
    SpikeInput,
    TTFSEncoder,
    Uniform,
    NeuronPolarity,
)
from lacuna.experiments import load_mnist


@dataclass(frozen=True)
class CandidateConfig:
    architecture: str
    encoder: str
    presentation: float
    settling: float
    sample_duration: float
    max_rate: float
    ttfs_max_latency: float
    silence_threshold: float
    pool_weight: float
    pool_threshold: float
    output_threshold: float
    readout_low: float
    readout_high: float
    readout_bound: float
    learning_rate: float
    trace_tau: float
    temperature: float
    first_spike_bonus: float
    seed: int
    outputs_per_class: int = 1
    local_weight_low: float = 2.0
    local_weight_high: float = 6.0
    local_inhibition: float = 0.0
    local_inhibition_trigger: float = 12.0
    reward_mode: str = "softmax"
    output_inhibition: float = 0.0
    reward_margin: float = 1.0
    readout_mode: str = "multiclass"
    ecoc_bits: int = 24
    output_tau_m: float = 20.0
    pool_kernel: int = 2
    pairwise_decode: str = "vote"
    pairwise_decode_temperature: float = 1.0
    pairwise_decode_alpha: float = 0.1
    pool_side: int = 14
    secondary_pool_weight: float = 3.0
    input_gamma: float = 1.0
    input_threshold: float = 0.0
    input_levels: int = 0
    shared_pool: bool = False
    shared_pool_channels: int = 1
    shared_pool_plasticity: str = "static"
    shared_pool_learning_rate: float = 0.01
    shared_pool_trace_tau: float = 5.0
    shared_pool_eligibility_tau_plus: float = 7_210.0
    shared_pool_eligibility_tau_minus: float = 3_610.0
    shared_pool_reward_mode: str = "correctness"
    shared_pool_reward_scale: float = 1.0
    shared_pool_correct_reward: float = 1.0
    shared_pool_incorrect_reward: float = -1.0
    shared_pool_weight_low: float = 2.0
    shared_pool_weight_high: float = 6.0
    shared_pool_weight_bound: float = 8.0
    shared_pool_competition: str = "shared_relay"
    shared_pool_inhibition_delay: float = 0.1
    shared_pool_adaptive: bool = False
    shared_pool_tau_adaptation: float = 120.0
    shared_pool_adaptation_increment: float = 0.5
    shared_pool_normalize_initial_kernels: bool = False
    shared_pool_kernel_sum: float = 0.0
    shared_pool_readout_scale: float = 0.0
    shared_pool_stride: int = 2
    shared_pool_normalization_interval: int = 0
    shared_pool_normalization_sum: float = 0.0
    event_queue_capacity: int = 16_384
    output_capacity: int = 8_192
    encoder_spike_capacity: int = 8_192
    readout_eligibility_tau_plus: float = 100.0
    readout_eligibility_tau_minus: float = 100.0
    readout_tau_pre: float | None = None
    readout_tau_post: float | None = None
    readout_consume_on_modulation: bool = True

    def __post_init__(self) -> None:
        if self.architecture not in {
            "direct",
            "multiscale",
            "onoff",
            "pool2",
            "local1",
            "local2",
        }:
            raise ValueError("unsupported candidate architecture")
        if self.encoder not in {"burst", "poisson", "regular", "ttfs"}:
            raise ValueError("unsupported candidate encoder")
        if self.outputs_per_class <= 0:
            raise ValueError("outputs_per_class must be positive")
        if not 0.0 <= self.local_weight_low <= self.local_weight_high:
            raise ValueError("local weights must be nonnegative and ordered")
        if self.local_inhibition < 0.0 or self.local_inhibition_trigger <= 0.0:
            raise ValueError("local inhibition magnitudes are invalid")
        if self.reward_mode not in {"softmax", "perceptron", "margin"}:
            raise ValueError("unsupported reward mode")
        if self.output_inhibition < 0.0:
            raise ValueError("output inhibition must be nonnegative")
        if self.reward_margin < 0.0:
            raise ValueError("reward margin must be nonnegative")
        if self.readout_mode not in {"multiclass", "pairwise", "ecoc"}:
            raise ValueError("unsupported readout mode")
        if self.ecoc_bits <= 0:
            raise ValueError("ecoc_bits must be positive")
        if self.output_tau_m <= 0.0:
            raise ValueError("output_tau_m must be positive")
        if not 1 <= self.pool_kernel <= 28:
            raise ValueError("pool_kernel must lie in [1, 28]")
        if self.pairwise_decode not in {"vote", "margin", "sigmoid", "hybrid"}:
            raise ValueError("unsupported pairwise decoder")
        if self.pairwise_decode_temperature <= 0.0:
            raise ValueError("pairwise decoder temperature must be positive")
        if self.pairwise_decode_alpha < 0.0:
            raise ValueError("pairwise decoder alpha must be nonnegative")
        if not 2 <= self.pool_side <= 28:
            raise ValueError("pool_side must lie in [2, 28]")
        if self.secondary_pool_weight < 0.0:
            raise ValueError("secondary_pool_weight must be nonnegative")
        if self.input_gamma <= 0.0:
            raise ValueError("input_gamma must be positive")
        if not 0.0 <= self.input_threshold < 1.0:
            raise ValueError("input_threshold must lie in [0, 1)")
        if self.input_levels not in {0} and self.input_levels < 2:
            raise ValueError("input_levels must be zero or at least two")
        if self.shared_pool_channels <= 0:
            raise ValueError("shared_pool_channels must be positive")
        if self.shared_pool_plasticity not in {"static", "pair", "modulated"}:
            raise ValueError(
                "shared_pool_plasticity must be static, pair, or modulated"
            )
        if self.shared_pool_trace_tau <= 0.0:
            raise ValueError("shared-pool spike-trace tau must be positive")
        if min(
            self.shared_pool_eligibility_tau_plus,
            self.shared_pool_eligibility_tau_minus,
        ) <= 0.0:
            raise ValueError("shared-pool eligibility taus must be positive")
        if self.shared_pool_reward_mode not in {"correctness", "signed_margin"}:
            raise ValueError("unsupported shared-pool reward mode")
        if self.shared_pool_reward_scale <= 0.0:
            raise ValueError("shared-pool reward scale must be positive")
        if not math.isfinite(self.shared_pool_correct_reward) or not math.isfinite(
            self.shared_pool_incorrect_reward
        ):
            raise ValueError("shared-pool correctness rewards must be finite")
        if self.shared_pool_correct_reward <= 0.0:
            raise ValueError("shared-pool correct reward must be positive")
        if self.shared_pool_incorrect_reward >= 0.0:
            raise ValueError("shared-pool incorrect reward must be negative")
        if self.shared_pool_competition not in {
            "none",
            "shared_relay",
            "winner_specific",
        }:
            raise ValueError("unsupported shared-pool competition mode")
        if self.shared_pool_inhibition_delay < 0.0:
            raise ValueError("shared-pool inhibition delay must be nonnegative")
        if self.shared_pool_tau_adaptation <= 0.0:
            raise ValueError("shared-pool adaptation tau must be positive")
        if self.shared_pool_adaptation_increment < 0.0:
            raise ValueError("shared-pool adaptation increment must be nonnegative")
        if self.shared_pool_kernel_sum < 0.0:
            raise ValueError("shared-pool kernel sum must be nonnegative")
        if self.shared_pool_readout_scale < 0.0:
            raise ValueError("shared-pool readout scale must be nonnegative")
        if self.shared_pool_stride <= 0:
            raise ValueError("shared-pool stride must be positive")
        if self.shared_pool_normalization_interval < 0:
            raise ValueError("shared-pool normalization interval must be nonnegative")
        if self.shared_pool_normalization_sum < 0.0:
            raise ValueError("shared-pool normalization sum must be nonnegative")
        if min(
            self.event_queue_capacity,
            self.output_capacity,
            self.encoder_spike_capacity,
        ) <= 0:
            raise ValueError("execution capacities must be positive")
        if min(
            self.readout_eligibility_tau_plus,
            self.readout_eligibility_tau_minus,
        ) <= 0.0:
            raise ValueError("readout eligibility taus must be positive")
        for name in ("readout_tau_pre", "readout_tau_post"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when supplied")
        if not isinstance(self.readout_consume_on_modulation, bool):
            raise ValueError("readout_consume_on_modulation must be boolean")
        if self.shared_pool and self.architecture != "pool2":
            raise ValueError("shared pooling is currently supported by pool2 only")
        expected_pool_side = (
            (28 - self.pool_kernel) // self.shared_pool_stride + 1
        )
        if self.shared_pool and self.pool_side != expected_pool_side:
            raise ValueError(
                "shared pool_side must match the valid convolution output size "
                f"({expected_pool_side} for kernel={self.pool_kernel}, "
                f"stride={self.shared_pool_stride})"
            )
        if self.shared_pool_learning_rate < 0.0:
            raise ValueError("shared_pool_learning_rate must be nonnegative")
        if not (
            0.0
            <= self.shared_pool_weight_low
            <= self.shared_pool_weight_high
            <= self.shared_pool_weight_bound
        ):
            raise ValueError("shared pooling weights must be ordered and within bounds")
        if not self.readout_low <= self.readout_high <= self.readout_bound:
            raise ValueError("readout initialization must lie inside its bound")
        if self.sample_duration <= self.presentation + self.settling + 5.0:
            raise ValueError("sample duration does not leave a post-reward rest")


@dataclass(frozen=True)
class CandidateNetwork:
    network: Network
    pixel_ports: tuple[str, ...]
    off_pixel_ports: tuple[str, ...]
    output_nodes: tuple[int, ...]
    class_output_nodes: tuple[tuple[int, ...], ...]
    pairwise_output_nodes: tuple[tuple[int, int, int, int], ...]
    ecoc_output_nodes: tuple[tuple[int, int], ...]
    ecoc_codes: tuple[tuple[int, ...], ...]
    pairwise_decode: str
    pairwise_decode_temperature: float
    pairwise_decode_alpha: float
    probe_port: str
    reward_ports: tuple[str, ...]
    shared_pool_reward_port: str | None
    shared_pool_edges: tuple[int, ...]
    readout_weight_scale: float
    feature_nodes: tuple[int, ...]
    competition_nodes: tuple[int, ...]


@dataclass(frozen=True)
class Metrics:
    samples: int
    accuracy: float
    mean_output_spikes: float
    silent_fraction: float
    mean_loss: float


@dataclass(frozen=True)
class TrialResult:
    config: CandidateConfig
    train: Metrics
    validation: Metrics
    evaluation_threshold: float
    validation_by_threshold: dict[str, Metrics]
    learning_rate_schedule: tuple[float, ...]
    weight_mean: float
    weight_at_lower_bound: float
    weight_at_upper_bound: float
    learned_weights: tuple[float, ...]
    shared_kernel_weights: tuple[float, ...]
    weight_parameter_count: int
    shared_weight_group_count: int
    readout_weight_scale: float
    seconds: float


def _encoder(config: CandidateConfig):
    if config.encoder == "burst":
        return BurstEncoder(
            0.0,
            config.max_rate,
            duration=config.presentation,
            amplitude=20.0,
        )
    if config.encoder == "poisson":
        return PoissonRateEncoder(0.0, config.max_rate, amplitude=20.0)
    if config.encoder == "regular":
        return RegularRateEncoder(0.0, config.max_rate, amplitude=20.0)
    return TTFSEncoder(
        min_latency=1.0,
        max_latency=config.ttfs_max_latency,
        amplitude=20.0,
        silence_threshold=config.silence_threshold,
    )


def _pool_pairs(kernel: int = 2, output_side: int = 14) -> tuple[tuple[int, int], ...]:
    pairs = []
    maximum_start = 28 - kernel
    for output_row in range(output_side):
        input_start_row = (
            output_row * 2
            if output_side == 14
            else round(output_row * maximum_start / (output_side - 1))
        )
        for output_column in range(output_side):
            input_start_column = (
                output_column * 2
                if output_side == 14
                else round(output_column * maximum_start / (output_side - 1))
            )
            target = output_row * output_side + output_column
            for row_offset in range(kernel):
                input_row = input_start_row + row_offset
                if input_row >= 28:
                    continue
                for column_offset in range(kernel):
                    input_column = input_start_column + column_offset
                    if input_column >= 28:
                        continue
                    source = input_row * 28 + input_column
                    pairs.append((source, target))
    return tuple(pairs)


def _local_pairs(
    input_side: int,
    input_channels: int,
    output_side: int,
    output_channels: int,
) -> tuple[tuple[int, int], ...]:
    pairs = []
    for output_row in range(output_side):
        for output_column in range(output_side):
            target_position = output_row * output_side + output_column
            for output_channel in range(output_channels):
                target = target_position * output_channels + output_channel
                for row_offset in range(2):
                    for column_offset in range(2):
                        input_row = min(output_row * 2 + row_offset, input_side - 1)
                        input_column = min(
                            output_column * 2 + column_offset, input_side - 1
                        )
                        source_position = input_row * input_side + input_column
                        for input_channel in range(input_channels):
                            source = source_position * input_channels + input_channel
                            pairs.append((source, target))
    return tuple(pairs)


def _add_local_inhibition(
    builder: NetworkBuilder,
    stage,
    *,
    side: int,
    channels: int,
    magnitude: float,
    trigger: float,
    index: int,
) -> None:
    if magnitude == 0.0:
        return
    inhibitors = builder.population(
        f"local{index}_inhibitors",
        side * side,
        LIF(
            name=f"search_local{index}_inhibitory_lif",
            tau_m=5.0,
            v_threshold=-55.0,
            refractory=1.0,
        ),
        polarity=NeuronPolarity.INHIBITORY,
    )
    forward = tuple(
        (position * channels + channel, position)
        for position in range(side * side)
        for channel in range(channels)
    )
    feedback = tuple(
        (position, position * channels + channel)
        for position in range(side * side)
        for channel in range(channels)
    )
    builder.connect(
        stage,
        inhibitors,
        pattern=ExplicitConnections(forward),
        weight=trigger,
        delay=0.1,
    )
    builder.connect(
        inhibitors,
        stage,
        pattern=ExplicitConnections(feedback),
        weight=magnitude,
        delay=0.1,
    )


def _add_winner_specific_inhibition(
    builder: NetworkBuilder,
    stage,
    *,
    side: int,
    channels: int,
    magnitude: float,
    trigger: float,
    delay: float,
) -> tuple[int, ...]:
    """Expand same-position, self-excluding competition through inhibitory proxies."""
    if magnitude == 0.0 or channels <= 1:
        return ()
    inhibitors = builder.population(
        "shared_pool_competition",
        side * side * channels,
        LIF(
            name="search_shared_pool_competition_lif",
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
        (
            position * channels + winner,
            position * channels + competitor,
        )
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


def _ecoc_codes(bits: int, seed: int) -> tuple[tuple[int, ...], ...]:
    generator = random.Random(seed + 9_999_991)
    rows = [[] for _ in range(10)]
    for _ in range(bits):
        column = [0] * 5 + [1] * 5
        generator.shuffle(column)
        for digit, value in enumerate(column):
            rows[digit].append(value)
    return tuple(tuple(row) for row in rows)


def _normalized_shared_pool_weights(
    config: CandidateConfig, parameter_count: int
) -> tuple[float, ...]:
    coefficients_per_kernel = parameter_count // config.shared_pool_channels
    target_sum = config.shared_pool_kernel_sum or (
        coefficients_per_kernel
        * (config.shared_pool_weight_low + config.shared_pool_weight_high)
        / 2.0
    )
    generator = random.Random(config.seed + 786_433)
    result = []
    for _ in range(config.shared_pool_channels):
        kernel = [
            generator.uniform(
                config.shared_pool_weight_low,
                config.shared_pool_weight_high,
            )
            for _ in range(coefficients_per_kernel)
        ]
        total = sum(kernel)
        if total == 0.0:
            kernel = [target_sum / coefficients_per_kernel] * coefficients_per_kernel
        else:
            kernel = [coefficient * target_sum / total for coefficient in kernel]
        if any(
            coefficient < 0.0
            or coefficient > config.shared_pool_weight_bound
            for coefficient in kernel
        ):
            raise ValueError(
                "normalized shared-pool initialization exceeds its weight bound"
            )
        result.extend(kernel)
    return tuple(result)


def build_candidate(config: CandidateConfig) -> CandidateNetwork:
    builder = NetworkBuilder(
        f"mnist_search_{config.architecture}_{config.encoder}",
        metadata={"optimizer": "mnist_neuron_search", "config": asdict(config)},
    )
    pixels = builder.population(
        "pixels",
        784,
        LIF(name="search_pixel_lif", tau_m=10.0, refractory=1.0),
    )
    pixel_ports = tuple(builder.inputs("pixel", pixels, encoder=_encoder(config)))
    off_pixel_ports = ()
    shared_pool_edges = ()
    shared_pool_reward_port = None
    feature_nodes = ()
    competition_nodes = ()
    if config.architecture == "onoff":
        off_pixels = builder.population(
            "off_pixels",
            784,
            LIF(name="search_off_pixel_lif", tau_m=10.0, refractory=1.0),
        )
        off_pixel_ports = tuple(
            builder.inputs("off_pixel", off_pixels, encoder=_encoder(config))
        )
        source = pixels
        readout_sources = (pixels, off_pixels)
    elif config.architecture == "multiscale":
        tight_pool = builder.population(
            "tight_pool",
            196,
            LIF(
                name="search_tight_pool_lif",
                tau_m=20.0,
                v_threshold=config.pool_threshold,
                refractory=2.0,
            ),
        )
        broad_pool = builder.population(
            "broad_pool",
            196,
            LIF(
                name="search_broad_pool_lif",
                tau_m=20.0,
                v_threshold=config.pool_threshold,
                refractory=2.0,
            ),
        )
        builder.connect(
            pixels,
            tight_pool,
            pattern=ExplicitConnections(_pool_pairs(2, 14)),
            weight=config.pool_weight,
            delay=0.1,
        )
        builder.connect(
            pixels,
            broad_pool,
            pattern=ExplicitConnections(_pool_pairs(3, 14)),
            weight=config.secondary_pool_weight,
            delay=0.1,
        )
        source = tight_pool
        readout_sources = (tight_pool, broad_pool)
    elif config.architecture == "pool2":
        pool_channels = config.shared_pool_channels if config.shared_pool else 1
        pool_model = (
            AdaptiveLIF(
                name="search_pool_adaptive_lif",
                tau_m=20.0,
                tau_adaptation=config.shared_pool_tau_adaptation,
                adaptation_increment=config.shared_pool_adaptation_increment,
                v_threshold=config.pool_threshold,
                refractory=2.0,
            )
            if config.shared_pool and config.shared_pool_adaptive
            else LIF(
                name="search_pool_lif",
                tau_m=20.0,
                v_threshold=config.pool_threshold,
                refractory=2.0,
            )
        )
        source = builder.population(
            "pool",
            config.pool_side * config.pool_side * pool_channels,
            pool_model,
        )
        feature_nodes = source.node_ids
        if config.shared_pool:
            pattern = Convolution2D(
                input_shape=(28, 28, 1),
                output_channels=pool_channels,
                kernel_size=config.pool_kernel,
                stride=config.shared_pool_stride,
            )
            pool_plasticity = None
            pool_weights = (config.pool_weight,) * pattern.kernel_parameter_count
            if config.shared_pool_plasticity in {"pair", "modulated"}:
                pool_weights = (
                    _normalized_shared_pool_weights(
                        config, pattern.kernel_parameter_count
                    )
                    if config.shared_pool_normalize_initial_kernels
                    else Uniform(
                        config.shared_pool_weight_low,
                        config.shared_pool_weight_high,
                        seed=config.seed + 786_433,
                    )
                )
                if config.shared_pool_plasticity == "pair":
                    pool_plasticity = PairSTDP(
                        tau_pre=config.trace_tau,
                        tau_post=config.trace_tau,
                        a_plus=0.6,
                        a_minus=0.3,
                        learning_rate=config.shared_pool_learning_rate,
                        bounds=(0.001, config.shared_pool_weight_bound),
                    )
                else:
                    pool_plasticity = ModulatedSTDP(
                        tau_pre=config.shared_pool_trace_tau,
                        tau_post=config.shared_pool_trace_tau,
                        tau_eligibility_plus=(
                            config.shared_pool_eligibility_tau_plus
                        ),
                        tau_eligibility_minus=(
                            config.shared_pool_eligibility_tau_minus
                        ),
                        learning_rate=config.shared_pool_learning_rate,
                        bounds=(0.001, config.shared_pool_weight_bound),
                        consume_on_modulation=True,
                    )
            shared_pool_edges = builder.connect(
                pixels,
                source,
                pattern=pattern,
                weight=pool_weights,
                delay=0.1,
                plasticity=pool_plasticity,
            )
            if config.shared_pool_plasticity == "modulated":
                shared_pool_reward_port = builder.modulator(
                    "shared_pool_reward", targets=shared_pool_edges
                )
            if config.shared_pool_competition == "shared_relay":
                _add_local_inhibition(
                    builder,
                    source,
                    side=config.pool_side,
                    channels=pool_channels,
                    magnitude=config.local_inhibition,
                    trigger=config.local_inhibition_trigger,
                    index=0,
                )
            elif config.shared_pool_competition == "winner_specific":
                competition_nodes = _add_winner_specific_inhibition(
                    builder,
                    source,
                    side=config.pool_side,
                    channels=pool_channels,
                    magnitude=config.local_inhibition,
                    trigger=config.local_inhibition_trigger,
                    delay=config.shared_pool_inhibition_delay,
                )
        else:
            builder.connect(
                pixels,
                source,
                pattern=ExplicitConnections(
                    _pool_pairs(config.pool_kernel, config.pool_side)
                ),
                weight=config.pool_weight,
                delay=0.1,
            )
    elif config.architecture in {"local1", "local2"}:
        source = pixels
        input_side = 28
        input_channels = 1
        channels_by_stage = (4,) if config.architecture == "local1" else (4, 8)
        for index, output_channels in enumerate(channels_by_stage, start=1):
            output_side = (input_side + 1) // 2
            stage = builder.population(
                f"local{index}",
                output_side * output_side * output_channels,
                LIF(
                    name=f"search_local{index}_lif",
                    tau_m=15.0,
                    v_threshold=config.pool_threshold,
                    refractory=2.0,
                ),
            )
            builder.connect(
                source,
                stage,
                pattern=ExplicitConnections(
                    _local_pairs(
                        input_side,
                        input_channels,
                        output_side,
                        output_channels,
                    )
                ),
                weight=Uniform(
                    config.local_weight_low / input_channels,
                    config.local_weight_high / input_channels,
                    seed=config.seed + index * 524_287,
                ),
                delay=0.1,
            )
            _add_local_inhibition(
                builder,
                stage,
                side=output_side,
                channels=output_channels,
                magnitude=config.local_inhibition,
                trigger=config.local_inhibition_trigger,
                index=index,
            )
            source = stage
            input_side = output_side
            input_channels = output_channels
    else:
        source = pixels
    if config.architecture not in {"onoff", "multiscale"}:
        readout_sources = (source,)
    readout_weight_scale = 1.0
    if config.shared_pool and config.shared_pool_channels > 1:
        readout_weight_scale = (
            config.shared_pool_readout_scale
            if config.shared_pool_readout_scale > 0.0
            else 1.0 / config.shared_pool_channels
        )

    digit_pairs = tuple(
        (first, second)
        for first in range(10)
        for second in range(first + 1, 10)
    )
    if config.readout_mode == "pairwise":
        output_count = 2 * len(digit_pairs)
    elif config.readout_mode == "ecoc":
        output_count = 2 * config.ecoc_bits
    else:
        output_count = 10 * config.outputs_per_class
    outputs = builder.population(
        "outputs",
        output_count,
        LIF(
            name="search_output_lif",
            tau_m=config.output_tau_m,
            v_threshold=config.output_threshold,
            refractory=2.0,
        ),
    )
    probe = builder.neuron(
        "eligibility_probe",
        LIF(name="search_probe_lif", tau_m=5.0, refractory=5.0),
    )
    probe_port = builder.input("probe_trigger", probe)
    builder.connect(probe, outputs, weight=20.0, delay=0.0)
    if config.output_inhibition > 0.0:
        output_inhibitor = builder.neuron(
            "output_inhibitor",
            LIF(
                name="search_output_inhibitory_lif",
                tau_m=5.0,
                v_threshold=-55.0,
                refractory=1.0,
            ),
            polarity=NeuronPolarity.INHIBITORY,
        )
        builder.connect(outputs, output_inhibitor, weight=12.0, delay=0.1)
        builder.connect(
            output_inhibitor,
            outputs,
            weight=config.output_inhibition,
            delay=0.1,
        )
    builder.outputs("digit", outputs)

    rule = ModulatedSTDP(
        tau_pre=(
            config.trace_tau
            if config.readout_tau_pre is None
            else config.readout_tau_pre
        ),
        tau_post=(
            config.trace_tau
            if config.readout_tau_post is None
            else config.readout_tau_post
        ),
        tau_eligibility_plus=config.readout_eligibility_tau_plus,
        tau_eligibility_minus=config.readout_eligibility_tau_minus,
        learning_rate=config.learning_rate,
        bounds=(0.001, config.readout_bound * readout_weight_scale),
        consume_on_modulation=config.readout_consume_on_modulation,
    )
    reward_ports = []
    pairwise_output_nodes = ()
    ecoc_output_nodes = ()
    ecoc_codes = ()

    def connect_readout(target, seed):
        edges = []
        for source_index, readout_source in enumerate(readout_sources):
            edges.extend(
                builder.connect(
                    readout_source,
                    target,
                    weight=Uniform(
                        config.readout_low * readout_weight_scale,
                        config.readout_high * readout_weight_scale,
                        seed=seed + source_index * 15_485_863,
                    ),
                    delay=0.0,
                    plasticity=rule,
                )
            )
        return tuple(edges)

    if config.readout_mode == "pairwise":
        class_nodes = [[] for _ in range(10)]
        pairwise = []
        for pair_index, (first, second) in enumerate(digit_pairs):
            first_node = outputs.node_ids[2 * pair_index]
            second_node = outputs.node_ids[2 * pair_index + 1]
            class_nodes[first].append(first_node)
            class_nodes[second].append(second_node)
            pairwise.append((first, second, first_node, second_node))
            for side, (digit, target) in enumerate(
                ((first, outputs[2 * pair_index]), (second, outputs[2 * pair_index + 1]))
            ):
                edges = connect_readout(
                    target, config.seed + pair_index * 131_071 + side
                )
                reward_ports.append(
                    builder.modulator(
                        f"reward[{first},{second}][{digit}]", targets=edges
                    )
                )
        class_output_nodes = tuple(tuple(nodes) for nodes in class_nodes)
        pairwise_output_nodes = tuple(pairwise)
    elif config.readout_mode == "ecoc":
        ecoc_codes = _ecoc_codes(config.ecoc_bits, config.seed)
        ecoc_output_nodes = tuple(
            (outputs.node_ids[2 * bit], outputs.node_ids[2 * bit + 1])
            for bit in range(config.ecoc_bits)
        )
        class_output_nodes = tuple(() for _ in range(10))
        for bit, (zero_node, one_node) in enumerate(ecoc_output_nodes):
            for value, target in ((0, outputs[2 * bit]), (1, outputs[2 * bit + 1])):
                edges = connect_readout(
                    target, config.seed + bit * 131_071 + value
                )
                reward_ports.append(
                    builder.modulator(f"reward[bit{bit}][{value}]", targets=edges)
                )
    else:
        class_output_nodes = tuple(
            tuple(
                outputs.node_ids[
                    digit * config.outputs_per_class : (digit + 1)
                    * config.outputs_per_class
                ]
            )
            for digit in range(10)
        )
        for digit in range(10):
            start = digit * config.outputs_per_class
            stop = (digit + 1) * config.outputs_per_class
            edges = connect_readout(
                outputs[start:stop], config.seed + digit * 65_537
            )
            reward_ports.append(builder.modulator(f"reward[{digit}]", targets=edges))
    return CandidateNetwork(
        network=builder.build(),
        pixel_ports=pixel_ports,
        off_pixel_ports=off_pixel_ports,
        output_nodes=outputs.node_ids,
        class_output_nodes=class_output_nodes,
        pairwise_output_nodes=pairwise_output_nodes,
        ecoc_output_nodes=ecoc_output_nodes,
        ecoc_codes=ecoc_codes,
        pairwise_decode=config.pairwise_decode,
        pairwise_decode_temperature=config.pairwise_decode_temperature,
        pairwise_decode_alpha=config.pairwise_decode_alpha,
        probe_port=probe_port,
        reward_ports=tuple(reward_ports),
        shared_pool_reward_port=shared_pool_reward_port,
        shared_pool_edges=tuple(shared_pool_edges),
        readout_weight_scale=readout_weight_scale,
        feature_nodes=tuple(feature_nodes),
        competition_nodes=tuple(competition_nodes),
    )


def _presentations(
    candidate: CandidateNetwork,
    image: Sequence[object],
    start: float,
    end: float,
) -> tuple[ScalarInput, ...]:
    # Incremental intervals are open at their left edge.  Regular-rate
    # encoders may emit immediately, so keep every presentation strictly
    # inside the interval without changing any meaningful simulation timing.
    encoded_start = start + 1.0e-9
    def encoded_value(value):
        normalized = float(value) / 255.0
        if normalized <= candidate.network.metadata["config"]["input_threshold"]:
            return 0.0
        transformed = normalized ** candidate.network.metadata["config"]["input_gamma"]
        levels = candidate.network.metadata["config"]["input_levels"]
        if levels:
            transformed = round(transformed * (levels - 1)) / (levels - 1)
        return transformed

    on_inputs = tuple(
        ScalarInput(encoded_start, end, port, encoded_value(value))
        for port, value in zip(candidate.pixel_ports, image)
        if encoded_value(value) > 0.0
    )
    off_inputs = tuple(
        ScalarInput(encoded_start, end, port, 1.0 - float(value) / 255.0)
        for port, value in zip(candidate.off_pixel_ports, image)
        if value < 255
    )
    return (*on_inputs, *off_inputs)


def _evidence(spikes, output_classes, start, duration, bonus):
    positions = {
        node: class_index
        for class_index, nodes in enumerate(output_classes)
        for node in nodes
    }
    counts = [0] * len(output_classes)
    first = [None] * len(output_classes)
    for event in spikes:
        index = positions.get(event.node)
        if index is None:
            continue
        counts[index] += 1
        if first[index] is None:
            first[index] = event.t
    values = []
    for count, spike_time in zip(counts, first):
        latency = 0.0
        if spike_time is not None:
            phase = min(max((spike_time - start) / duration, 0.0), 1.0)
            latency = bonus * (1.0 - phase)
        values.append(count + latency)
    return tuple(values), sum(counts)


def _candidate_evidence(spikes, candidate, start, duration, bonus):
    if not candidate.pairwise_output_nodes and not candidate.ecoc_output_nodes:
        values, count = _evidence(
            spikes, candidate.class_output_nodes, start, duration, bonus
        )
        return values, count, None
    singleton_groups = tuple((node,) for node in candidate.output_nodes)
    individual, count = _evidence(
        spikes, singleton_groups, start, duration, bonus
    )
    by_node = dict(zip(candidate.output_nodes, individual))
    if candidate.ecoc_output_nodes:
        bit_differences = tuple(
            by_node[one_node] - by_node[zero_node]
            for zero_node, one_node in candidate.ecoc_output_nodes
        )
        values = tuple(
            sum(
                (1.0 if code[bit] else -1.0) * bit_differences[bit]
                for bit in range(len(bit_differences))
            )
            for code in candidate.ecoc_codes
        )
        return values, count, by_node
    votes = [0.0] * 10
    for first, second, first_node, second_node in candidate.pairwise_output_nodes:
        first_value = by_node[first_node]
        second_value = by_node[second_node]
        if candidate.pairwise_decode == "margin":
            difference = first_value - second_value
            votes[first] += difference
            votes[second] -= difference
        elif candidate.pairwise_decode == "sigmoid":
            scaled = (first_value - second_value) / (
                candidate.pairwise_decode_temperature
            )
            first_probability = 1.0 / (1.0 + math.exp(-scaled))
            votes[first] += first_probability
            votes[second] += 1.0 - first_probability
        else:
            if first_value > second_value:
                votes[first] += 1.0
            elif second_value > first_value:
                votes[second] += 1.0
            else:
                votes[first] += 0.5
                votes[second] += 0.5
            if candidate.pairwise_decode == "hybrid":
                correction = candidate.pairwise_decode_alpha * (
                    first_value - second_value
                )
                votes[first] += correction
                votes[second] -= correction
    return tuple(votes), count, by_node


def _probabilities(evidence, temperature):
    scaled = tuple(value / temperature for value in evidence)
    maximum = max(scaled)
    exponential = tuple(math.exp(value - maximum) for value in scaled)
    total = sum(exponential)
    return tuple(value / total for value in exponential)


def _shared_pool_reward(config, evidence, label, prediction):
    """Return the single task-level third factor for all feature kernels."""
    if config.shared_pool_reward_mode == "correctness":
        return (
            config.shared_pool_correct_reward
            if prediction == label
            else config.shared_pool_incorrect_reward
        )
    rival = max(
        (digit for digit in range(10) if digit != label),
        key=evidence.__getitem__,
    )
    margin = evidence[label] - evidence[rival]
    return math.tanh(margin / config.shared_pool_reward_scale)


def _metrics(correct, loss, output_spikes, silent, samples):
    return Metrics(
        samples=samples,
        accuracy=correct / samples,
        mean_output_spikes=output_spikes / samples,
        silent_fraction=silent / samples,
        mean_loss=loss / samples,
    )


def _combine_metrics(parts: Sequence[Metrics]) -> Metrics:
    samples = sum(item.samples for item in parts)
    return Metrics(
        samples=samples,
        accuracy=sum(item.accuracy * item.samples for item in parts) / samples,
        mean_output_spikes=(
            sum(item.mean_output_spikes * item.samples for item in parts) / samples
        ),
        silent_fraction=(
            sum(item.silent_fraction * item.samples for item in parts) / samples
        ),
        mean_loss=sum(item.mean_loss * item.samples for item in parts) / samples,
    )


def candidate_with_weights(
    candidate: CandidateNetwork, learned_weights
) -> CandidateNetwork:
    graph = replace(
        candidate.network.graph,
        edges=tuple(
            replace(edge, weight=learned_weights[edge.id])
            for edge in candidate.network.graph.edges
        ),
    )
    return replace(candidate, network=replace(candidate.network, graph=graph))


def _project_kernel_sum(
    values: Sequence[float],
    target_sum: float,
    *,
    lower: float,
    upper: float,
) -> tuple[float, ...]:
    """Rescale positive coefficients to an exact bounded L1 sum."""
    count = len(values)
    if count == 0:
        raise ValueError("cannot normalize an empty kernel")
    if target_sum < count * lower or target_sum > count * upper:
        raise ValueError(
            "kernel normalization target is outside the plasticity bounds"
        )
    result = [0.0] * count
    active = set(range(count))
    remaining = target_sum
    while active:
        total = sum(max(float(values[index]), 0.0) for index in active)
        proposals = {
            index: (
                remaining / len(active)
                if total == 0.0
                else max(float(values[index]), 0.0) * remaining / total
            )
            for index in active
        }
        below = {index for index, value in proposals.items() if value < lower}
        above = {index for index, value in proposals.items() if value > upper}
        fixed = below | above
        if not fixed:
            for index, value in proposals.items():
                result[index] = value
            break
        for index in below:
            result[index] = lower
            remaining -= lower
        for index in above:
            result[index] = upper
            remaining -= upper
        active -= fixed
    # Absorb floating-point residue without changing the mathematical target.
    residue = target_sum - sum(result)
    if residue:
        for index, value in enumerate(result):
            adjusted = value + residue
            if lower <= adjusted <= upper:
                result[index] = adjusted
                break
    return tuple(result)


def normalize_shared_pool_kernels(
    candidate: CandidateNetwork,
    config: CandidateConfig,
    learned_weights,
) -> tuple[float, ...]:
    """Experiment-only checkpoint normalization for shared convolution kernels.

    Spatial copies in a weight group are updated together.  This function
    normalizes the distinct coefficients of each output-channel kernel and
    writes that value back to every expanded edge in the group.
    """
    if not candidate.shared_pool_edges:
        raise ValueError("candidate has no shared-pool kernel to normalize")
    groups: dict[int, list[int]] = {}
    for edge_id in candidate.shared_pool_edges:
        group = candidate.network.graph.edges[edge_id].weight_group
        if group is None:
            raise ValueError("shared-pool edge is missing its weight group")
        groups.setdefault(group, []).append(edge_id)
    ordered_groups = sorted(groups)
    channels = config.shared_pool_channels
    if len(ordered_groups) % channels:
        raise ValueError("shared-pool groups cannot be divided into kernels")
    coefficients_per_kernel = len(ordered_groups) // channels
    target_sum = (
        config.shared_pool_normalization_sum
        or config.shared_pool_kernel_sum
        or coefficients_per_kernel
        * (config.shared_pool_weight_low + config.shared_pool_weight_high)
        / 2.0
    )
    normalized_weights = [float(value) for value in learned_weights]
    for channel in range(channels):
        kernel_groups = ordered_groups[
            channel * coefficients_per_kernel :
            (channel + 1) * coefficients_per_kernel
        ]
        values = [normalized_weights[groups[group][0]] for group in kernel_groups]
        normalized = _project_kernel_sum(
            values,
            target_sum,
            lower=0.001,
            upper=config.shared_pool_weight_bound,
        )
        for group, value in zip(kernel_groups, normalized):
            for edge_id in groups[group]:
                normalized_weights[edge_id] = value
    return tuple(normalized_weights)


def train_candidate(
    core: CoreEvaluator,
    candidate: CandidateNetwork,
    config: CandidateConfig,
    images,
    labels,
) -> tuple[object, Metrics]:
    graph = candidate.network.graph
    resolved = graph.resolve()
    samples = len(labels)
    correct = 0
    loss = 0.0
    output_spikes = 0
    silent = 0
    final = None
    with resolved.compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=config.sample_duration * samples,
            encoder_seed=config.seed,
            queue_capacity=config.event_queue_capacity,
            output_capacity=config.output_capacity,
            encoder_spike_capacity=config.encoder_spike_capacity,
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
                    scalar_inputs=_presentations(
                        candidate, image, start, presentation_end
                    ),
                )
                evidence, spike_count, node_evidence = _candidate_evidence(
                    decision.core.spikes,
                    candidate,
                    start,
                    decision_end - start,
                    config.first_spike_bonus,
                )
                probabilities = _probabilities(evidence, config.temperature)
                prediction = max(range(10), key=evidence.__getitem__)
                correct += int(prediction == label)
                loss -= math.log(max(probabilities[label], 1.0e-300))
                output_spikes += spike_count
                silent += int(spike_count == 0)
                if candidate.ecoc_output_nodes:
                    assert node_evidence is not None
                    bit_rewards = []
                    target_code = candidate.ecoc_codes[label]
                    for bit, (zero_node, one_node) in enumerate(
                        candidate.ecoc_output_nodes
                    ):
                        values = (
                            node_evidence[zero_node],
                            node_evidence[one_node],
                        )
                        target_side = target_code[bit]
                        rival_side = 1 - target_side
                        if config.reward_mode == "softmax":
                            bit_probability = _probabilities(
                                values, config.temperature
                            )
                            bit_rewards.extend(
                                (1.0 if side == target_side else 0.0)
                                - bit_probability[side]
                                for side in range(2)
                            )
                        else:
                            needs_update = values[target_side] <= values[rival_side]
                            if config.reward_mode == "margin":
                                needs_update = (
                                    values[target_side]
                                    < values[rival_side] + config.reward_margin
                                )
                            bit_rewards.extend(
                                (
                                    1.0 if side == target_side else -1.0
                                    for side in range(2)
                                )
                                if needs_update
                                else (0.0, 0.0)
                            )
                    rewards = tuple(bit_rewards)
                elif candidate.pairwise_output_nodes:
                    assert node_evidence is not None
                    pair_rewards = []
                    for first, second, first_node, second_node in (
                        candidate.pairwise_output_nodes
                    ):
                        if label not in (first, second):
                            pair_rewards.extend((0.0, 0.0))
                            continue
                        values = (
                            node_evidence[first_node],
                            node_evidence[second_node],
                        )
                        target_side = 0 if label == first else 1
                        rival_side = 1 - target_side
                        if config.reward_mode == "softmax":
                            pair_probability = _probabilities(
                                values, config.temperature
                            )
                            pair_rewards.extend(
                                (1.0 if side == target_side else 0.0)
                                - pair_probability[side]
                                for side in range(2)
                            )
                        else:
                            needs_update = values[target_side] <= values[rival_side]
                            if config.reward_mode == "margin":
                                needs_update = (
                                    values[target_side]
                                    < values[rival_side] + config.reward_margin
                                )
                            pair_rewards.extend(
                                (
                                    1.0 if side == target_side else -1.0
                                    for side in range(2)
                                )
                                if needs_update
                                else (0.0, 0.0)
                            )
                    rewards = tuple(pair_rewards)
                elif config.reward_mode in {"perceptron", "margin"}:
                    rewards = [0.0] * 10
                    rival = max(
                        (digit for digit in range(10) if digit != label),
                        key=evidence.__getitem__,
                    )
                    needs_update = prediction != label or (
                        config.reward_mode == "margin"
                        and evidence[label]
                        < evidence[rival] + config.reward_margin
                    )
                    if needs_update:
                        rewards[label] = 1.0
                        rewards[rival] = -1.0
                    rewards = tuple(rewards)
                else:
                    rewards = tuple(
                        (1.0 if digit == label else 0.0) - probabilities[digit]
                        for digit in range(10)
                    )
                has_modulated_edges = bool(candidate.reward_ports) or (
                    candidate.shared_pool_reward_port is not None
                )
                spike_inputs = (
                    (SpikeInput(probe_time, candidate.probe_port, 20.0),)
                    if has_modulated_edges
                    else ()
                )
                modulation_inputs = tuple(
                    ModulationInput(reward_time, port, reward)
                    for port, reward in zip(candidate.reward_ports, rewards)
                )
                if candidate.shared_pool_reward_port is not None:
                    modulation_inputs += (
                        ModulationInput(
                            reward_time,
                            candidate.shared_pool_reward_port,
                            _shared_pool_reward(
                                config,
                                evidence,
                                label,
                                prediction,
                            ),
                        ),
                    )
                final = (
                    run.finish(
                        spike_inputs=spike_inputs,
                        modulation_inputs=modulation_inputs,
                    )
                    if sample + 1 == samples
                    else run.advance_until(
                        sample_end,
                        spike_inputs=spike_inputs,
                        modulation_inputs=modulation_inputs,
                    )
                )
    assert final is not None
    return final.core.weights, _metrics(
        correct, loss, output_spikes, silent, samples
    )


def evaluate_candidate(
    core: CoreEvaluator,
    candidate: CandidateNetwork,
    config: CandidateConfig,
    learned_weights,
    images,
    labels,
) -> Metrics:
    frozen_graph = replace(
        candidate.network.graph,
        edges=tuple(
            replace(edge, weight=learned_weights[edge.id], plasticity=None)
            for edge in candidate.network.graph.edges
        ),
        modulator_ports=(),
    )
    resolved = frozen_graph.resolve()
    samples = len(labels)
    correct = 0
    loss = 0.0
    output_spikes = 0
    silent = 0
    with resolved.compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=config.sample_duration * samples,
            encoder_seed=config.seed + 1_000_003,
            queue_capacity=config.event_queue_capacity,
            output_capacity=config.output_capacity,
            encoder_spike_capacity=config.encoder_spike_capacity,
        ) as run:
            for sample, (image, raw_label) in enumerate(zip(images, labels)):
                label = int(raw_label)
                start = sample * config.sample_duration
                presentation_end = start + config.presentation
                sample_end = start + config.sample_duration
                result = (
                    run.finish(
                        scalar_inputs=_presentations(
                            candidate, image, start, presentation_end
                        )
                    )
                    if sample + 1 == samples
                    else run.advance_until(
                        sample_end,
                        scalar_inputs=_presentations(
                            candidate, image, start, presentation_end
                        ),
                    )
                )
                evidence, spike_count, _ = _candidate_evidence(
                    result.core.spikes,
                    candidate,
                    start,
                    config.presentation + config.settling,
                    config.first_spike_bonus,
                )
                probabilities = _probabilities(evidence, config.temperature)
                prediction = max(range(10), key=evidence.__getitem__)
                correct += int(prediction == label)
                loss -= math.log(max(probabilities[label], 1.0e-300))
                output_spikes += spike_count
                silent += int(spike_count == 0)
    return _metrics(correct, loss, output_spikes, silent, samples)


def balanced_indices(labels, per_class, *, offset=0, seed=0):
    selected = []
    labels = np.asarray(labels)
    for digit in range(10):
        indices = np.flatnonzero(labels == digit)
        selected.extend(indices[offset : offset + per_class])
    generator = np.random.default_rng(seed)
    result = np.asarray(selected, dtype=np.int64)
    generator.shuffle(result)
    return result


def base_candidates(seed: int) -> list[CandidateConfig]:
    base = dict(
        presentation=25.0,
        settling=5.0,
        sample_duration=80.0,
        max_rate=0.20,
        ttfs_max_latency=20.0,
        silence_threshold=0.05,
        pool_weight=4.0,
        pool_threshold=-55.0,
        output_threshold=-55.0,
        readout_low=0.04,
        readout_high=0.08,
        readout_bound=0.15,
        learning_rate=0.002,
        trace_tau=20.0,
        temperature=1.0,
        first_spike_bonus=0.5,
    )
    handcrafted = [
        CandidateConfig("direct", "poisson", seed=seed, **base),
        CandidateConfig(
            "direct",
            "poisson",
            seed=seed + 1,
            **{**base, "output_threshold": -57.0, "learning_rate": 0.005},
        ),
        CandidateConfig(
            "direct",
            "ttfs",
            seed=seed + 2,
            **{
                **base,
                "output_threshold": -60.0,
                "readout_low": 0.015,
                "readout_high": 0.04,
                "learning_rate": 0.005,
            },
        ),
        CandidateConfig(
            "pool2",
            "ttfs",
            seed=seed + 3,
            **{
                **base,
                "pool_weight": 4.0,
                "pool_threshold": -57.0,
                "output_threshold": -62.0,
                "readout_low": 0.04,
                "readout_high": 0.10,
                "readout_bound": 0.30,
                "learning_rate": 0.005,
            },
        ),
        CandidateConfig(
            "pool2",
            "poisson",
            seed=seed + 4,
            **{
                **base,
                "pool_weight": 5.0,
                "pool_threshold": -58.0,
                "output_threshold": -62.0,
                "readout_low": 0.04,
                "readout_high": 0.12,
                "readout_bound": 0.30,
                "learning_rate": 0.005,
            },
        ),
    ]
    generator = random.Random(seed)
    while len(handcrafted) < 12:
        architecture = generator.choice(("direct", "pool2"))
        encoder = generator.choice(("poisson", "ttfs"))
        if architecture == "direct":
            low, high, bound = generator.choice(
                ((0.015, 0.05, 0.15), (0.03, 0.08, 0.20), (0.04, 0.10, 0.25))
            )
            output_threshold = generator.choice((-60.0, -57.0, -55.0))
        else:
            low, high, bound = generator.choice(
                ((0.03, 0.09, 0.25), (0.05, 0.12, 0.30), (0.08, 0.18, 0.40))
            )
            output_threshold = generator.choice((-63.0, -62.0, -60.0))
        handcrafted.append(
            CandidateConfig(
                architecture=architecture,
                encoder=encoder,
                seed=seed + len(handcrafted),
                **{
                    **base,
                    "pool_weight": generator.choice((3.0, 4.0, 5.0, 6.0)),
                    "pool_threshold": generator.choice((-60.0, -58.0, -55.0)),
                    "output_threshold": output_threshold,
                    "readout_low": low,
                    "readout_high": high,
                    "readout_bound": bound,
                    "learning_rate": generator.choice((0.002, 0.005, 0.01)),
                    "trace_tau": generator.choice((10.0, 20.0, 30.0)),
                    "temperature": generator.choice((0.5, 1.0, 2.0)),
                    "first_spike_bonus": generator.choice((0.0, 0.5, 1.0)),
                    "silence_threshold": generator.choice((0.0, 0.05, 0.1)),
                },
            )
        )
    return handcrafted


def run_trial(
    core,
    config,
    train_x,
    train_y,
    validation_x,
    validation_y,
    *,
    epochs=1,
    evaluation_thresholds=(),
    learning_rate_schedule=(),
    initial_weights=None,
    initial_metrics=(),
    completed_training_blocks=0,
    elapsed_seconds=0.0,
    checkpoint_callback=None,
):
    started = time.perf_counter()
    candidate = build_candidate(config)
    epoch_metrics = list(initial_metrics)
    weights = (
        None
        if initial_weights is None
        else tuple(float(value) for value in initial_weights)
    )
    if weights is not None:
        candidate = candidate_with_weights(candidate, weights)
    if completed_training_blocks < 0:
        raise ValueError("completed_training_blocks cannot be negative")
    if len(epoch_metrics) != completed_training_blocks:
        raise ValueError(
            "initial_metrics must contain one entry per completed training block"
        )
    generator = np.random.default_rng(config.seed)
    schedule = tuple(learning_rate_schedule or ())
    normalization_interval = config.shared_pool_normalization_interval
    training_block = 0

    def train_indices(block_config, indices):
        nonlocal candidate, weights, training_block
        blocks = (
            tuple(
                indices[start : start + normalization_interval]
                for start in range(0, len(indices), normalization_interval)
            )
            if normalization_interval
            else (indices,)
        )
        for block in blocks:
            if training_block < completed_training_blocks:
                training_block += 1
                continue
            current_config = replace(
                block_config,
                seed=block_config.seed + training_block * 104_729,
            )
            current_candidate = build_candidate(current_config)
            if weights is not None:
                current_candidate = candidate_with_weights(
                    current_candidate, weights
                )
            weights, train = train_candidate(
                core,
                current_candidate,
                current_config,
                train_x[block],
                train_y[block],
            )
            if normalization_interval:
                weights = normalize_shared_pool_kernels(
                    current_candidate, current_config, weights
                )
            candidate = candidate_with_weights(current_candidate, weights)
            epoch_metrics.append(train)
            training_block += 1
            if checkpoint_callback is not None:
                checkpoint_callback(
                    weights,
                    tuple(epoch_metrics),
                    training_block,
                    elapsed_seconds + time.perf_counter() - started,
                )

    if schedule:
        order = generator.permutation(len(train_y))
        chunks = np.array_split(order, len(schedule))
        for phase, (learning_rate, chunk) in enumerate(zip(schedule, chunks)):
            pool_rate = config.shared_pool_learning_rate
            if config.learning_rate > 0.0:
                pool_rate *= learning_rate / config.learning_rate
            phase_config = replace(
                config,
                learning_rate=learning_rate,
                shared_pool_learning_rate=pool_rate,
                seed=config.seed + phase * 1_000_003,
            )
            train_indices(phase_config, chunk)
    else:
        for epoch in range(epochs):
            order = generator.permutation(len(train_y))
            epoch_config = replace(config, seed=config.seed + epoch * 1_000_003)
            train_indices(epoch_config, order)
    assert weights is not None
    thresholds = tuple(evaluation_thresholds or ()) or (config.output_threshold,)
    validations = {}
    for threshold in thresholds:
        evaluation_config = replace(config, output_threshold=threshold)
        evaluation_candidate = candidate_with_weights(
            build_candidate(evaluation_config), weights
        )
        validations[threshold] = evaluate_candidate(
            core,
            evaluation_candidate,
            evaluation_config,
            weights,
            validation_x,
            validation_y,
        )
    evaluation_threshold = max(
        thresholds, key=lambda threshold: validations[threshold].accuracy
    )
    validation = validations[evaluation_threshold]
    plastic_edge_ids = tuple(
        edge.id
        for edge in candidate.network.graph.edges
        if edge.plasticity is not None and edge.id not in candidate.shared_pool_edges
    )
    plastic_weights = np.asarray(
        [weights[edge_id] for edge_id in plastic_edge_ids], dtype=float
    )
    upper = config.readout_bound * candidate.readout_weight_scale
    lower = upper * 1.0e-6 + 0.001
    shared_kernel_weights = []
    seen_groups = set()
    for edge_id in candidate.shared_pool_edges:
        edge = candidate.network.graph.edges[edge_id]
        if edge.weight_group in seen_groups:
            continue
        seen_groups.add(edge.weight_group)
        shared_kernel_weights.append(float(weights[edge_id]))
    report = candidate.network.validate()
    return TrialResult(
        config=config,
        train=_combine_metrics(epoch_metrics),
        validation=validation,
        evaluation_threshold=evaluation_threshold,
        validation_by_threshold={
            str(threshold): validations[threshold] for threshold in thresholds
        },
        learning_rate_schedule=schedule,
        weight_mean=float(plastic_weights.mean()),
        weight_at_lower_bound=float(np.mean(plastic_weights <= lower)),
        weight_at_upper_bound=float(np.mean(plastic_weights >= upper - 1.0e-9)),
        learned_weights=tuple(float(weight) for weight in weights),
        shared_kernel_weights=tuple(shared_kernel_weights),
        weight_parameter_count=report.weight_parameter_count,
        shared_weight_group_count=report.shared_weight_group_count,
        readout_weight_scale=candidate.readout_weight_scale,
        seconds=elapsed_seconds + time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="build/liblacuna_core.dylib")
    parser.add_argument("--data-dir", default="artifacts/mnist")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--train-per-class", type=int, default=20)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--validation-per-class", type=int, default=20)
    parser.add_argument("--validation-offset", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate-schedule", type=float, nargs="*")
    parser.add_argument("--outputs-per-class", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output-threshold", type=float)
    parser.add_argument("--evaluation-thresholds", type=float, nargs="*")
    parser.add_argument("--max-rate", type=float)
    parser.add_argument("--first-spike-bonus", type=float)
    parser.add_argument(
        "--architecture",
        choices=("direct", "multiscale", "onoff", "pool2", "local1", "local2"),
    )
    parser.add_argument("--pool-threshold", type=float)
    parser.add_argument("--pool-weight", type=float)
    parser.add_argument("--readout-low", type=float)
    parser.add_argument("--readout-high", type=float)
    parser.add_argument("--readout-bound", type=float)
    parser.add_argument("--local-weight-low", type=float)
    parser.add_argument("--local-weight-high", type=float)
    parser.add_argument("--local-inhibition", type=float)
    parser.add_argument(
        "--encoder", choices=("burst", "poisson", "regular", "ttfs")
    )
    parser.add_argument("--trace-tau", type=float)
    parser.add_argument("--silence-threshold", type=float)
    parser.add_argument(
        "--reward-mode", choices=("softmax", "perceptron", "margin")
    )
    parser.add_argument("--output-inhibition", type=float)
    parser.add_argument("--reward-margin", type=float)
    parser.add_argument(
        "--readout-mode", choices=("multiclass", "pairwise", "ecoc")
    )
    parser.add_argument("--ecoc-bits", type=int)
    parser.add_argument("--output-tau-m", type=float)
    parser.add_argument("--pool-kernel", type=int)
    parser.add_argument("--pool-side", type=int)
    parser.add_argument("--secondary-pool-weight", type=float)
    parser.add_argument("--input-gamma", type=float)
    parser.add_argument("--input-threshold", type=float)
    parser.add_argument("--input-levels", type=int)
    parser.add_argument("--shared-pool", action="store_true", default=None)
    parser.add_argument("--shared-pool-channels", type=int)
    parser.add_argument(
        "--shared-pool-plasticity", choices=("static", "pair", "modulated")
    )
    parser.add_argument("--shared-pool-learning-rate", type=float)
    parser.add_argument("--shared-pool-trace-tau", type=float)
    parser.add_argument("--shared-pool-eligibility-tau-plus", type=float)
    parser.add_argument("--shared-pool-eligibility-tau-minus", type=float)
    parser.add_argument(
        "--shared-pool-reward-mode", choices=("correctness", "signed_margin")
    )
    parser.add_argument("--shared-pool-reward-scale", type=float)
    parser.add_argument("--shared-pool-correct-reward", type=float)
    parser.add_argument("--shared-pool-incorrect-reward", type=float)
    parser.add_argument("--shared-pool-weight-low", type=float)
    parser.add_argument("--shared-pool-weight-high", type=float)
    parser.add_argument("--shared-pool-weight-bound", type=float)
    parser.add_argument(
        "--shared-pool-competition",
        choices=("none", "shared_relay", "winner_specific"),
    )
    parser.add_argument("--shared-pool-inhibition-delay", type=float)
    parser.add_argument("--shared-pool-adaptive", action="store_true", default=None)
    parser.add_argument("--shared-pool-tau-adaptation", type=float)
    parser.add_argument("--shared-pool-adaptation-increment", type=float)
    parser.add_argument(
        "--shared-pool-normalize-initial-kernels",
        action="store_true",
        default=None,
    )
    parser.add_argument("--shared-pool-kernel-sum", type=float)
    parser.add_argument("--shared-pool-readout-scale", type=float)
    parser.add_argument("--shared-pool-stride", type=int)
    parser.add_argument("--shared-pool-normalization-interval", type=int)
    parser.add_argument("--shared-pool-normalization-sum", type=float)
    parser.add_argument("--event-queue-capacity", type=int)
    parser.add_argument("--output-capacity", type=int)
    parser.add_argument("--encoder-spike-capacity", type=int)
    parser.add_argument("--readout-eligibility-tau-plus", type=float)
    parser.add_argument("--readout-eligibility-tau-minus", type=float)
    parser.add_argument(
        "--pairwise-decode", choices=("vote", "margin", "sigmoid", "hybrid")
    )
    parser.add_argument("--pairwise-decode-temperature", type=float)
    parser.add_argument("--pairwise-decode-alpha", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--configs-from",
        help="reuse the ranked candidate configurations from an earlier result",
    )
    parser.add_argument(
        "--output", default="artifacts/optimization/mnist_search_stage1.json"
    )
    args = parser.parse_args()
    train_images, train_labels, _, _ = load_mnist(args.data_dir, download=False)
    train_indices = balanced_indices(
        train_labels,
        args.train_per_class,
        offset=args.train_offset,
        seed=args.seed,
    )
    validation_offset = (
        args.train_offset + args.train_per_class
        if args.validation_offset is None
        else args.validation_offset
    )
    validation_indices = balanced_indices(
        train_labels,
        args.validation_per_class,
        offset=validation_offset,
        seed=args.seed + 1,
    )
    core = CoreEvaluator(args.library)
    if args.configs_from:
        ranked = json.loads(Path(args.configs_from).read_text(encoding="utf-8"))
        candidates = [CandidateConfig(**item["config"]) for item in ranked]
    else:
        candidates = base_candidates(args.seed)
    overrides = {
        name: value
        for name, value in (
            ("outputs_per_class", args.outputs_per_class),
            ("learning_rate", args.learning_rate),
            ("temperature", args.temperature),
            ("output_threshold", args.output_threshold),
            ("max_rate", args.max_rate),
            ("first_spike_bonus", args.first_spike_bonus),
            ("architecture", args.architecture),
            ("pool_threshold", args.pool_threshold),
            ("pool_weight", args.pool_weight),
            ("readout_low", args.readout_low),
            ("readout_high", args.readout_high),
            ("readout_bound", args.readout_bound),
            ("local_weight_low", args.local_weight_low),
            ("local_weight_high", args.local_weight_high),
            ("local_inhibition", args.local_inhibition),
            ("encoder", args.encoder),
            ("trace_tau", args.trace_tau),
            ("silence_threshold", args.silence_threshold),
            ("reward_mode", args.reward_mode),
            ("output_inhibition", args.output_inhibition),
            ("reward_margin", args.reward_margin),
            ("readout_mode", args.readout_mode),
            ("ecoc_bits", args.ecoc_bits),
            ("output_tau_m", args.output_tau_m),
            ("pool_kernel", args.pool_kernel),
            ("pool_side", args.pool_side),
            ("secondary_pool_weight", args.secondary_pool_weight),
            ("input_gamma", args.input_gamma),
            ("input_threshold", args.input_threshold),
            ("input_levels", args.input_levels),
            ("shared_pool", args.shared_pool),
            ("shared_pool_channels", args.shared_pool_channels),
            ("shared_pool_plasticity", args.shared_pool_plasticity),
            ("shared_pool_learning_rate", args.shared_pool_learning_rate),
            ("shared_pool_trace_tau", args.shared_pool_trace_tau),
            (
                "shared_pool_eligibility_tau_plus",
                args.shared_pool_eligibility_tau_plus,
            ),
            (
                "shared_pool_eligibility_tau_minus",
                args.shared_pool_eligibility_tau_minus,
            ),
            ("shared_pool_reward_mode", args.shared_pool_reward_mode),
            ("shared_pool_reward_scale", args.shared_pool_reward_scale),
            ("shared_pool_correct_reward", args.shared_pool_correct_reward),
            ("shared_pool_incorrect_reward", args.shared_pool_incorrect_reward),
            ("shared_pool_weight_low", args.shared_pool_weight_low),
            ("shared_pool_weight_high", args.shared_pool_weight_high),
            ("shared_pool_weight_bound", args.shared_pool_weight_bound),
            ("shared_pool_competition", args.shared_pool_competition),
            ("shared_pool_inhibition_delay", args.shared_pool_inhibition_delay),
            ("shared_pool_adaptive", args.shared_pool_adaptive),
            ("shared_pool_tau_adaptation", args.shared_pool_tau_adaptation),
            (
                "shared_pool_adaptation_increment",
                args.shared_pool_adaptation_increment,
            ),
            (
                "shared_pool_normalize_initial_kernels",
                args.shared_pool_normalize_initial_kernels,
            ),
            ("shared_pool_kernel_sum", args.shared_pool_kernel_sum),
            ("shared_pool_readout_scale", args.shared_pool_readout_scale),
            ("shared_pool_stride", args.shared_pool_stride),
            (
                "shared_pool_normalization_interval",
                args.shared_pool_normalization_interval,
            ),
            (
                "shared_pool_normalization_sum",
                args.shared_pool_normalization_sum,
            ),
            ("event_queue_capacity", args.event_queue_capacity),
            ("output_capacity", args.output_capacity),
            ("encoder_spike_capacity", args.encoder_spike_capacity),
            (
                "readout_eligibility_tau_plus",
                args.readout_eligibility_tau_plus,
            ),
            (
                "readout_eligibility_tau_minus",
                args.readout_eligibility_tau_minus,
            ),
            ("pairwise_decode", args.pairwise_decode),
            (
                "pairwise_decode_temperature",
                args.pairwise_decode_temperature,
            ),
            ("pairwise_decode_alpha", args.pairwise_decode_alpha),
        )
        if value is not None
    }
    if overrides:
        candidates = [replace(candidate, **overrides) for candidate in candidates]
    results = []
    candidates = candidates[: args.trials]
    for index, config in enumerate(candidates, start=1):
        print(
            f"trial {index}/{len(candidates)}: {config.architecture}/{config.encoder} "
            f"lr={config.learning_rate} threshold={config.output_threshold} "
            f"shared_pool={config.shared_pool} "
            f"pool_rule={config.shared_pool_plasticity}",
            flush=True,
        )
        result = run_trial(
            core,
            config,
            train_images[train_indices],
            train_labels[train_indices],
            train_images[validation_indices],
            train_labels[validation_indices],
            epochs=args.epochs,
            evaluation_thresholds=args.evaluation_thresholds,
            learning_rate_schedule=args.learning_rate_schedule,
        )
        results.append(result)
        print(
            f"  train={100 * result.train.accuracy:.1f}% "
            f"validation={100 * result.validation.accuracy:.1f}% "
            f"spikes={result.validation.mean_output_spikes:.2f} "
            f"silent={100 * result.validation.silent_fraction:.1f}% "
            f"eval_threshold={result.evaluation_threshold:g} "
            f"weights={result.weight_mean:.4f} "
            f"parameters={result.weight_parameter_count} "
            f"bounds=({100 * result.weight_at_lower_bound:.1f}%,"
            f"{100 * result.weight_at_upper_bound:.1f}%) "
            f"seconds={result.seconds:.1f}",
            flush=True,
        )
        if len(result.validation_by_threshold) > 1:
            print(
                "  evaluation grid: "
                + ", ".join(
                    f"{threshold}={100 * metrics.accuracy:.1f}%"
                    for threshold, metrics in result.validation_by_threshold.items()
                ),
                flush=True,
            )
    results.sort(key=lambda item: item.validation.accuracy, reverse=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(item) for item in results], indent=2),
        encoding="utf-8",
    )
    print("ranking:", flush=True)
    for rank, result in enumerate(results, start=1):
        print(
            f"  {rank:>2}. {100 * result.validation.accuracy:5.1f}% "
            f"{result.config.architecture}/{result.config.encoder} "
            f"lr={result.config.learning_rate}",
            flush=True,
        )


if __name__ == "__main__":
    main()
