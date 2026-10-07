from dataclasses import replace

import pytest

from lacuna import ModulatedSTDP, PairSTDP

from optimization.mnist_alternating_cnn import (
    build_alternating_cnn,
    default_config,
    normalize_convolution_kernels,
)
from optimization.mnist_neuron_search import candidate_with_weights


def _small_config(mode: str):
    return replace(
        default_config(41),
        plasticity_mode=mode,
        cnn_channels=(2, 2, 2, 2),
        cnn_kernels=(7, 3, 1, 1),
        cnn_strides=(4, 2, 1, 1),
        cnn_paddings=(0, 0, 0, 0),
        cnn_thresholds=(-60.0, -60.0, -59.0, -58.0),
        shared_pool_channels=2,
        convolution_inhibition=0.0,
    )


def test_pair_and_reward_passes_have_identical_weight_topology() -> None:
    pair = build_alternating_cnn(_small_config("pair"))
    reward = build_alternating_cnn(_small_config("modulated"))

    pair_signature = tuple(
        (edge.pre, edge.post, edge.weight, edge.weight_group)
        for edge in pair.network.graph.edges
    )
    reward_signature = tuple(
        (edge.pre, edge.post, edge.weight, edge.weight_group)
        for edge in reward.network.graph.edges
    )
    assert pair_signature == reward_signature
    assert pair.stage_edges == reward.stage_edges
    assert pair.feedforward_edges == reward.feedforward_edges
    assert pair.stage_nodes == reward.stage_nodes


def test_whole_feedforward_graph_switches_plasticity_rule() -> None:
    pair = build_alternating_cnn(_small_config("pair"))
    reward = build_alternating_cnn(_small_config("modulated"))

    assert all(
        isinstance(pair.network.graph.edges[edge_id].plasticity, PairSTDP)
        for edge_id in pair.feedforward_edges
    )
    assert all(
        isinstance(reward.network.graph.edges[edge_id].plasticity, ModulatedSTDP)
        for edge_id in reward.feedforward_edges
    )
    assert pair.reward_ports == ()
    assert pair.shared_pool_reward_port is None
    assert len(reward.reward_ports) == 90
    feature_port = next(
        port
        for port in reward.network.graph.modulator_ports
        if port.id == reward.shared_pool_reward_port
    )
    assert feature_port.edges == reward.shared_pool_edges


def test_weight_transfer_and_per_stage_normalization_are_exact() -> None:
    config = _small_config("pair")
    pair = build_alternating_cnn(config)
    weights = tuple(float(edge.weight) for edge in pair.network.graph.edges)
    normalized = normalize_convolution_kernels(pair, config, weights)
    reward = candidate_with_weights(
        build_alternating_cnn(replace(config, plasticity_mode="modulated")),
        normalized,
    )

    assert tuple(edge.weight for edge in reward.network.graph.edges) == normalized
    input_channels = 1
    for edges, output_channels in zip(pair.stage_edges, config.cnn_channels):
        groups = {}
        for edge_id in edges:
            group = pair.network.graph.edges[edge_id].weight_group
            groups.setdefault(group, []).append(edge_id)
        ordered = sorted(groups)
        coefficients = len(ordered) // output_channels
        expected = config.convolution_kernel_sum_per_input_channel * input_channels
        for channel in range(output_channels):
            channel_groups = ordered[
                channel * coefficients : (channel + 1) * coefficients
            ]
            total = sum(normalized[groups[group][0]] for group in channel_groups)
            assert total == pytest.approx(expected)
        input_channels = output_channels
