"""Export a restricted official SLAYER model to the ordinary Lacuna engine.

This module does not implement a neuron, loss, gradient, or optimizer. Training
stays in Lava-DL. Its fixed-point state rounding is not reproduced by Lacuna.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from pathlib import Path

from .dense_lif import DenseLIFDeployment, DenseLIFLayer, build_dense_lif
from .feedforward_lif import (
    Conv2dLIFLayer,
    FeedforwardLIFDeployment,
    build_feedforward_lif,
    normalize_input_shape,
)


@dataclass(frozen=True)
class SlayerDenseImport:
    """A static deployment and the source contract used to construct it."""

    deployment: DenseLIFDeployment
    timestep: float
    source_fingerprint: str
    network_sha256: str
    metadata: dict


@dataclass(frozen=True)
class SlayerFeedforwardImport:
    """A spatial or dense deployment with its source and geometry contract."""

    deployment: FeedforwardLIFDeployment
    timestep: float
    source_fingerprint: str
    network_sha256: str
    metadata: dict


def _validate_pool_geometry(synapse, shape):
    """Keep channel boundaries aligned with upstream's folded spatial axis."""

    kernel = tuple(synapse.weight.shape[2:4])
    if (
        tuple(synapse.weight.shape[:2]) != (1, 1)
        or synapse.groups != 1
        or synapse.in_channels != 1
        or synapse.out_channels != 1
    ):
        raise ValueError("Pool requires a single shared spatial kernel")
    if synapse.weight.requires_grad:
        raise ValueError("Pool kernel weights must be fixed")
    if (
        min(kernel) <= 0
        or tuple(synapse.stride[:2]) != kernel
        or tuple(synapse.padding[:2]) != (0, 0)
        or tuple(synapse.dilation[:2]) != (1, 1)
    ):
        raise ValueError(
            "Pool supports only nonoverlapping kernel-sized strides, "
            "zero padding, and unit dilation"
        )
    if any(size % width for size, width in zip(shape[1:], kernel)):
        raise ValueError(
            "Pool spatial dimensions must be divisible by its kernel"
        )


def _source_description(model, timestep, *, input_shape=None):
    import torch
    from lava.lib.dl import slayer

    if type(model) is not torch.nn.Sequential or not len(model):
        raise ValueError(
            "expected a nonempty torch.nn.Sequential of supported CUBA blocks"
        )
    if any(module.training for module in model.modules()):
        raise ValueError("call model.eval() before exporting")
    for module in model.modules():
        if "forward" in vars(module):
            raise ValueError("instance forward overrides are unsupported")
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError(
                "forward hooks are outside the supported contract"
            )

    layers = []
    parameters = []
    shape = input_shape
    source_blocks = []
    for index, block in enumerate(model):
        if (
            input_shape is not None
            and type(block) is slayer.block.cuba.Flatten
        ):
            if len(shape) != 3 or block.count_log or index == len(model) - 1:
                raise ValueError(
                    "Flatten must map spatial spikes to a following Dense block"
                )
            if type(model[index + 1]) is not slayer.block.cuba.Dense:
                raise ValueError("Flatten must be followed by a Dense block")
            shape = (math.prod(shape),)
            continue
        convolution = (
            input_shape is not None and type(block) is slayer.block.cuba.Conv
        )
        pooling = (
            input_shape is not None and type(block) is slayer.block.cuba.Pool
        )
        spatial = convolution or pooling
        if not spatial and type(block) is not slayer.block.cuba.Dense:
            raise ValueError(
                f"layer {index} is not an official CUBA Dense block"
                + (
                    " or supported Conv/Pool/Flatten block"
                    if input_shape is not None
                    else ""
                )
            )
        neuron, synapse = block.neuron, block.synapse
        if type(neuron) is not slayer.neuron.cuba.Neuron:
            raise ValueError("only the official CUBA neuron is supported")
        expected_synapse = slayer.synapse.Dense
        if convolution:
            expected_synapse = slayer.synapse.Conv
        elif pooling:
            expected_synapse = slayer.synapse.Pool
        if type(synapse) is not expected_synapse:
            raise ValueError(
                "only the corresponding official real synapse is supported"
            )
        if block.delay is not None or block.delay_shift is not False:
            raise ValueError(
                "use delay=False and delay_shift=False for this profile"
            )
        if getattr(block, "mask", None) is not None or block.count_log:
            raise ValueError(
                "masks and count_log are not supported by this importer"
            )
        if synapse.pre_hook_fx is not None or synapse.weight_norm_enabled:
            raise ValueError(
                "use pre_hook_fx=None and weight_norm=False explicitly"
            )
        if synapse.bias is not None or synapse.complex:
            raise ValueError("bias and complex weights are not supported")
        if (
            neuron.norm is not None
            or neuron.drop is not None
            or neuron.persistent_state
            or neuron.graded_spike
            or not neuron.shared_param
            or neuron.requires_grad
        ):
            raise ValueError(
                "use fixed shared neurons without normalization or dropout"
            )
        if (
            neuron.p_scale != 4096
            or not 0 < neuron.s_scale <= 2**30
            or neuron.s_scale != 64 * neuron.w_scale
        ):
            raise ValueError(
                "unsupported source decay or state representation"
            )
        if synapse.weight.dtype is not torch.float32:
            raise ValueError(
                "the validated source profile uses float32 weights"
            )
        if synapse.weight.device.type != "cpu":
            raise ValueError(
                "the first validated source profile uses CPU inference"
            )
        if synapse.weight.ndim != 5:
            raise ValueError(
                "source synaptic weights must have five dimensions"
            )
        if not spatial and tuple(synapse.weight.shape[2:]) != (
            1,
            1,
            1,
        ):
            raise ValueError("only flat dense layers are supported")
        if not spatial and (
            synapse.groups != 1
            or tuple(synapse.stride) != (1, 1, 1)
            or tuple(synapse.padding) != (0, 0, 0)
            or tuple(synapse.dilation) != (1, 1, 1)
        ):
            raise ValueError(
                "modified dense convolution geometry is unsupported"
            )
        if spatial and (
            len(shape) != 3
            or synapse.weight.shape[-1] != 1
            or tuple(synapse.kernel_size) != tuple(synapse.weight.shape[2:])
            or len(synapse.stride) != 3
            or len(synapse.padding) != 3
            or len(synapse.dilation) != 3
            or synapse.stride[-1] != 1
            or synapse.padding[-1] != 0
            or synapse.dilation[-1] != 1
        ):
            raise ValueError(
                "Conv/Pool requires spatial input and no temporal convolution"
            )
        if pooling:
            _validate_pool_geometry(synapse, shape)
        if synapse.out_channels != synapse.weight.shape[0] or (
            synapse.in_channels != synapse.weight.shape[1] * synapse.groups
        ):
            raise ValueError("source channel metadata does not match weights")
        if (
            input_shape is not None
            and not spatial
            and (len(shape) != 1 or shape[0] != synapse.in_channels)
        ):
            raise ValueError(
                "Dense input width must match, with explicit Flatten after Conv"
            )
        if any(
            not bool(torch.all(state == 0).item())
            for state in (neuron.current_state, neuron.voltage_state)
        ):
            raise ValueError(
                "source current and voltage initial states must be zero"
            )
        if (
            neuron.current_decay.numel() != 1
            or neuron.voltage_decay.numel() != 1
        ):
            raise ValueError("decays must be shared scalars")
        current_decay = float(
            slayer.utils.quantize(neuron.current_decay).item()
        )
        decay = float(slayer.utils.quantize(neuron.voltage_decay).item())
        if current_decay != 4096:
            raise ValueError("current_decay must be 1.0 for delta synapses")
        if not math.isfinite(decay) or not 0 < decay < 4096:
            raise ValueError(
                "voltage decay must map to a finite positive LIF time constant"
            )
        # Source comparisons convert the threshold to the voltage tensor dtype.
        threshold = float(
            torch.tensor(
                neuron.threshold + neuron.threshold_eps, dtype=torch.float32
            ).item()
        )
        tensor = synapse.weight.detach().cpu()
        tau_m = -timestep / math.log1p(-decay / 4096)
        if spatial:
            weights = tensor[..., 0].tolist()
            groups = synapse.groups
            if pooling:
                # A fixed pool is one identical kernel per feature channel.
                weights = weights * shape[0]
                groups = shape[0]
            layer = Conv2dLIFLayer(
                weights=weights,
                tau_m=tau_m,
                threshold=threshold,
                stride=tuple(synapse.stride[:2]),
                padding=tuple(synapse.padding[:2]),
                dilation=tuple(synapse.dilation[:2]),
                groups=groups,
            )
            shape = layer.output_shape(shape)
        else:
            weights = tensor.reshape(
                synapse.out_channels, synapse.in_channels
            ).tolist()
            layer = DenseLIFLayer(
                weights=weights, tau_m=tau_m, threshold=threshold
            )
            shape = (len(weights),)
        if neuron.shape is not None and tuple(neuron.shape) != shape:
            raise ValueError(
                "source neuron shape does not match the declared input geometry"
            )
        # A binary input row bounds both positive and negative accumulation.
        row_bound = max(
            math.fsum(abs(w) for w in row)
            for row in tensor.reshape(tensor.shape[0], -1).tolist()
        )
        scaled_bound = row_bound * neuron.s_scale * 4096 / (decay / 4096)
        if not math.isfinite(scaled_bound) or scaled_bound >= 2**60:
            raise ValueError(
                "weights exceed the conservative source integer-state bound"
            )
        layers.append(layer)
        source_blocks.append(index)
        parameters.append(
            {
                "weights": weights,
                "decay_integer": int(decay),
                "threshold": threshold,
                "state_scale": neuron.s_scale,
                "state_quantum": 1 / neuron.s_scale,
                "tau_m": layer.tau_m,
            }
        )
        if input_shape is not None:
            kind = "dense"
            if convolution:
                kind = "conv2d"
            elif pooling:
                kind = "pool2d"
            parameters[-1].update(
                {
                    "kind": kind,
                    "source_block": index,
                    "output_shape": list(shape),
                }
            )
            if spatial:
                parameters[-1].update(
                    {
                        "stride": list(layer.stride),
                        "padding": list(layer.padding),
                        "dilation": list(layer.dilation),
                        "groups": layer.groups,
                    }
                )
            if pooling:
                parameters[-1]["lowered_kind"] = "conv2d"

    if not layers:
        raise ValueError("source must contain at least one neural layer")

    source_paths = {
        Path(inspect.getfile(type(module))).resolve()
        for module in model.modules()
        if type(module).__module__.startswith("lava.lib.dl.slayer")
    }
    from lava.lib.dl.slayer.neuron.dynamics import leaky_integrator
    from lava.lib.dl.slayer.spike.spike import Spike
    from lava.lib.dl.slayer.utils import int_utils
    from lava.lib.dl.slayer.utils.quantize import quantize

    source_paths.update(
        (
            Path(inspect.getfile(leaky_integrator)).resolve(),
            Path(inspect.getfile(Spike)).resolve(),
            Path(inspect.getfile(slayer.neuron.base)).resolve(),
            Path(inspect.getfile(slayer.block.base)).resolve(),
            Path(inspect.getfile(int_utils)).resolve(),
            Path(inspect.getfile(quantize)).resolve(),
        )
    )
    source_hashes = {
        str(path)
        .split("/slayer/", 1)[-1]: hashlib.sha256(path.read_bytes())
        .hexdigest()
        for path in sorted(source_paths)
    }
    description = {
        "profile": "official-slayer-dense-delta-lif-v1",
        "timestep": timestep,
        "layers": parameters,
        "torch_version": torch.__version__,
        "source_files_sha256": source_hashes,
        "delay_shift": False,
        "input_contract": "binary [batch, channel, time] at t=k*timestep",
        "rounding": "source fixed-point state, target analytical binary64",
        "universal_exact_equivalence": False,
    }
    if input_shape is not None:
        description.update(
            {
                "profile": "official-slayer-feedforward-delta-lif-v1",
                "input_shape": list(input_shape),
                "source_blocks": source_blocks,
                "input_contract": "binary [batch, *input_shape, time] at t=k*timestep",
                "flatten_order": "channel, row, column with column fastest",
                "convolution": "cross-correlation with zero spatial padding",
            }
        )
    fingerprint = hashlib.sha256(
        json.dumps(description, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return layers, description, fingerprint


def _validate_import_options(timestep, acknowledge_quantization):
    if (
        isinstance(timestep, bool)
        or not math.isfinite(timestep)
        or timestep <= 0
    ):
        raise ValueError("timestep must be finite and positive")
    if acknowledge_quantization is not True:
        raise ValueError(
            "set acknowledge_quantization=True and validate spike agreement"
        )


def import_slayer_dense(
    model,
    *,
    timestep: float = 1.0,
    acknowledge_quantization: bool = False,
    name: str = "slayer-trained-lif",
) -> SlayerDenseImport:
    """Copy compatible weights and parameters without altering either engine.

    Acknowledgment is required because matching equations does not reproduce
    SLAYER's fixed-point state rounding. Validate representative inputs before
    deployment. Unsupported blocks or runtime features raise ``ValueError``.
    """

    _validate_import_options(timestep, acknowledge_quantization)
    layers, metadata, fingerprint = _source_description(model, float(timestep))
    deployment = build_dense_lif(layers, name=name)
    return SlayerDenseImport(
        deployment,
        float(timestep),
        fingerprint,
        deployment.network.semantic_sha256,
        metadata,
    )


def import_slayer_feedforward(
    model,
    *,
    input_shape,
    timestep: float = 1.0,
    acknowledge_quantization: bool = False,
    name: str = "slayer-trained-feedforward-lif",
) -> SlayerFeedforwardImport:
    """Export compatible Conv, Pool, Flatten, and Dense blocks as ordinary edges.

    ``input_shape`` excludes batch and time dimensions. Convolution shares
    weights in the source model, but deployment expands them into static
    signed connections. Pool supports fixed nonoverlapping kernels on evenly
    divisible spatial dimensions and lowers to depthwise connectivity. No
    convolution, pooling operator, or training operation runs in C.
    """

    _validate_import_options(timestep, acknowledge_quantization)
    shape = normalize_input_shape(input_shape)
    layers, metadata, fingerprint = _source_description(
        model, float(timestep), input_shape=shape
    )
    deployment = build_feedforward_lif(layers, input_shape=shape, name=name)
    return SlayerFeedforwardImport(
        deployment,
        float(timestep),
        fingerprint,
        deployment.network.semantic_sha256,
        metadata,
    )


def validate_slayer_dense(model, imported, inputs, *, library=None) -> dict:
    """Compare every layer's spikes on a supplied batch using both real engines.

    This is a finite-input validation report, not a general equivalence proof.
    The source must still have the weights and parameters that were exported.
    """

    if not isinstance(imported, SlayerDenseImport):
        raise TypeError("expected SlayerDenseImport")
    return _validate_slayer(model, imported, inputs, library=library)


def validate_slayer_feedforward(
    model, imported, inputs, *, library=None, source_batch_size=64
) -> dict:
    """Compare each neural layer after flattening spatial coordinates to CHW.

    Inputs have shape ``[batch, *input_shape, time]``. Flatten is structural,
    so it does not add a neural layer or a second spike stream to the report.
    """

    if not isinstance(imported, SlayerFeedforwardImport):
        raise TypeError("expected SlayerFeedforwardImport")
    return _validate_slayer(
        model,
        imported,
        inputs,
        library=library,
        source_batch_size=source_batch_size,
    )


def _validate_slayer(
    model, imported, inputs, *, library, source_batch_size=64
):
    import torch
    from lacuna import Engine, RunOptions, SpikeTrain

    if (
        isinstance(source_batch_size, bool)
        or not isinstance(source_batch_size, int)
        or source_batch_size <= 0
    ):
        raise ValueError("source_batch_size must be a positive integer")
    spatial = isinstance(imported, SlayerFeedforwardImport)
    input_shape = (
        imported.deployment.input_shape
        if spatial
        else (len(imported.deployment.input_ports),)
    )
    if imported.deployment.network.semantic_sha256 != imported.network_sha256:
        raise ValueError("deployment network changed after export")
    if (
        hashlib.sha256(
            json.dumps(
                imported.metadata, sort_keys=True, allow_nan=False
            ).encode()
        ).hexdigest()
        != imported.source_fingerprint
    ):
        raise ValueError("source metadata changed after export")
    _, _, fingerprint = _source_description(
        model,
        imported.timestep,
        input_shape=input_shape if spatial else None,
    )
    if fingerprint != imported.source_fingerprint:
        raise ValueError("source model changed after export")
    if (
        not isinstance(inputs, torch.Tensor)
        or inputs.ndim != len(input_shape) + 2
        or any(size <= 0 for size in inputs.shape)
        or tuple(inputs.shape[1:-1]) != input_shape
    ):
        raise ValueError("inputs must be nonempty [batch, *input_shape, time]")
    if not bool(torch.all((inputs == 0) | (inputs == 1)).item()):
        raise ValueError("inputs must contain only binary spikes")
    device = next(model.parameters()).device
    block_indices = (
        imported.metadata["source_blocks"]
        if spatial
        else list(range(len(model)))
    )
    pieces = [[] for _ in block_indices]
    with torch.no_grad():
        # Bound convolution workspace without changing independent samples.
        for chunk in inputs.split(source_batch_size):
            value = chunk.to(device=device, dtype=torch.float32)
            layer_index = 0
            for block_index, block in enumerate(model):
                value = block(value)
                if block_index in block_indices:
                    pieces[layer_index].append(
                        value.detach()
                        .cpu()
                        .reshape(value.shape[0], -1, value.shape[-1])
                    )
                    layer_index += 1
    source = [torch.cat(chunks, dim=0) for chunks in pieces]
    target = [torch.zeros_like(layer) for layer in source]
    node_map = {
        node: (layer, channel)
        for layer, nodes in enumerate(imported.deployment.layer_nodes)
        for channel, node in enumerate(nodes)
    }
    batch, bins = inputs.shape[0], inputs.shape[-1]
    flat_inputs = inputs.reshape(batch, -1, bins)
    dt = imported.timestep
    off_grid = []
    # This profile emits at most one spike per neuron per input bin.
    capacity = (
        bins * (len(node_map) + len(imported.deployment.input_ports)) + 1
    )
    # In this zero-delay feedforward profile each edge delivers at most once
    # per bin. Reserve logical deliveries, not only neuron worklist entries.
    graph = imported.deployment.network.graph
    options = RunOptions(
        output_capacity=capacity,
        queue_capacity=max(4096, len(graph.edges) + len(graph.nodes) + 1),
        encoder_spike_capacity=max(
            4096, bins * len(imported.deployment.input_ports) + 1
        ),
    )
    with Engine(library).compile(imported.deployment.network) as simulation:
        for sample in range(batch):
            stimulus = {
                port: SpikeTrain(
                    times=tuple(
                        k * dt
                        for k in torch.nonzero(
                            flat_inputs[sample, channel], as_tuple=True
                        )[0].tolist()
                    ),
                    values=1.0,
                )
                for channel, port in enumerate(imported.deployment.input_ports)
            }
            result = simulation.run(
                (bins - 1) * dt, inputs=stimulus, options=options
            )
            for spike in result.spikes:
                if spike.node not in node_map:
                    continue
                layer, channel = node_map[spike.node]
                k = round(spike.t / dt)
                if not 0 <= k < bins or not math.isclose(
                    spike.t, k * dt, rel_tol=0.0, abs_tol=1e-9 * dt
                ):
                    off_grid.append([sample, layer, channel, spike.t])
                else:
                    target[layer][sample, channel, k] += 1
    comparisons = []
    for index, (expected, actual) in enumerate(zip(source, target)):
        mismatches = torch.nonzero(expected != actual)
        comparisons.append(
            {
                "layer": index,
                "source_spikes": int(expected.sum().item()),
                "lacuna_spikes": int(actual.sum().item()),
                "mismatched_bins": len(mismatches),
                "first_mismatch_sample_channel_bin": (
                    mismatches[0].tolist() if len(mismatches) else None
                ),
            }
        )
        if spatial:
            comparisons[-1].update(
                {
                    "source_block": block_indices[index],
                    "shape": list(imported.deployment.layer_shapes[index]),
                }
            )
    source_counts, target_counts = source[-1].sum(-1), target[-1].sum(-1)
    source_predictions = source_counts.argmax(-1)
    target_predictions = target_counts.argmax(-1)
    return {
        "samples": batch,
        "bins": bins,
        "source_batch_size": source_batch_size,
        "source_fingerprint": fingerprint,
        "network_sha256": imported.network_sha256,
        "exact_spike_match_on_batch": not off_grid
        and all(item["mismatched_bins"] == 0 for item in comparisons),
        "layers": comparisons,
        "off_grid_spikes": off_grid,
        "prediction_agreement": float(
            (source_predictions == target_predictions).float().mean().item()
        ),
        "source_predictions": source_predictions.tolist(),
        "lacuna_predictions": target_predictions.tolist(),
        "source_output_counts": source_counts.tolist(),
        "lacuna_output_counts": target_counts.tolist(),
        "tie_policy": "lowest output index wins, including an all-silent output",
    }
