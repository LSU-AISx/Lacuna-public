"""Train a compact spiking AlexNet for MNIST and deploy it through Lacuna.

This is an AlexNet-style adaptation with five convolutional and three dense
layers, not the original ImageNet network or an accuracy reproduction.
All neurons and gradients come from unmodified official Lava-DL SLAYER.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from lava.lib.dl import slayer

from lacuna import Engine
from lacuna.experiments import load_mnist
from lacuna.importers import (
    import_slayer_feedforward,
    validate_slayer_feedforward,
)
from examples.train_official_slayer_conv_mnist import encode_spatial
from examples.train_official_slayer_mnist import (
    classification_metrics,
    evaluate,
    write_json as _write_json,
)
from examples.train_official_slayer_pool_mnist import (
    frozen_provenance,
    selection_key,
    sha256,
    tensor_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINED_BLOCKS = (0, 2, 4, 5, 6, 9, 10, 11)
POOL_BLOCKS = (1, 3, 7)
NEURAL_BLOCKS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11)


def write_json(path, value):
    """Publish metadata only after the full document is written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


@dataclass(frozen=True)
class AlexNetConfig:
    channels: tuple[int, ...] = (8, 16, 24, 24, 16)
    hidden: tuple[int, ...] = (64, 32)
    tau_grad: float = 0.01
    scale_grad: float = 10.0
    weight_scale: float = 2.0

    def __post_init__(self):
        for name, count in (("channels", 5), ("hidden", 2)):
            values = tuple(getattr(self, name))
            if len(values) != count or any(
                type(value) is not int or value <= 0 for value in values
            ):
                raise ValueError(f"{name} requires {count} positive integers")
            object.__setattr__(self, name, values)
        for name in ("tau_grad", "scale_grad", "weight_scale"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


def architecture(config):
    """Record every source block and its spatial or dense output shape."""
    rows = [{"type": "input", "output_shape": [1, 32, 32]}]
    channels, width, index = 1, 32, 0
    for number, outputs in enumerate(config.channels):
        kernel = 5 if number < 2 else 3
        rows.append(
            {
                "source_block": index,
                "type": "conv_lif",
                "channels": [channels, outputs],
                "kernel": kernel,
                "padding": kernel // 2,
                "stride": 1,
                "output_shape": [outputs, width, width],
            }
        )
        index += 1
        channels = outputs
        if number in (0, 1, 4):
            width //= 2
            rows.append(
                {
                    "source_block": index,
                    "type": "pool_lif",
                    "kernel": 2,
                    "stride": 2,
                    "weight_scale": 1.0,
                    "weights_trainable": False,
                    "output_shape": [channels, width, width],
                }
            )
            index += 1
    features = channels * width * width
    rows.append(
        {
            "source_block": index,
            "type": "flatten",
            "output_shape": [features],
        }
    )
    index += 1
    for outputs in (*config.hidden, 10):
        rows.append(
            {
                "source_block": index,
                "type": "dense_lif",
                "neurons": [features, outputs],
                "output_shape": [outputs],
            }
        )
        index += 1
        features = outputs
    return rows


def resource_estimate(config):
    """Count expanded connectivity without constructing a Lacuna graph."""
    rows = architecture(config)
    nodes, edges, learned = 1024, 0, 0
    for row in rows[1:]:
        kind = row["type"]
        if kind == "flatten":
            continue
        count = math.prod(row["output_shape"])
        nodes += count
        if kind == "conv_lif":
            width = row["output_shape"][1]
            kernel, pad = row["kernel"], row["padding"]
            valid_axis_pairs = sum(
                0 <= position + offset - pad < width
                for position in range(width)
                for offset in range(kernel)
            )
            channels = math.prod(row["channels"])
            edges += channels * valid_axis_pairs**2
            learned += channels * kernel**2
        elif kind == "pool_lif":
            edges += count * 4
        else:
            weights = math.prod(row["neurons"])
            edges += weights
            learned += weights
    return {
        "neurons_including_input_relays": nodes,
        "edges_before_zero_weight_omission": edges,
        "learned_weights": learned,
        "fixed_pool_kernel_weights": 12,
    }


def build_model(config=AlexNetConfig()):
    params = dict(
        threshold=1.0,
        current_decay=1.0,
        voltage_decay=512 / 4096,
        scale=4096,
        tau_grad=config.tau_grad,
        scale_grad=config.scale_grad,
        persistent_state=False,
        requires_grad=False,
    )
    options = dict(
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
    )
    blocks = []
    for row in architecture(config)[1:]:
        if row["type"] == "conv_lif":
            block = slayer.block.cuba.Conv(
                params,
                *row["channels"],
                row["kernel"],
                padding=row["padding"],
                weight_scale=config.weight_scale,
                **options,
            )
        elif row["type"] == "pool_lif":
            block = slayer.block.cuba.Pool(
                params, 2, stride=2, weight_scale=1.0, **options
            )
        elif row["type"] == "flatten":
            block = slayer.block.cuba.Flatten()
        else:
            block = slayer.block.cuba.Dense(
                params,
                *row["neurons"],
                weight_scale=config.weight_scale,
                **options,
            )
        blocks.append(block)
    return torch.nn.Sequential(*blocks), params


def encode_inputs(images, *, bins, seed):
    """Pad encoded 28 by 28 pixels with a two-pixel silent border."""
    spikes = encode_spatial(images, bins=bins, seed=seed)
    return torch.nn.functional.pad(spikes, (0, 0, 2, 2, 2, 2))


def split_indices(
    train_labels,
    test_labels,
    train_samples,
    validation_samples,
    test_samples,
    seed,
    *,
    training_only=(),
):
    """Use seeded disjoint subsets without discarding majority-class images."""
    sizes = (train_samples, validation_samples, test_samples)
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("sample counts must be positive integers")
    if train_samples + validation_samples > len(train_labels):
        raise ValueError(
            "training and validation exceed the training partition"
        )
    if test_samples > len(test_labels):
        raise ValueError("test samples exceed the canonical test partition")
    reserved = tuple(training_only)
    if len(set(reserved)) != len(reserved) or any(
        type(index) is not int or not 0 <= index < len(train_labels)
        for index in reserved
    ):
        raise ValueError("training-only indices must be unique valid integers")
    if len(reserved) > train_samples:
        raise ValueError(
            "training subset cannot hold all training-only indices"
        )
    order = np.random.default_rng(seed).permutation(len(train_labels))
    if reserved:
        order = np.concatenate(
            (
                np.asarray(reserved, dtype=np.int64),
                order[~np.isin(order, reserved)],
            )
        )
    test = np.random.default_rng(seed + 1).permutation(len(test_labels))
    return (
        order[:train_samples],
        order[train_samples : train_samples + validation_samples],
        test[:test_samples],
    )


def weight_snapshot(model):
    return {
        index: model[index].synapse.weight.detach().clone()
        for index in NEURAL_BLOCKS
    }


def weight_audit(model, before):
    rows = []
    for index in NEURAL_BLOCKS:
        weights = model[index].synapse.weight
        trained = index in TRAINED_BLOCKS
        unchanged = torch.equal(weights.detach(), before[index])
        if weights.requires_grad != trained:
            raise RuntimeError(f"unexpected trainability at block {index}")
        if not trained and not unchanged:
            raise RuntimeError(f"fixed pool weights changed at block {index}")
        if not bool(torch.isfinite(weights).all()):
            raise RuntimeError(f"nonfinite weights at block {index}")
        rows.append(
            {
                "source_block": index,
                "trained": trained,
                "weight_count": weights.numel(),
                "unchanged": unchanged,
                "weight_change_l2": float(
                    torch.linalg.vector_norm(weights.detach() - before[index])
                ),
                "initial_sha256": tensor_sha256(before[index]),
                "selected_sha256": tensor_sha256(weights),
            }
        )
    return rows


def loss_function():
    return slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.02, reduction="mean"
    )


def preflight(config, *, bins, batch_size, seed, learning_rate):
    """Measure activity and an optimizer step without consuming training RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, _ = build_model(config)
        before = weight_snapshot(model)
        images = np.full((batch_size, 784), 200, dtype=np.uint8)
        inputs = encode_inputs(images, bins=bins, seed=seed + 2)
        labels = torch.arange(batch_size) % 10
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )
        started = time.perf_counter()
        value, counts = inputs, []
        for index, block in enumerate(model):
            value = block(value)
            if index in NEURAL_BLOCKS:
                counts.append(int(value.detach().sum()))
        loss = loss_function()(value, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite preflight loss")
        loss.backward()
        gradients = [
            float(model[index].synapse.weight.grad.abs().max())
            for index in TRAINED_BLOCKS
        ]
        if any(not math.isfinite(value) or value <= 0 for value in gradients):
            raise RuntimeError("preflight gradient is zero or nonfinite")
        optimizer.step()
        audit = weight_audit(model, before)
        seconds = time.perf_counter() - started
        if any(row["trained"] and row["unchanged"] for row in audit):
            raise RuntimeError(
                "preflight did not update all eight learned layers"
            )
    return {
        "data": "synthetic constant-pixel images, not MNIST accuracy",
        "samples": batch_size,
        "bins": bins,
        "spikes_by_neural_block": counts,
        "gradient_max_abs_by_trained_block": gradients,
        "weight_audit": audit,
        "forward_backward_optimizer_seconds": seconds,
        "timing_scope": "single synthetic batch, includes first-use overhead",
    }


def provenance(data_dir, engine):
    result = frozen_provenance(data_dir, engine)
    for name in (
        "examples/train_official_slayer_alexnet_mnist.py",
        "examples/train_official_slayer_pool_mnist.py",
    ):
        result["workflow_sources_sha256"][name] = sha256(ROOT / name)
    return result


def verify_protocol(output, protocol, data_dir, engine):
    for key, value in provenance(data_dir, engine).items():
        if protocol[key] != value:
            raise ValueError(f"frozen {key} changed")
    on_disk = json.loads((output / "protocol.json").read_text())
    if on_disk != protocol:
        raise ValueError("frozen protocol changed")


def save_checkpoint(path, checkpoint):
    """Replace one checkpoint only after its complete contents have been saved."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_resume_state(output, checkpoint):
    """Bind the latest complete epoch to a checkpoint hash for resume."""
    path = output / f"epoch-{checkpoint['epoch']:06d}.pt"
    save_checkpoint(path, checkpoint)
    write_json(
        output / "resume.json",
        {
            "checkpoint_file": path.name,
            "checkpoint_sha256": sha256(path),
            "protocol_sha256": checkpoint["protocol_sha256"],
            "epoch": checkpoint["epoch"],
        },
    )


def load_resume_state(output, protocol_hash):
    manifest = json.loads((output / "resume.json").read_text())
    filename = f"epoch-{manifest['epoch']:06d}.pt"
    if manifest["checkpoint_file"] != filename:
        raise ValueError("resume checkpoint filename does not match epoch")
    path = output / filename
    if manifest["protocol_sha256"] != protocol_hash:
        raise ValueError("resume protocol does not match")
    if sha256(path) != manifest["checkpoint_sha256"]:
        raise ValueError("resume checkpoint changed or is incomplete")
    state = torch.load(path, weights_only=True)
    if state["protocol_sha256"] != protocol_hash or (
        state["epoch"] != manifest["epoch"]
    ):
        raise ValueError("resume checkpoint metadata does not match")
    return state


def train(args, output, config, images, labels, validation, validation_y):
    torch.manual_seed(args.seed)
    model, params = build_model(config)
    before = weight_snapshot(model)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    loss_fn = loss_function()
    protocol_hash = sha256(output / "protocol.json")
    history, best, start_epoch = [], None, 1
    if (output / "resume.json").exists():
        state = load_resume_state(output, protocol_hash)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        before, history, best = (
            state["initial_weights"],
            state["history"],
            state["best"],
        )
        start_epoch = state["epoch"] + 1
        torch.random.set_rng_state(state["torch_rng_state"])
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = np.random.default_rng(args.seed + 1000 + epoch).permutation(
            len(labels)
        )
        weighted_loss, gradient_maxima = 0.0, [0.0] * len(TRAINED_BLOCKS)
        for batch_index, start in enumerate(
            range(0, len(order), args.batch_size)
        ):
            chosen = order[start : start + args.batch_size]
            inputs = encode_inputs(
                images[chosen],
                bins=args.bins,
                seed=args.seed + 1000000 * epoch + batch_index,
            )
            targets = torch.tensor(labels[chosen], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite training loss")
            loss.backward()
            for slot, index in enumerate(TRAINED_BLOCKS):
                maximum = float(model[index].synapse.weight.grad.abs().max())
                if not math.isfinite(maximum):
                    raise RuntimeError(f"nonfinite gradient at block {index}")
                gradient_maxima[slot] = max(gradient_maxima[slot], maximum)
            optimizer.step()
            weighted_loss += float(loss.item()) * len(chosen)
        audit = weight_audit(model, before)
        metrics = evaluate(
            model, validation, validation_y, loss_fn, args.batch_size
        )
        if best is None or selection_key(metrics, epoch) > selection_key(
            best["validation"], best["epoch"]
        ):
            best = {
                "epoch": epoch,
                "validation": metrics,
                "model_state_dict": copy.deepcopy(model.state_dict()),
            }
        history.append(
            {
                "epoch": epoch,
                "train_loss": weighted_loss / len(labels),
                "validation": metrics,
                "gradient_max_abs_by_trained_block": gradient_maxima,
                "weight_audit": audit,
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        save_resume_state(
            output,
            {
                "epoch": epoch,
                "protocol_sha256": protocol_hash,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "torch_rng_state": torch.random.get_rng_state(),
                "initial_weights": before,
                "history": history,
                "best": best,
            },
        )
        write_json(output / "history.json", history)
        print(
            json.dumps(
                {
                    "stage": "training",
                    "epoch": epoch,
                    "train_loss": history[-1]["train_loss"],
                    "validation_accuracy": metrics["accuracy"],
                    "best_epoch": best["epoch"],
                    "seconds": history[-1]["seconds"],
                }
            ),
            flush=True,
        )
        if args.max_training_seconds and (
            time.perf_counter() - started >= args.max_training_seconds
        ):
            break
    if best is None:
        raise RuntimeError("no complete epoch is available for deployment")
    model.load_state_dict(best["model_state_dict"])
    audit = weight_audit(model, before)
    summary = {
        "epochs_run": len(history),
        "epochs_requested": args.epochs,
        "training_completed": len(history) == args.epochs,
        "stop_reason": (
            "epochs_complete" if len(history) == args.epochs else "time_budget"
        ),
        "selected_epoch": best["epoch"],
        "selected_validation": best["validation"],
        "training_seconds": sum(row["seconds"] for row in history),
        "weight_audit": audit,
        "all_learned_tensors_changed": all(
            not row["unchanged"] for row in audit if row["trained"]
        ),
        "pool_weights_unchanged": all(
            row["unchanged"] for row in audit if not row["trained"]
        ),
        "protocol_sha256": protocol_hash,
    }
    save_checkpoint(
        output / "official_checkpoint.pt",
        {
            "model_state_dict": best["model_state_dict"],
            "neuron_params": params,
            "architecture": architecture(config),
            "selected_epoch": best["epoch"],
            "protocol_sha256": protocol_hash,
            "initial_weights": before,
        },
    )
    summary["checkpoint_sha256"] = sha256(output / "official_checkpoint.pt")
    write_json(output / "training_summary.json", summary)
    return summary


def restore_selected(output, config, summary):
    path = output / "official_checkpoint.pt"
    if sha256(path) != summary["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed")
    if sha256(output / "protocol.json") != summary["protocol_sha256"]:
        raise ValueError("selected checkpoint protocol changed")
    state = torch.load(path, weights_only=True)
    model, params = build_model(config)
    if (
        state["architecture"] != architecture(config)
        or state["neuron_params"] != params
    ):
        raise ValueError("selected checkpoint model contract changed")
    if state["selected_epoch"] != summary["selected_epoch"] or (
        state["protocol_sha256"] != summary["protocol_sha256"]
    ):
        raise ValueError("selected checkpoint selection metadata changed")
    model.load_state_dict(state["model_state_dict"])
    if (
        weight_audit(model, state["initial_weights"])
        != summary["weight_audit"]
    ):
        raise ValueError("selected checkpoint weight audit changed")
    return model.eval()


def compare_in_chunks(model, imported, inputs, *, batch_size, chunk_size):
    """Bound retained layer tensors while preserving source batch boundaries."""
    if chunk_size < batch_size or chunk_size % batch_size:
        raise ValueError(
            "comparison chunk size must be a multiple of batch size"
        )
    merged = None
    for start in range(0, len(inputs), chunk_size):
        report = validate_slayer_feedforward(
            model,
            imported,
            inputs[start : start + chunk_size],
            source_batch_size=batch_size,
        )
        if merged is None:
            merged = copy.deepcopy(report)
            merged["samples"] = 0
            merged["off_grid_spikes"] = []
            merged["exact_spike_match_on_batch"] = True
            for key in (
                "source_predictions",
                "lacuna_predictions",
                "source_output_counts",
                "lacuna_output_counts",
            ):
                merged[key] = []
            for row in merged["layers"]:
                row.update(
                    {
                        "source_spikes": 0,
                        "lacuna_spikes": 0,
                        "mismatched_bins": 0,
                        "first_mismatch_sample_channel_bin": None,
                    }
                )
        for key in ("source_fingerprint", "network_sha256", "bins"):
            if report[key] != merged[key]:
                raise ValueError("comparison identity changed between chunks")
        merged["samples"] += report["samples"]
        merged["exact_spike_match_on_batch"] &= report[
            "exact_spike_match_on_batch"
        ]
        merged["off_grid_spikes"].extend(
            [row[0] + start, *row[1:]] for row in report["off_grid_spikes"]
        )
        for key in (
            "source_predictions",
            "lacuna_predictions",
            "source_output_counts",
            "lacuna_output_counts",
        ):
            merged[key].extend(report[key])
        for target, source in zip(merged["layers"], report["layers"]):
            for key in ("source_spikes", "lacuna_spikes", "mismatched_bins"):
                target[key] += source[key]
            first = source["first_mismatch_sample_channel_bin"]
            if (
                first is not None
                and target["first_mismatch_sample_channel_bin"] is None
            ):
                target["first_mismatch_sample_channel_bin"] = [
                    first[0] + start,
                    *first[1:],
                ]
        print(
            json.dumps(
                {
                    "stage": "comparison chunk",
                    "completed": merged["samples"],
                    "total": len(inputs),
                }
            ),
            flush=True,
        )
    if merged is None:
        raise ValueError("comparison requires at least one sample")
    merged["comparison_chunk_size"] = chunk_size
    merged["prediction_agreement"] = (
        sum(
            left == right
            for left, right in zip(
                merged["source_predictions"], merged["lacuna_predictions"]
            )
        )
        / merged["samples"]
    )
    return merged


def export(
    args, output, config, summary, validation, validation_y, test, test_y
):
    """Export the saved checkpoint and retain every neural layer comparison."""
    model = restore_selected(output, config, summary)
    print(json.dumps({"stage": "importing selected checkpoint"}), flush=True)
    imported = import_slayer_feedforward(
        model,
        input_shape=(1, 32, 32),
        timestep=1.0,
        acknowledge_quantization=True,
        name="official-slayer-alexnet-mnist",
    )
    graph = imported.deployment.network.graph
    if len(imported.metadata["layers"]) != len(NEURAL_BLOCKS):
        raise RuntimeError("import omitted a neural layer")
    expected = resource_estimate(config)
    if len(graph.nodes) != expected["neurons_including_input_relays"] or (
        len(graph.edges) > expected["edges_before_zero_weight_omission"]
    ):
        raise RuntimeError(
            "imported graph geometry does not match architecture"
        )
    imported.deployment.network.save(output / "network.json")
    torch.save(
        {
            "validation_inputs": validation.to(torch.uint8),
            "validation_labels": torch.tensor(validation_y),
            "test_inputs": test.to(torch.uint8),
            "test_labels": torch.tensor(test_y),
        },
        output / "validation_inputs.pt",
    )
    comparisons = {}
    for split, inputs in (("validation", validation), ("test", test)):
        print(
            json.dumps(
                {
                    "stage": "cross-engine comparison",
                    "split": split,
                    "samples": len(inputs),
                }
            ),
            flush=True,
        )
        comparison = compare_in_chunks(
            model,
            imported,
            inputs,
            batch_size=args.batch_size,
            chunk_size=args.comparison_chunk_size,
        )
        write_json(output / f"{split}_comparison.json", comparison)
        if comparison["off_grid_spikes"] or len(comparison["layers"]) != 11:
            raise RuntimeError("off-grid spike or missing neural layer")
        comparisons[split] = comparison
    with Engine().compile(imported.deployment.network) as simulation:
        simulation.save_compiled_graph_image(output / "network.lcbin")
        execution_path = simulation.preferred_execution_path
    result = {
        "graph_neurons_including_input_relays": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "execution_path": execution_path,
        "import": imported.metadata,
    }
    for split, labels in (("validation", validation_y), ("test", test_y)):
        comparison = comparisons[split]
        for engine, field in (("official", "source"), ("lacuna", "lacuna")):
            result[f"{engine}_{split}"] = classification_metrics(
                labels, comparison[f"{field}_predictions"]
            )
        matched = sum(
            left == right
            for left, right in zip(
                comparison["source_predictions"],
                comparison["lacuna_predictions"],
            )
        )
        result[f"{split}_matching_predictions"] = matched
        result[f"{split}_prediction_agreement"] = matched / len(labels)
        result[f"{split}_spikes_match"] = comparison[
            "exact_spike_match_on_batch"
        ]
        result[f"{split}_layer_comparison"] = comparison["layers"]
    return result


def validate_args(args):
    for name in (
        "epochs",
        "bins",
        "batch_size",
        "threads",
        "comparison_chunk_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    if (
        not math.isfinite(args.max_training_seconds)
        or args.max_training_seconds < 0
    ):
        raise ValueError("max training seconds must be finite and nonnegative")
    if args.seed < 0 or args.split_seed < 0:
        raise ValueError("seeds must be nonnegative")
    if args.comparison_chunk_size % args.batch_size:
        raise ValueError(
            "comparison chunk size must be a multiple of batch size"
        )


def run(args):
    validate_args(args)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.resume:
        output = Path(args.resume)
        if (output / "report.json").exists():
            raise ValueError("completed examples cannot be resumed")
        protocol = json.loads((output / "protocol.json").read_text())
        budget = args.max_training_seconds
        args = argparse.Namespace(**protocol["config"])
        args.data_dir = Path(args.data_dir)
        args.max_training_seconds = budget
        torch.set_num_threads(args.threads)
        config = AlexNetConfig(**protocol["model_config"])
    else:
        config = AlexNetConfig(
            channels=tuple(args.channels),
            hidden=tuple(args.hidden),
            tau_grad=args.tau_grad,
            scale_grad=args.scale_grad,
            weight_scale=args.weight_scale,
        )
        output = (
            Path(args.output)
            if args.output
            else (
                Path("artifacts/official-slayer-alexnet-mnist")
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            )
        )
        output.mkdir(parents=True, exist_ok=False)
        protocol = None
    print(
        json.dumps(
            {
                "output": str(output),
                "architecture_estimate": resource_estimate(config),
            }
        ),
        flush=True,
    )
    if protocol is None:
        diagnostic = preflight(
            config,
            bins=args.bins,
            batch_size=args.batch_size,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        write_json(output / "preflight.json", diagnostic)
        print(
            json.dumps(
                {
                    "stage": "preflight",
                    "seconds": diagnostic[
                        "forward_backward_optimizer_seconds"
                    ],
                    "all_eight_learned_layers_updated": True,
                    "projected_training_batch_seconds": diagnostic[
                        "forward_backward_optimizer_seconds"
                    ]
                    * math.ceil(args.train_samples / args.batch_size)
                    * args.epochs,
                    "estimate_excludes": "validation, conversion, replay, and runtime variability",
                }
            ),
            flush=True,
        )
        if args.preflight_only:
            return output
    train_x, train_y, test_x, test_y = load_mnist(
        args.data_dir, download=False
    )
    engine = Engine()
    if protocol is None:
        reserved = []
        if args.training_only_indices:
            evidence = json.loads(Path(args.training_only_indices).read_text())
            reserved = (
                evidence
                if isinstance(evidence, list)
                else evidence["input"]["indices"]
            )
        train_ids, val_ids, test_ids = split_indices(
            train_y,
            test_y,
            args.train_samples,
            args.validation_samples,
            args.test_samples,
            args.split_seed,
            training_only=reserved,
        )
        protocol = {
            "task": "compact spiking AlexNet-style MNIST conversion example",
            "model_config": json.loads(json.dumps(asdict(config))),
            "architecture": architecture(config),
            "resource_estimate": resource_estimate(config),
            "config": vars(args) | {"data_dir": str(args.data_dir)},
            "train_indices": train_ids.tolist(),
            "validation_indices": val_ids.tolist(),
            "canonical_test_indices": test_ids.tolist(),
            "development_training_only_indices": reserved,
            "development_evidence_sha256": (
                sha256(args.training_only_indices)
                if args.training_only_indices
                else None
            ),
            "split": "seeded random disjoint train/validation, canonical test",
            "test_used_for_selection": False,
            "encoder": "Bernoulli p=0.4*pixel/255, then two-pixel silent border",
            "bins": args.bins,
            "timestep": 1.0,
            "selection": "validation accuracy, lower loss, earlier epoch",
            "trained_source_blocks": list(TRAINED_BLOCKS),
            "fixed_pool_source_blocks": list(POOL_BLOCKS),
            "neural_source_blocks": list(NEURAL_BLOCKS),
            "optimizer": "Adam on all eight Conv/Dense layers, pools fixed",
            "loss": "official SpikeRate(true_rate=0.2, false_rate=0.02)",
            **provenance(args.data_dir, engine),
        }
        protocol = json.loads(json.dumps(protocol, allow_nan=False))
        write_json(output / "protocol.json", protocol)
    else:
        verify_protocol(output, protocol, args.data_dir, engine)
        train_ids, val_ids, test_ids = (
            np.asarray(protocol[key], dtype=np.int64)
            for key in (
                "train_indices",
                "validation_indices",
                "canonical_test_indices",
            )
        )
    validation = encode_inputs(
        train_x[val_ids], bins=args.bins, seed=args.seed + 200000
    )
    summary_path = output / "training_summary.json"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    if summary is None or not summary["training_completed"]:
        summary = train(
            args,
            output,
            config,
            train_x[train_ids],
            train_y[train_ids],
            validation,
            train_y[val_ids],
        )
    verify_protocol(output, protocol, args.data_dir, engine)
    if not summary["training_completed"]:
        print(
            json.dumps(
                {
                    "stage": "paused at epoch boundary",
                    "output": str(output),
                    "epochs_completed": summary["epochs_run"],
                    "resume": f"--resume {output}",
                    "test_evaluated": False,
                }
            ),
            flush=True,
        )
        return output
    test = encode_inputs(
        test_x[test_ids], bins=args.bins, seed=args.seed + 300000
    )
    started = time.perf_counter()
    report = export(
        args,
        output,
        config,
        summary,
        validation,
        train_y[val_ids],
        test,
        test_y[test_ids],
    )
    verify_protocol(output, protocol, args.data_dir, engine)
    report.update(
        {
            "task": protocol["task"],
            "architecture": architecture(config),
            "train_samples": len(train_ids),
            "validation_samples": len(val_ids),
            "test_samples": len(test_ids),
            "model_config": protocol["model_config"],
            "training": summary,
            "export_and_comparison_seconds": time.perf_counter() - started,
            "lacuna_library_sha256": protocol["lacuna_library_sha256"],
            "frozen_provenance_verified_after_training_and_replay": True,
            "files_sha256": {
                path.name: sha256(path)
                for path in sorted(output.iterdir())
                if path.is_file()
                and path.name not in ("report.json", "failure.json")
            },
        }
    )
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_epoch": summary["selected_epoch"],
                "official_test_accuracy": report["official_test"]["accuracy"],
                "lacuna_test_accuracy": report["lacuna_test"]["accuracy"],
                "matching_test_predictions": report[
                    "test_matching_predictions"
                ],
                "test_samples": len(test_ids),
            }
        ),
        flush=True,
    )
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--data-dir", type=Path, default=Path("artifacts/mnist-data")
    )
    result.add_argument(
        "--channels", type=int, nargs=5, default=(8, 16, 24, 24, 16)
    )
    result.add_argument("--hidden", type=int, nargs=2, default=(64, 32))
    result.add_argument("--train-samples", type=int, default=10000)
    result.add_argument("--validation-samples", type=int, default=1000)
    result.add_argument("--test-samples", type=int, default=1000)
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--bins", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--comparison-chunk-size", type=int, default=64)
    result.add_argument("--learning-rate", type=float, default=0.001)
    result.add_argument("--tau-grad", type=float, default=0.01)
    result.add_argument("--scale-grad", type=float, default=10.0)
    result.add_argument("--weight-scale", type=float, default=2.0)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--split-seed", type=int, default=137)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument("--max-training-seconds", type=float, default=0)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument(
        "--training-only-indices",
        help="JSON indices or diagnostic report used in development, never validation",
    )
    result.add_argument("--output")
    result.add_argument(
        "--resume", help="resume from the latest complete epoch or export"
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
