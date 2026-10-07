"""Train a bounded official SLAYER CNN+Pool and validate ordinary Lacuna replay.

The default is one fixed-seed, validation-selected experiment, not a search.
Pooling uses the official frozen synapses and spiking CUBA neurons: this is
neither max pooling nor an importer-side approximation of pooling.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from lava.lib.dl import slayer

from lacuna import Engine
from lacuna.experiments import load_mnist
from examples.train_official_slayer_conv_mnist import (
    data_hashes,
    encode_spatial,
    source_provenance,
)
from examples.train_official_slayer_mnist import (
    classification_metrics,
    evaluate,
    stratified_groups,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINED_BLOCKS = (0, 2, 5)
POOL_BLOCKS = (1, 3)
NEURAL_BLOCKS = (0, 1, 2, 3, 5)
ARCHITECTURE = [
    {"type": "input", "shape": [1, 28, 28]},
    {
        "type": "conv_lif",
        "channels": [1, 4],
        "kernel": 3,
        "stride": 1,
        "padding": 1,
        "output_shape": [4, 28, 28],
    },
    {
        "type": "pool_lif",
        "kernel": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "weight_scale": 1.0,
        "weights_trainable": False,
        "output_shape": [4, 14, 14],
    },
    {
        "type": "conv_lif",
        "channels": [4, 8],
        "kernel": 3,
        "stride": 1,
        "padding": 1,
        "output_shape": [8, 14, 14],
    },
    {
        "type": "pool_lif",
        "kernel": 2,
        "stride": 2,
        "padding": 0,
        "dilation": 1,
        "weight_scale": 1.0,
        "weights_trainable": False,
        "output_shape": [8, 7, 7],
    },
    {"type": "flatten", "order": "channel, row, column"},
    {"type": "dense_lif", "neurons": [392, 10]},
]


def build_model():
    """Use unmodified official blocks, with all Conv/Dense weights learned."""
    params = dict(
        threshold=1.0,
        current_decay=1.0,
        voltage_decay=512 / 4096,
        scale=4096,
        tau_grad=1.0,
        scale_grad=1.0,
        persistent_state=False,
        requires_grad=False,
    )
    options = dict(
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
    )
    return (
        torch.nn.Sequential(
            slayer.block.cuba.Conv(
                params, 1, 4, 3, padding=1, weight_scale=2.0, **options
            ),
            slayer.block.cuba.Pool(
                params,
                2,
                stride=2,
                padding=0,
                dilation=1,
                weight_scale=1.0,
                **options,
            ),
            slayer.block.cuba.Conv(
                params, 4, 8, 3, padding=1, weight_scale=2.0, **options
            ),
            slayer.block.cuba.Pool(
                params,
                2,
                stride=2,
                padding=0,
                dilation=1,
                weight_scale=1.0,
                **options,
            ),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(
                params, 392, 10, weight_scale=2.0, **options
            ),
        ),
        params,
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def workflow_source_hashes():
    """Freeze training utilities, importer/frontend, and core implementation."""
    paths = {
        ROOT / "examples/train_official_slayer_pool_mnist.py",
        ROOT / "examples/train_official_slayer_conv_mnist.py",
        ROOT / "examples/train_official_slayer_mnist.py",
        *list((ROOT / "src/lacuna").rglob("*.py")),
        *list((ROOT / "c").rglob("*.c")),
        *list((ROOT / "c").rglob("*.h")),
    }
    return {
        str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)
    }


def official_source_hashes():
    """Hash every tracked official implementation file, in addition to its commit."""
    source_root = Path(slayer.__file__).resolve().parents[5]
    names = (
        subprocess.check_output(
            ["git", "-C", str(source_root), "ls-files", "-z", "src"],
        )
        .decode()
        .split("\0")
    )
    return {name: sha256(source_root / name) for name in sorted(names) if name}


def frozen_provenance(data_dir, engine):
    provenance = source_provenance()
    if provenance["official_source_dirty"]:
        raise ValueError("official SLAYER source must be unmodified")
    return {
        "source_provenance": provenance,
        "official_sources_sha256": official_source_hashes(),
        "workflow_sources_sha256": workflow_source_hashes(),
        "dataset_sha256": data_hashes(data_dir),
        "lacuna_library_sha256": sha256(engine.core._lib._name),
    }


def verify_frozen_protocol(protocol, current):
    if protocol["architecture"] != ARCHITECTURE:
        raise ValueError("architecture changed since the protocol was frozen")
    for key, actual in current.items():
        if protocol[key] != actual:
            raise ValueError(f"{key} changed since the protocol was frozen")


def weight_snapshot(model):
    return {
        index: model[index].synapse.weight.detach().clone()
        for index in NEURAL_BLOCKS
    }


def weight_audit(model, before, *, require_learned_change=False):
    """Require frozen pools and report each trained tensor separately."""
    rows = []
    for index in NEURAL_BLOCKS:
        weight = model[index].synapse.weight
        trained = index in TRAINED_BLOCKS
        unchanged = torch.equal(weight.detach(), before[index])
        if weight.requires_grad != trained:
            raise RuntimeError(
                f"unexpected trainability at source block {index}"
            )
        if not trained and not unchanged:
            raise RuntimeError(
                f"fixed pool weights changed at source block {index}"
            )
        if trained and unchanged and require_learned_change:
            raise RuntimeError(
                f"trained weights did not change at source block {index}"
            )
        rows.append(
            {
                "source_block": index,
                "type": type(model[index]).__name__,
                "trained": trained,
                "weight_count": weight.numel(),
                "initial_sha256": tensor_sha256(before[index]),
                "selected_sha256": tensor_sha256(weight),
                "weight_change_l2": float(
                    torch.linalg.vector_norm(weight.detach() - before[index])
                ),
                "unchanged": unchanged,
            }
        )
    return rows


def selection_key(validation, epoch):
    """Use only validation metrics and favor earlier epochs for ties."""
    return validation["accuracy"], -validation["loss"], -epoch


def synthetic_preflight():
    """Check end-to-end learning on synthetic data without consuming pilot RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model, _ = build_model()
        before = weight_snapshot(model)
        inputs = encode_spatial(np.full((4, 784), 200), bins=32, seed=2)
        output = inputs
        counts = []
        for block in model:
            output = block(output)
            counts.append(int(output.detach().sum()))
        loss_function = slayer.loss.SpikeRate(
            true_rate=0.2, false_rate=0.02, reduction="mean"
        )
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=0.003
        )
        optimizer.zero_grad()
        loss = loss_function(output, torch.arange(4))
        loss.backward()
        gradient_maxima = [
            float(model[index].synapse.weight.grad.abs().max())
            for index in TRAINED_BLOCKS
        ]
        optimizer.step()
        audit = weight_audit(model, before, require_learned_change=True)
    return {
        "data": "synthetic constant-pixel images, not MNIST accuracy results",
        "samples": 4,
        "pixel_value": 200,
        "bins": 32,
        "model_seed": 0,
        "encoding_seed": 2,
        "labels": [0, 1, 2, 3],
        "decision": "before MNIST training, use official default sum-pool weights 1.0 instead of quarter weights 0.25 to establish activity and end-to-end updates",
        "prior_quarter_weight_diagnostic": {
            "spike_totals_by_source_block": [21705, 2858, 42, 0, 0, 0],
            "gradient_max_abs_by_trained_block": [
                8.911685518830513e-16,
                4.132735820228106e-11,
                0.0,
            ],
            "mnist_training_or_test_evaluation_performed": False,
        },
        "frozen_default_sum_pool_diagnostic": {
            "spike_totals_by_source_block": counts,
            "gradient_max_abs_by_trained_block": gradient_maxima,
            "single_adam_step_learning_rate": 0.003,
            "weight_audit": audit,
        },
    }


def train(args, output, images, labels, validation_inputs, validation_y):
    model, neuron_params = build_model()
    initial_weights = weight_snapshot(model)
    weight_audit(model, initial_weights)
    loss_function = slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.02, reduction="mean"
    )
    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
    )
    initial_validation = evaluate(
        model, validation_inputs, validation_y, loss_function, args.batch_size
    )
    best_key, best_state, best_epoch, best_validation = None, None, None, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = np.random.default_rng(args.seed + 1000 + epoch).permutation(
            len(labels)
        )
        weighted_loss = 0.0
        for batch_index, start in enumerate(
            range(0, len(order), args.batch_size)
        ):
            chosen = order[start : start + args.batch_size]
            inputs = encode_spatial(
                images[chosen],
                bins=args.bins,
                seed=args.seed + 1000000 * epoch + batch_index,
            )
            targets = torch.tensor(labels[chosen], dtype=torch.long)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite official SLAYER loss")
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.item()) * len(chosen)
        weight_audit(model, initial_weights)
        validation = evaluate(
            model,
            validation_inputs,
            validation_y,
            loss_function,
            args.batch_size,
        )
        key = selection_key(validation, epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_validation = key, epoch, validation
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "neuron_params": neuron_params,
                    "architecture": ARCHITECTURE,
                    "selected_epoch": epoch,
                    "initial_synapse_weights": initial_weights,
                    "protocol_sha256": sha256(output / "protocol.json"),
                },
                output / "official_checkpoint.pt",
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": weighted_loss / len(labels),
                "validation": validation,
                "seconds": time.perf_counter() - epoch_started,
            }
        )
        write_json(output / "history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": history[-1]["train_loss"],
                    "validation_accuracy": validation["accuracy"],
                    "best_epoch": best_epoch,
                    "seconds": history[-1]["seconds"],
                }
            ),
            flush=True,
        )
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    audit = weight_audit(model, initial_weights, require_learned_change=True)
    summary = {
        "epochs_run": args.epochs,
        "selected_epoch": best_epoch,
        "initial_validation": initial_validation,
        "selected_validation": best_validation,
        "training_seconds": training_seconds,
        "weight_audit": audit,
        "weight_change_l2_by_layer": [
            row["weight_change_l2"] for row in audit if row["trained"]
        ],
        "trainable_weight_count_by_layer": [
            row["weight_count"] for row in audit if row["trained"]
        ],
        "pool_weights_unchanged": all(
            row["unchanged"] for row in audit if not row["trained"]
        ),
        "trained_source_blocks": list(TRAINED_BLOCKS),
        "fixed_pool_source_blocks": list(POOL_BLOCKS),
        "checkpoint_sha256": sha256(output / "official_checkpoint.pt"),
        "protocol_sha256": sha256(output / "protocol.json"),
    }
    write_json(output / "training_summary.json", summary)
    return model.eval(), summary


def restore_model(output, summary):
    if (
        sha256(output / "official_checkpoint.pt")
        != summary["checkpoint_sha256"]
    ):
        raise ValueError("selected checkpoint changed since training")
    if sha256(output / "protocol.json") != summary["protocol_sha256"]:
        raise ValueError("protocol changed since training")
    checkpoint = torch.load(
        output / "official_checkpoint.pt", weights_only=True
    )
    if checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("checkpoint architecture does not match the pilot")
    if checkpoint["selected_epoch"] != summary["selected_epoch"]:
        raise ValueError("checkpoint epoch does not match the selected epoch")
    if checkpoint["protocol_sha256"] != summary["protocol_sha256"]:
        raise ValueError("checkpoint protocol identity does not match")
    model, params = build_model()
    if checkpoint["neuron_params"] != params:
        raise ValueError("checkpoint neuron parameters do not match the pilot")
    model.load_state_dict(checkpoint["model_state_dict"])
    audit = weight_audit(
        model,
        checkpoint["initial_synapse_weights"],
        require_learned_change=True,
    )
    if audit != summary["weight_audit"]:
        raise ValueError(
            "selected weight audit does not match training summary"
        )
    return model.eval()


def run(args):
    if args.resume:
        output = Path(args.resume)
        if (output / "report.json").exists():
            raise ValueError("this pilot already has a completed report")
        protocol = json.loads((output / "protocol.json").read_text())
        args = argparse.Namespace(**protocol["config"])
        args.data_dir = Path(args.data_dir)
        summary = json.loads((output / "training_summary.json").read_text())
    else:
        if any(
            value <= 0 for value in (args.epochs, args.bins, args.batch_size)
        ):
            raise ValueError("epochs, bins, and batch size must be positive")
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        output = (
            Path(args.output)
            if args.output
            else (
                Path("artifacts/official-slayer-pool-mnist")
                / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            )
        )
        output.mkdir(parents=True, exist_ok=False)
        protocol, summary = None, None
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    print(
        json.dumps({"output": str(output), "stage": "loading verified data"}),
        flush=True,
    )
    train_x, train_y, test_x, test_y = load_mnist(
        args.data_dir, download=False
    )
    engine = Engine()
    provenance = frozen_provenance(args.data_dir, engine)
    if protocol is None:
        train_indices, val_indices = stratified_groups(
            train_y,
            (args.train_samples, args.validation_samples),
            seed=args.split_seed,
        )
        (test_indices,) = stratified_groups(
            test_y, (args.test_samples,), seed=args.split_seed + 1
        )
        _, neuron_params = build_model()
        # Model construction consumes random numbers. Restart the frozen seed
        # so training and restore always use the same initialization.
        torch.manual_seed(args.seed)
        protocol = {
            "pilot": "one fixed-seed CNN+Pool subset validation, no hyperparameter search",
            "canonical_train_size": len(train_y),
            "canonical_test_size": len(test_y),
            "train_indices": train_indices.tolist(),
            "validation_indices": val_indices.tolist(),
            "canonical_test_indices": test_indices.tolist(),
            "class_sampling": "equal per digit, without replacement",
            "train_validation_disjoint": True,
            "test_used_for_selection": False,
            "architecture": ARCHITECTURE,
            "neuron_params": neuron_params,
            "trained_source_blocks": list(TRAINED_BLOCKS),
            "fixed_pool_source_blocks": list(POOL_BLOCKS),
            "neural_source_blocks": list(NEURAL_BLOCKS),
            "split_seed": args.split_seed,
            "model_seed": args.seed,
            "encoder": "binary Bernoulli p=0.4*uint8_pixel/255, row-major 28x28 pixels",
            "bins": args.bins,
            "timestep": 1.0,
            "training_encoding_seed": "model_seed + 1000000*epoch + batch_index",
            "training_order_seed": "model_seed + 1000 + epoch",
            "validation_encoding_seed": args.seed + 200000,
            "test_encoding_seed": args.seed + 300000,
            "source_inference_batch_size": args.batch_size,
            "checkpoint_selection": "validation accuracy, lower validation loss, earlier epoch",
            "config": vars(args) | {"data_dir": str(args.data_dir)},
            "preflight_notes": synthetic_preflight(),
            **provenance,
        }
        write_json(output / "protocol.json", protocol)
    else:
        verify_frozen_protocol(protocol, provenance)
        train_indices = np.asarray(protocol["train_indices"], dtype=np.int64)
        val_indices = np.asarray(
            protocol["validation_indices"], dtype=np.int64
        )
        test_indices = np.asarray(
            protocol["canonical_test_indices"], dtype=np.int64
        )
    protocol_hash = sha256(output / "protocol.json")
    validation_inputs = encode_spatial(
        train_x[val_indices], bins=args.bins, seed=args.seed + 200000
    )
    if summary is None:
        _, summary = train(
            args,
            output,
            train_x[train_indices],
            train_y[train_indices],
            validation_inputs,
            train_y[val_indices],
        )
    # Deployment always reads the selected persisted checkpoint, not an
    # unpersisted model, and verifies the full weight audit from initial tensors.
    verify_frozen_protocol(protocol, frozen_provenance(args.data_dir, engine))
    model = restore_model(output, summary)
    test_inputs = encode_spatial(
        test_x[test_indices], bins=args.bins, seed=args.seed + 300000
    )
    torch.save(
        {
            "validation_inputs": validation_inputs.to(torch.uint8),
            "validation_labels": torch.tensor(train_y[val_indices]),
            "test_inputs": test_inputs.to(torch.uint8),
            "test_labels": torch.tensor(test_y[test_indices]),
        },
        output / "validation_inputs.pt",
    )
    from lacuna.importers.slayer import (
        import_slayer_feedforward,
        validate_slayer_feedforward,
    )

    transfer_started = time.perf_counter()
    print(
        json.dumps(
            {"stage": "export", "selected_epoch": summary["selected_epoch"]}
        ),
        flush=True,
    )
    imported = import_slayer_feedforward(
        model,
        input_shape=(1, 28, 28),
        timestep=1.0,
        acknowledge_quantization=True,
        name="official-slayer-pool-mnist",
    )
    if len(imported.metadata["layers"]) != len(NEURAL_BLOCKS):
        raise RuntimeError(
            "deployment did not retain every neural layer including pools"
        )
    imported.deployment.network.save(output / "network.json")
    comparisons = {}
    for split, inputs in (
        ("validation", validation_inputs),
        ("test", test_inputs),
    ):
        print(
            json.dumps(
                {
                    "stage": "cross-engine replay",
                    "split": split,
                    "samples": len(inputs),
                }
            ),
            flush=True,
        )
        comparison = validate_slayer_feedforward(
            model,
            imported,
            inputs,
            source_batch_size=args.batch_size,
        )
        write_json(output / f"{split}_comparison.json", comparison)
        if comparison["off_grid_spikes"]:
            raise RuntimeError(f"unexpected off-grid spikes in {split} replay")
        if len(comparison["layers"]) != len(NEURAL_BLOCKS):
            raise RuntimeError("comparison omitted a neural layer")
        comparisons[split] = comparison
    with engine.compile(imported.deployment.network) as simulation:
        simulation.save_compiled_graph_image(output / "network.lcbin")
        execution_path = simulation.preferred_execution_path
    export_seconds = time.perf_counter() - transfer_started
    verify_frozen_protocol(protocol, frozen_provenance(args.data_dir, engine))
    if sha256(output / "protocol.json") != protocol_hash:
        raise ValueError("protocol changed during the pilot")
    test_comparison = comparisons["test"]
    graph = imported.deployment.network.graph
    report = {
        "task": "MNIST official SLAYER convolution+pool to ordinary Lacuna, subset pilot",
        "architecture": ARCHITECTURE,
        "preflight_notes": protocol["preflight_notes"],
        "graph_neurons_including_input_relays": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "torch.optim.Adam, both convolutional layers and dense layer trained, pooling fixed",
        "loss": "official slayer.loss.SpikeRate(0.2,0.02,reduction='mean')",
        **summary,
        "official_test": classification_metrics(
            test_y[test_indices], test_comparison["source_predictions"]
        ),
        "lacuna_test": classification_metrics(
            test_y[test_indices], test_comparison["lacuna_predictions"]
        ),
        "export_and_comparison_seconds": export_seconds,
        "test_prediction_agreement": test_comparison["prediction_agreement"],
        "test_spikes_match": test_comparison["exact_spike_match_on_batch"],
        "validation_spikes_match": comparisons["validation"][
            "exact_spike_match_on_batch"
        ],
        "test_layer_comparison": test_comparison["layers"],
        "validation_layer_comparison": comparisons["validation"]["layers"],
        "import": imported.metadata,
        "execution_path": execution_path,
        **provenance["source_provenance"],
        "official_sources_sha256": provenance["official_sources_sha256"],
        "workflow_sources_sha256": provenance["workflow_sources_sha256"],
        "lacuna_library_sha256": provenance["lacuna_library_sha256"],
        "frozen_provenance_verified_after_training_and_replay": True,
        "files_sha256": {
            path.name: sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    write_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "output": str(output),
                "official_test_accuracy": report["official_test"]["accuracy"],
                "lacuna_test_accuracy": report["lacuna_test"]["accuracy"],
                "test_prediction_agreement": report[
                    "test_prediction_agreement"
                ],
                "test_layer_comparison": report["test_layer_comparison"],
                "training_seconds": summary["training_seconds"],
                "export_and_comparison_seconds": export_seconds,
                "pool_weights_unchanged": report["pool_weights_unchanged"],
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
    result.add_argument("--train-samples", type=int, default=10000)
    result.add_argument("--validation-samples", type=int, default=1000)
    result.add_argument("--test-samples", type=int, default=1000)
    result.add_argument("--epochs", type=int, default=10)
    result.add_argument("--bins", type=int, default=32)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=0.003)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--split-seed", type=int, default=137)
    result.add_argument("--output")
    result.add_argument(
        "--resume", help="resume export from a completed training directory"
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
