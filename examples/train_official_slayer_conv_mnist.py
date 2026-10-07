"""Train a small official SLAYER CNN and replay its spikes in ordinary Lacuna."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from lava.lib.dl import slayer

from lacuna import Engine
from lacuna.experiments import load_mnist
from examples.train_official_slayer_mnist import (
    classification_metrics,
    encode_images,
    evaluate,
    stratified_groups,
    write_json,
)


ARCHITECTURE = [
    {"type": "input", "shape": [1, 28, 28]},
    {
        "type": "conv_lif",
        "channels": [1, 4],
        "kernel": 5,
        "stride": 2,
        "padding": 0,
        "output_shape": [4, 12, 12],
    },
    {
        "type": "conv_lif",
        "channels": [4, 8],
        "kernel": 3,
        "stride": 2,
        "padding": 0,
        "output_shape": [8, 5, 5],
    },
    {"type": "flatten", "order": "channel, row, column"},
    {"type": "dense_lif", "neurons": [200, 10]},
]


def build_model():
    """Use shared convolutional weights only in the official training model."""

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
        weight_scale=2.0,
        pre_hook_fx=None,
        weight_norm=False,
        delay=False,
        delay_shift=False,
    )
    return (
        torch.nn.Sequential(
            slayer.block.cuba.Conv(params, 1, 4, 5, stride=2, **options),
            slayer.block.cuba.Conv(params, 4, 8, 3, stride=2, **options),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(params, 200, 10, **options),
        ),
        params,
    )


def encode_spatial(images, *, bins, seed):
    """Preserve the dense pilot's exact pixel encoding and restore image axes."""

    spikes = encode_images(images, bins=bins, seed=seed)
    return spikes.reshape(len(spikes), 1, 28, 28, bins)


def source_provenance():
    source_root = Path(slayer.__file__).resolve().parents[5]
    return {
        "official_source_commit": subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "official_source_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                text=True,
            ).strip()
        ),
        "torch_version": str(torch.__version__),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def data_hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(directory).glob("*.gz"))
    }


def training_source_hashes():
    worktree = Path(__file__).resolve().parents[1]
    return {
        name: hashlib.sha256((worktree / name).read_bytes()).hexdigest()
        for name in (
            "examples/train_official_slayer_conv_mnist.py",
            "examples/train_official_slayer_mnist.py",
            "src/lacuna/experiments/mnist_rstdp.py",
        )
    }


def train(args, output, images, labels, validation_inputs, validation_y):
    model, neuron_params = build_model()
    learned_blocks = [block for block in model if hasattr(block, "synapse")]
    initial_weights = [
        block.synapse.weight.detach().clone() for block in learned_blocks
    ]
    loss_function = slayer.loss.SpikeRate(
        true_rate=0.2, false_rate=0.02, reduction="mean"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    initial_validation = evaluate(
        model, validation_inputs, validation_y, loss_function, args.batch_size
    )
    best_key, best_state, best_epoch, best_validation = None, None, None, None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = np.random.default_rng(args.seed + 1000 + epoch).permutation(len(labels))
        weighted_loss = 0.0
        for batch_index, start in enumerate(range(0, len(order), args.batch_size)):
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
        validation = evaluate(
            model, validation_inputs, validation_y, loss_function, args.batch_size
        )
        key = (validation["accuracy"], -validation["loss"], -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_validation = key, epoch, validation
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "neuron_params": neuron_params,
                    "architecture": ARCHITECTURE,
                    "selected_epoch": epoch,
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
    summary = {
        "epochs_run": args.epochs,
        "selected_epoch": best_epoch,
        "initial_validation": initial_validation,
        "selected_validation": best_validation,
        "training_seconds": training_seconds,
        "weight_change_l2_by_layer": [
            float(torch.linalg.vector_norm(block.synapse.weight.detach() - before))
            for block, before in zip(learned_blocks, initial_weights)
        ],
        "trainable_weight_count_by_layer": [
            block.synapse.weight.numel() for block in learned_blocks
        ],
        "checkpoint_sha256": hashlib.sha256(
            (output / "official_checkpoint.pt").read_bytes()
        ).hexdigest(),
    }
    write_json(output / "training_summary.json", summary)
    return model.eval(), summary


def run(args):
    """Keep training and test-independent selection separate from deployment."""

    if args.resume:
        output = Path(args.resume)
        if (output / "report.json").exists():
            raise ValueError("this pilot already has a completed report")
        protocol = json.loads((output / "protocol.json").read_text())
        args = argparse.Namespace(**protocol["config"])
        args.data_dir = Path(args.data_dir)
        summary = json.loads((output / "training_summary.json").read_text())
    else:
        if any(value <= 0 for value in (args.epochs, args.bins, args.batch_size)):
            raise ValueError("epochs, bins, and batch size must be positive")
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            raise ValueError("learning rate must be finite and positive")
        output = (
            Path(args.output)
            if args.output
            else Path("artifacts/official-slayer-conv-mnist")
            / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
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
    train_x, train_y, test_x, test_y = load_mnist(args.data_dir, download=False)
    provenance = source_provenance()
    if provenance["official_source_dirty"]:
        raise ValueError("official SLAYER source must be unmodified")
    if protocol is None:
        train_indices, val_indices = stratified_groups(
            train_y, (args.train_samples, args.validation_samples), seed=args.split_seed
        )
        (test_indices,) = stratified_groups(
            test_y, (args.test_samples,), seed=args.split_seed + 1
        )
        protocol = {
            "canonical_train_size": len(train_y),
            "canonical_test_size": len(test_y),
            "train_indices": train_indices.tolist(),
            "validation_indices": val_indices.tolist(),
            "canonical_test_indices": test_indices.tolist(),
            "class_sampling": "equal per digit, without replacement",
            "train_validation_disjoint": True,
            "test_used_for_selection": False,
            "architecture": ARCHITECTURE,
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
            "dataset_sha256": data_hashes(args.data_dir),
            "source_provenance": provenance,
            "training_sources_sha256": training_source_hashes(),
        }
        write_json(output / "protocol.json", protocol)
    else:
        if protocol["architecture"] != ARCHITECTURE:
            raise ValueError("architecture changed since training")
        if protocol["training_sources_sha256"] != training_source_hashes():
            raise ValueError("training code changed since training")
        if protocol["source_provenance"] != provenance:
            raise ValueError("source or environment changed since training")
        if protocol["dataset_sha256"] != data_hashes(args.data_dir):
            raise ValueError("dataset changed since training")
        actual = hashlib.sha256(
            (output / "official_checkpoint.pt").read_bytes()
        ).hexdigest()
        if actual != summary["checkpoint_sha256"]:
            raise ValueError("selected checkpoint changed since training")
        train_indices = np.asarray(protocol["train_indices"], dtype=np.int64)
        val_indices = np.asarray(protocol["validation_indices"], dtype=np.int64)
        test_indices = np.asarray(protocol["canonical_test_indices"], dtype=np.int64)
    validation_inputs = encode_spatial(
        train_x[val_indices], bins=args.bins, seed=args.seed + 200000
    )
    if summary is None:
        model, summary = train(
            args,
            output,
            train_x[train_indices],
            train_y[train_indices],
            validation_inputs,
            train_y[val_indices],
        )
    else:
        model, _ = build_model()
        checkpoint = torch.load(output / "official_checkpoint.pt", weights_only=True)
        if checkpoint["architecture"] != ARCHITECTURE:
            raise ValueError("checkpoint architecture does not match the pilot")
        if checkpoint["selected_epoch"] != summary["selected_epoch"]:
            raise ValueError("checkpoint epoch does not match the selected epoch")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
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
        json.dumps({"stage": "export", "selected_epoch": summary["selected_epoch"]}),
        flush=True,
    )
    imported = import_slayer_feedforward(
        model,
        input_shape=(1, 28, 28),
        timestep=1.0,
        acknowledge_quantization=True,
        name="official-slayer-conv-mnist",
    )
    imported.deployment.network.save(output / "network.json")
    comparisons = {}
    for name, inputs in (("validation", validation_inputs), ("test", test_inputs)):
        print(
            json.dumps(
                {"stage": "cross-engine replay", "split": name, "samples": len(inputs)}
            ),
            flush=True,
        )
        comparisons[name] = validate_slayer_feedforward(
            model, imported, inputs, source_batch_size=args.batch_size
        )
        write_json(output / f"{name}_comparison.json", comparisons[name])
    engine = Engine()
    with engine.compile(imported.deployment.network) as simulation:
        simulation.save_compiled_graph_image(output / "network.lcbin")
        execution_path = simulation.preferred_execution_path
    export_seconds = time.perf_counter() - transfer_started
    test_comparison = comparisons["test"]
    graph = imported.deployment.network.graph
    worktree = Path(__file__).resolve().parents[1]
    report = {
        "task": "MNIST official SLAYER convolutional to ordinary Lacuna, subset pilot",
        "architecture": ARCHITECTURE,
        "graph_neurons_including_input_relays": len(graph.nodes),
        "graph_edges": len(graph.edges),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "test_samples": len(test_indices),
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "torch.optim.Adam, both convolutional layers and dense layer trained",
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
        **provenance,
        "lacuna_library_sha256": hashlib.sha256(
            Path(engine.core._lib._name).read_bytes()
        ).hexdigest(),
        "workflow_sources_sha256": {
            name: hashlib.sha256((worktree / name).read_bytes()).hexdigest()
            for name in (
                "examples/train_official_slayer_conv_mnist.py",
                "examples/train_official_slayer_mnist.py",
                "src/lacuna/importers/slayer.py",
                "src/lacuna/importers/dense_lif.py",
                "src/lacuna/importers/feedforward_lif.py",
                "src/lacuna/experiments/mnist_rstdp.py",
            )
        },
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
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
                "test_prediction_agreement": report["test_prediction_agreement"],
                "test_layer_comparison": report["test_layer_comparison"],
                "training_seconds": summary["training_seconds"],
                "export_and_comparison_seconds": export_seconds,
            }
        ),
        flush=True,
    )
    return output


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data-dir", type=Path, default=Path("artifacts/mnist-data"))
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
