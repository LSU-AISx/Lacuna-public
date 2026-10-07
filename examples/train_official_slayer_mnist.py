"""A bounded MNIST pilot using official SLAYER and ordinary Lacuna inference."""

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
from lacuna.importers.slayer import import_slayer_dense, validate_slayer_dense


def stratified_groups(labels, sizes, *, seed):
    """Select disjoint equal-per-digit groups from one canonical partition."""

    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, np.arange(10)).all():
        raise ValueError("expected a vector of digit labels")
    if not sizes or any(
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or size % 10
        for size in sizes
    ):
        raise ValueError("group sizes must be positive multiples of ten")
    generator = np.random.default_rng(seed)
    groups = [[] for _ in sizes]
    for digit in range(10):
        candidates = np.flatnonzero(labels == digit)
        if sum(sizes) // 10 > len(candidates):
            raise ValueError(
                "not enough examples per digit for disjoint groups"
            )
        generator.shuffle(candidates)
        offset = 0
        for group, size in zip(groups, sizes):
            count = size // 10
            group.extend(candidates[offset : offset + count].tolist())
            offset += count
    for group in groups:
        generator.shuffle(group)
    return tuple(np.asarray(group, dtype=np.int64) for group in groups)


def encode_images(images, *, bins, seed):
    """Produce binary spikes at p=0.4*pixel/255, with no background current."""

    values = torch.as_tensor(np.array(images, copy=True), dtype=torch.float32)
    if (
        values.ndim != 2
        or values.shape[1] != 784
        or values.shape[0] == 0
        or not bool(torch.all((values >= 0) & (values <= 255)))
    ):
        raise ValueError("images must be a nonempty [N,784] matrix in [0,255]")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    probabilities = 0.4 * values / 255.0
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(*values.shape, bins, generator=generator)
        < probabilities[..., None]
    ).float()


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if (
        labels.ndim != 1
        or labels.shape != predictions.shape
        or not len(labels)
        or not np.isin(labels, np.arange(10)).all()
        or not np.isin(predictions, np.arange(10)).all()
    ):
        raise ValueError("expected equal nonempty vectors of digit labels")
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (labels.astype(int), predictions.astype(int)), 1)
    totals = confusion.sum(axis=1)
    correct = int(np.trace(confusion))
    return {
        "samples": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "per_digit_recall": [
            float(confusion[k, k] / totals[k]) if totals[k] else None
            for k in range(10)
        ],
        "confusion_matrix": confusion.tolist(),
    }


def build_model(hidden):
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
    return (
        torch.nn.Sequential(
            *[
                slayer.block.cuba.Dense(
                    params,
                    source,
                    target,
                    weight_scale=2.0,
                    pre_hook_fx=None,
                    weight_norm=False,
                    delay=False,
                    delay_shift=False,
                )
                for source, target in ((784, hidden), (hidden, 10))
            ]
        ),
        params,
    )


def evaluate(model, inputs, labels, loss_function, batch_size):
    model.eval()
    predictions = []
    weighted_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            batch_y = torch.tensor(
                labels[start : start + batch_size], dtype=torch.long
            )
            spikes = model(inputs[start : start + batch_size])
            predictions.extend(slayer.classifier.Rate.predict(spikes).tolist())
            weighted_loss += float(
                loss_function(spikes, batch_y).item()
            ) * len(batch_y)
    return {
        **classification_metrics(labels, predictions),
        "loss": weighted_loss / len(labels),
    }


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run(args):
    if any(
        value <= 0
        for value in (args.epochs, args.hidden, args.bins, args.batch_size)
    ):
        raise ValueError(
            "epochs, hidden size, bins, and batch size must be positive"
        )
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    output = (
        Path(args.output)
        if args.output
        else Path("artifacts/official-slayer-mnist")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    print(
        json.dumps({"output": str(output), "stage": "loading verified data"}),
        flush=True,
    )
    train_images, train_labels, test_images, test_labels = load_mnist(
        args.data_dir, download=False
    )
    train_indices, validation_indices = stratified_groups(
        train_labels,
        (args.train_samples, args.validation_samples),
        seed=args.split_seed,
    )
    (test_indices,) = stratified_groups(
        test_labels, (args.test_samples,), seed=args.split_seed + 1
    )
    train_x, train_y = train_images[train_indices], train_labels[train_indices]
    validation_x, validation_y = (
        train_images[validation_indices],
        train_labels[validation_indices],
    )
    validation_inputs = encode_images(
        validation_x, bins=args.bins, seed=args.seed + 200000
    )
    protocol = {
        "canonical_train_size": len(train_labels),
        "canonical_test_size": len(test_labels),
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "canonical_test_indices": test_indices.tolist(),
        "class_sampling": "equal per digit, without replacement",
        "train_validation_disjoint": True,
        "test_used_for_selection": False,
        "split_seed": args.split_seed,
        "model_seed": args.seed,
        "encoder": "binary Bernoulli p=0.4*uint8_pixel/255, row-major 28x28 pixels",
        "bins": args.bins,
        "timestep": 1.0,
        "training_encoding_seed": "model_seed + 1000000*epoch + batch_index",
        "training_order_seed": "model_seed + 1000 + epoch",
        "validation_encoding_seed": args.seed + 200000,
        "test_encoding_seed": args.seed + 300000,
        "checkpoint_selection": "each epoch: validation accuracy, lower validation loss, earlier epoch",
        "config": vars(args) | {"data_dir": str(args.data_dir)},
        "dataset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(args.data_dir).glob("*.gz"))
        },
    }
    write_json(output / "protocol.json", protocol)
    model, neuron_params = build_model(args.hidden)
    initial_weights = [
        block.synapse.weight.detach().clone() for block in model
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
        order = np.random.default_rng(args.seed + 1000 + epoch).permutation(
            len(train_y)
        )
        weighted_train_loss = 0.0
        for batch_index, start in enumerate(
            range(0, len(order), args.batch_size)
        ):
            chosen = order[start : start + args.batch_size]
            inputs = encode_images(
                train_x[chosen],
                bins=args.bins,
                seed=args.seed + 1000000 * epoch + batch_index,
            )
            targets = torch.tensor(train_y[chosen], dtype=torch.long)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite official SLAYER loss")
            loss.backward()
            optimizer.step()
            weighted_train_loss += float(loss.item()) * len(chosen)
        validation = evaluate(
            model,
            validation_inputs,
            validation_y,
            loss_function,
            args.batch_size,
        )
        key = (validation["accuracy"], -validation["loss"], -epoch)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_validation = key, epoch, validation
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "model_state_dict": best_state,
                    "neuron_params": neuron_params,
                    "architecture": [784, args.hidden, 10],
                    "selected_epoch": epoch,
                },
                output / "official_checkpoint.pt",
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": weighted_train_loss / len(train_y),
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
    model.eval()
    test_inputs = encode_images(
        test_images[test_indices], bins=args.bins, seed=args.seed + 300000
    )
    torch.save(
        {
            "validation_inputs": validation_inputs.to(torch.uint8),
            "validation_labels": torch.tensor(validation_y),
            "test_inputs": test_inputs.to(torch.uint8),
            "test_labels": torch.tensor(test_labels[test_indices]),
        },
        output / "validation_inputs.pt",
    )
    print(
        json.dumps(
            {
                "stage": "export and spike comparison",
                "selected_epoch": best_epoch,
            }
        ),
        flush=True,
    )
    transfer_started = time.perf_counter()
    imported = import_slayer_dense(
        model, acknowledge_quantization=True, name="official-slayer-mnist"
    )
    imported.deployment.network.save(output / "network.json")
    comparisons = {}
    for name, inputs in (
        ("validation", validation_inputs),
        ("test", test_inputs),
    ):
        print(
            json.dumps(
                {
                    "stage": "cross-engine replay",
                    "split": name,
                    "samples": len(inputs),
                }
            ),
            flush=True,
        )
        comparisons[name] = validate_slayer_dense(model, imported, inputs)
        write_json(output / f"{name}_comparison.json", comparisons[name])
    engine = Engine()
    with engine.compile(imported.deployment.network) as simulation:
        simulation.save_compiled_graph_image(output / "network.lcbin")
        execution_path = simulation.preferred_execution_path
    test_comparison = comparisons["test"]
    source_test = classification_metrics(
        test_labels[test_indices], test_comparison["source_predictions"]
    )
    target_test = classification_metrics(
        test_labels[test_indices], test_comparison["lacuna_predictions"]
    )
    source_root = Path(slayer.__file__).resolve().parents[5]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(source_root), "status", "--porcelain"], text=True
        ).strip()
    )
    worktree = Path(__file__).resolve().parents[1]
    report = {
        "task": "MNIST official SLAYER to ordinary Lacuna, subset pilot",
        "architecture": [784, args.hidden, 10],
        "train_samples": len(train_y),
        "validation_samples": len(validation_y),
        "test_samples": len(test_indices),
        "epochs_run": args.epochs,
        "selected_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "optimizer": "torch.optim.Adam, both layers trained",
        "loss": "official slayer.loss.SpikeRate(0.2,0.02,reduction='mean')",
        "initial_validation": initial_validation,
        "selected_validation": best_validation,
        "official_test": source_test,
        "lacuna_test": target_test,
        "training_seconds": training_seconds,
        "export_and_comparison_seconds": time.perf_counter()
        - transfer_started,
        "weight_change_l2_by_layer": [
            float(
                torch.linalg.vector_norm(
                    block.synapse.weight.detach() - before
                ).item()
            )
            for block, before in zip(model, initial_weights)
        ],
        "test_prediction_agreement": test_comparison["prediction_agreement"],
        "test_spikes_match": test_comparison["exact_spike_match_on_batch"],
        "validation_spikes_match": comparisons["validation"][
            "exact_spike_match_on_batch"
        ],
        "test_layer_comparison": test_comparison["layers"],
        "validation_layer_comparison": comparisons["validation"]["layers"],
        "import": imported.metadata,
        "execution_path": execution_path,
        "official_source_commit": source_commit,
        "official_source_dirty": source_dirty,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lacuna_library_sha256": hashlib.sha256(
            Path(engine.core._lib._name).read_bytes()
        ).hexdigest(),
        "workflow_sources_sha256": {
            name: hashlib.sha256((worktree / name).read_bytes()).hexdigest()
            for name in (
                "examples/train_official_slayer_mnist.py",
                "src/lacuna/importers/slayer.py",
                "src/lacuna/importers/dense_lif.py",
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
                "official_test_accuracy": source_test["accuracy"],
                "lacuna_test_accuracy": target_test["accuracy"],
                "test_prediction_agreement": test_comparison[
                    "prediction_agreement"
                ],
                "test_spikes_match": report["test_spikes_match"],
                "test_layer_comparison": test_comparison["layers"],
                "training_seconds": training_seconds,
            }
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("artifacts/mnist-data")
    )
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=137)
    parser.add_argument("--output")
    run(parser.parse_args())
