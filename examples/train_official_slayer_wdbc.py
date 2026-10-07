"""Train official SLAYER on WDBC and validate ordinary Lacuna deployment."""

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
from lacuna.importers.slayer import import_slayer_dense, validate_slayer_dense

WDBC_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "breast-cancer-wisconsin/wdbc.data"
)
WDBC_SHA256 = "d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a"


def load_wdbc(path: Path):
    """Load the verified local dataset used by this deployment example."""

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != WDBC_SHA256:
        raise ValueError(f"unexpected WDBC dataset SHA-256: {digest}")
    rows = np.genfromtxt(path, delimiter=",", dtype=str)
    if rows.shape != (569, 32):
        raise ValueError(f"unexpected WDBC dataset shape: {rows.shape}")
    labels = np.asarray(
        [0 if label == "B" else 1 for label in rows[:, 1]], dtype=np.int64
    )
    if np.bincount(labels, minlength=2).tolist() != [357, 212]:
        raise ValueError("unexpected WDBC class distribution")
    return rows[:, 2:].astype(float), labels, digest


def stratified_split(labels: np.ndarray, seed: int):
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for label in range(int(np.max(labels)) + 1):
        indices = np.flatnonzero(labels == label)
        generator.shuffle(indices)
        test_count = int(round(0.20 * len(indices)))
        validation_count = int(round(0.20 * len(indices)))
        test.extend(int(value) for value in indices[:test_count])
        validation.extend(
            int(value) for value in indices[test_count : test_count + validation_count]
        )
        train.extend(int(value) for value in indices[test_count + validation_count :])
    for values in (train, validation, test):
        generator.shuffle(values)
    return (
        np.asarray(train, dtype=np.int64),
        np.asarray(validation, dtype=np.int64),
        np.asarray(test, dtype=np.int64),
    )


def scale_features(train: np.ndarray, *others: np.ndarray):
    minimum = np.min(train, axis=0)
    maximum = np.max(train, axis=0)
    span = np.maximum(maximum - minimum, 1.0e-12)

    def transform(values: np.ndarray) -> np.ndarray:
        return np.clip((values - minimum) / span, 0.0, 1.0)

    return (
        (transform(train), *(transform(values) for values in others)),
        minimum,
        maximum,
    )


def encode_features(features, *, bins: int, seed: int):
    """Encode each scaled feature with independent complementary rate channels."""

    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2 or not bool(torch.all((values >= 0) & (values <= 1))):
        raise ValueError("features must be a matrix in [0, 1]")
    if bins <= 0:
        raise ValueError("bins must be positive")
    rails = torch.stack((values, 1 - values), dim=2).flatten(1)
    probabilities = 0.02 + 0.43 * rails
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(*rails.shape, bins, generator=generator)
        < probabilities[..., None]
    ).float()


def metrics(labels, predictions):
    """Return exact counts and class-balanced metrics with malignant positive."""

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.shape != predictions.shape or labels.ndim != 1:
        raise ValueError("labels and predictions must be matching vectors")
    if (
        not np.isin(labels, (0, 1)).all()
        or not np.isin(predictions, (0, 1)).all()
    ):
        raise ValueError("WDBC labels must be binary")
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    if np.any(confusion.sum(axis=1) == 0):
        raise ValueError("both classes must be present")
    recalls = confusion.diagonal() / confusion.sum(axis=1)
    correct = int(confusion.diagonal().sum())
    return {
        "samples": len(labels),
        "correct": correct,
        "accuracy": correct / len(labels),
        "balanced_accuracy": float(recalls.mean()),
        "benign_specificity": float(recalls[0]),
        "malignant_sensitivity": float(recalls[1]),
        "confusion_matrix": confusion.tolist(),
        "confusion_convention": "rows true, columns predicted, order benign/malignant",
    }


def selection_key(epoch, result):
    return (
        result["balanced_accuracy"],
        result["accuracy"],
        -result["loss"],
        -epoch,
    )


def build_model(hidden: int):
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
    blocks = [
        slayer.block.cuba.Dense(
            params,
            inputs,
            outputs,
            weight_scale=2.0,
            pre_hook_fx=None,
            weight_norm=False,
            delay=False,
            delay_shift=False,
        )
        for inputs, outputs in ((60, hidden), (hidden, 2))
    ]
    return torch.nn.Sequential(*blocks), params


def run(args):
    if args.epochs <= 0 or args.hidden <= 0 or args.bins <= 0:
        raise ValueError("epochs, hidden size, and bins must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    output = (
        Path(args.output)
        if args.output
        else Path("artifacts/official-slayer-wdbc")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)

    features, labels, digest = load_wdbc(Path(args.data))
    train, validation, test = stratified_split(labels, args.split_seed)
    (train_x, validation_x, test_x), minimum, maximum = scale_features(
        features[train], features[validation], features[test]
    )
    train_y = torch.as_tensor(labels[train], dtype=torch.long)
    validation_spikes = encode_features(
        validation_x, bins=args.bins, seed=args.seed + 200000
    )
    # The test spikes and predictions are not used until checkpoint selection ends.
    model, neuron_params = build_model(args.hidden)
    initial_weights = [
        block.synapse.weight.detach().clone() for block in model
    ]
    loss_function = slayer.loss.SpikeRate(
        true_rate=0.25, false_rate=0.02, reduction="mean"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    validation_labels = torch.as_tensor(labels[validation], dtype=torch.long)

    def evaluate_validation():
        model.eval()
        with torch.no_grad():
            spikes = model(validation_spikes)
            result = metrics(
                labels[validation],
                slayer.classifier.Rate.predict(spikes).tolist(),
            )
            result["loss"] = float(
                loss_function(spikes, validation_labels).item()
            )
        return result

    initial_validation = evaluate_validation()
    best_key = None
    best_state = None
    best_epoch = None
    best_validation = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        # New spike realizations augment training without changing held-out inputs.
        train_spikes = encode_features(
            train_x, bins=args.bins, seed=args.seed + 1000 + epoch
        )
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_spikes), train_y)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite official SLAYER training loss")
        loss.backward()
        optimizer.step()
        validation_result = evaluate_validation()
        key = selection_key(epoch, validation_result)
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_validation = validation_result
        entry = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "validation": validation_result,
        }
        history.append(entry)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": float(loss.item()),
                        "validation_accuracy": validation_result["accuracy"],
                        "validation_balanced_accuracy": validation_result[
                            "balanced_accuracy"
                        ],
                        "best_epoch": best_epoch,
                    }
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started

    model.load_state_dict(best_state)
    model.eval()
    imported = import_slayer_dense(
        model, acknowledge_quantization=True, name="official-slayer-wdbc"
    )
    test_spikes = encode_features(
        test_x, bins=args.bins, seed=args.seed + 300000
    )
    validation_comparison = validate_slayer_dense(
        model, imported, validation_spikes
    )
    test_comparison = validate_slayer_dense(model, imported, test_spikes)
    source_test = metrics(labels[test], test_comparison["source_predictions"])
    target_test = metrics(labels[test], test_comparison["lacuna_predictions"])
    weight_changes = [
        float(
            torch.linalg.vector_norm(
                block.synapse.weight.detach() - before
            ).item()
        )
        for block, before in zip(model, initial_weights)
    ]
    imported.deployment.network.save(output / "network.json")
    engine = Engine()
    library_path = Path(engine.core._lib._name).resolve()
    with engine.compile(imported.deployment.network) as simulation:
        simulation.save_compiled_graph_image(output / "network.lcbin")
        execution_path = simulation.preferred_execution_path
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "neuron_params": neuron_params,
            "architecture": [60, args.hidden, 2],
            "selected_epoch": best_epoch,
        },
        output / "official_checkpoint.pt",
    )
    torch.save(
        {
            "validation_inputs": validation_spikes,
            "validation_labels": validation_labels,
            "test_inputs": test_spikes,
            "test_labels": torch.as_tensor(labels[test], dtype=torch.long),
        },
        output / "validation_inputs.pt",
    )
    protocol = {
        "data_source": WDBC_URL,
        "data_sha256": digest,
        "class_mapping": {"benign": 0, "malignant": 1},
        "split_seed": args.split_seed,
        "train_indices": train.tolist(),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
        "scaling": "training-only min/max, held-out values clipped to [0,1]",
        "feature_minimum": minimum.tolist(),
        "feature_maximum": maximum.tolist(),
        "encoding": "adjacent x and 1-x rails, Bernoulli p=0.02+0.43*rail",
        "bins": args.bins,
        "timestep": 1.0,
        "training_encoding_seed": "model_seed + 1000 + epoch",
        "validation_encoding_seed": args.seed + 200000,
        "test_encoding_seed": args.seed + 300000,
        "checkpoint_selection": "every epoch: balanced accuracy, accuracy, lower loss, earlier epoch",
        "test_used_for_selection": False,
    }
    (output / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_dir = Path(slayer.__file__).resolve().parent
    source_root = source_dir.parents[4]
    source_commit, source_dirty = None, None
    if (
        source_root / ".git"
    ).exists() and source_dir == source_root / "src/lava/lib/dl/slayer":
        source_commit = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        source_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(source_root), "status", "--porcelain"],
                text=True,
            ).strip()
        )
    worktree = Path(__file__).resolve().parents[1]
    report = {
        "task": "WDBC official SLAYER to ordinary Lacuna, single pilot",
        "architecture": [60, args.hidden, 2],
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
        "model_seed": args.seed,
        "epochs_run": args.epochs,
        "selected_epoch": best_epoch,
        "learning_rate": args.learning_rate,
        "optimizer": "torch.optim.Adam, full-batch updates to both layers",
        "loss": "official slayer.loss.SpikeRate(0.25, 0.02, reduction='mean')",
        "initial_validation": initial_validation,
        "selected_validation": best_validation,
        "official_test": source_test,
        "lacuna_test": target_test,
        "training_seconds": training_seconds,
        "weight_change_l2_by_layer": weight_changes,
        "execution_path": execution_path,
        "official_source_commit": source_commit,
        "official_source_dirty": source_dirty,
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "lacuna_library_sha256": hashlib.sha256(
            library_path.read_bytes()
        ).hexdigest(),
        "workflow_sources_sha256": {
            filename: hashlib.sha256(
                (worktree / filename).read_bytes()
            ).hexdigest()
            for filename in (
                "examples/train_official_slayer_wdbc.py",
                "src/lacuna/importers/slayer.py",
                "src/lacuna/importers/dense_lif.py",
            )
        },
        "import": imported.metadata,
        "validation_comparison": validation_comparison,
        "test_comparison": test_comparison,
        "history": history,
        "test_labels": labels[test].tolist(),
        "files_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.iterdir()
            if path.is_file()
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "selected_epoch": best_epoch,
                "official_test": source_test,
                "lacuna_test": target_test,
                "all_layer_test_spikes_match": test_comparison[
                    "exact_spike_match_on_batch"
                ],
                "test_spike_comparison": test_comparison["layers"],
                "weight_change_l2_by_layer": weight_changes,
                "training_seconds": training_seconds,
            }
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("artifacts/datasets/wdbc.data")
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=137)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output")
    run(parser.parse_args())
