"""Train with official SLAYER, then validate a normal Lacuna deployment.

The small noisy XOR task checks the workflow, not MNIST performance. Both
layers learn through Lava-DL's own neurons, surrogate gradients, and loss.
"""

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

import torch
from lava.lib.dl import slayer

from lacuna import Engine
from lacuna.importers.slayer import import_slayer_dense, validate_slayer_dense


def xor_spikes(repeats: int, bins: int, seed: int):
    """Give each Boolean input a true and false input neuron."""

    pairs = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).repeat(repeats, 1)
    labels = pairs[:, 0] ^ pairs[:, 1]
    active = torch.stack(
        (1 - pairs[:, 0], pairs[:, 0], 1 - pairs[:, 1], pairs[:, 1]), dim=1
    )
    probability = 0.02 + 0.43 * active.float()
    generator = torch.Generator().manual_seed(seed)
    spikes = (
        torch.rand(len(labels), 4, bins, generator=generator)
        < probability[..., None]
    ).float()
    return spikes, labels


def accuracy(model, inputs, labels):
    with torch.no_grad():
        predictions = slayer.classifier.Rate.predict(model(inputs))
    return float((predictions == labels).float().mean().item())


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    output = (
        Path(args.output)
        if args.output
        else Path("artifacts/official-slayer-lif")
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(parents=True, exist_ok=False)
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

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
    model = torch.nn.Sequential(
        slayer.block.cuba.Dense(
            params,
            4,
            16,
            weight_scale=2.0,
            pre_hook_fx=None,
            weight_norm=False,
            delay=False,
            delay_shift=False,
        ),
        slayer.block.cuba.Dense(
            params,
            16,
            2,
            weight_scale=2.0,
            pre_hook_fx=None,
            weight_norm=False,
            delay=False,
            delay_shift=False,
        ),
    )
    train_x, train_y = xor_spikes(32, 48, args.seed + 100)
    validation_x, validation_y = xor_spikes(16, 48, args.seed + 200)
    test_x, test_y = xor_spikes(32, 48, args.seed + 300)
    model.eval()
    initial_validation_accuracy = accuracy(model, validation_x, validation_y)
    initial_weights = [
        block.synapse.weight.detach().clone() for block in model
    ]
    loss_function = slayer.loss.SpikeRate(
        true_rate=0.25, false_rate=0.02, reduction="mean"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_key = (-1.0, -float("inf"))
    best_state = None
    best_epoch = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("nonfinite official training loss")
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                validation_output = model(validation_x)
                validation_loss = float(
                    loss_function(validation_output, validation_y).item()
                )
                validation_accuracy = float(
                    (
                        slayer.classifier.Rate.predict(validation_output)
                        == validation_y
                    )
                    .float()
                    .mean()
                    .item()
                )
            key = (validation_accuracy, -validation_loss)
            if key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.item()),
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )
            print(json.dumps(history[-1]), flush=True)
    training_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.eval()
    imported = import_slayer_dense(model, acknowledge_quantization=True)
    validation_comparison = validate_slayer_dense(
        model, imported, validation_x
    )
    test_comparison = validate_slayer_dense(model, imported, test_x)
    test_accuracy = float(
        (torch.tensor(test_comparison["source_predictions"]) == test_y)
        .float()
        .mean()
        .item()
    )
    lacuna_accuracy = float(
        (torch.tensor(test_comparison["lacuna_predictions"]) == test_y)
        .float()
        .mean()
        .item()
    )
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
            "neuron_params": params,
            "architecture": [4, 16, 2],
            "selected_epoch": best_epoch,
        },
        output / "official_checkpoint.pt",
    )
    torch.save(
        {
            "validation_inputs": validation_x,
            "validation_labels": validation_y,
            "test_inputs": test_x,
            "test_labels": test_y,
        },
        output / "validation_inputs.pt",
    )
    source_dir = Path(slayer.__file__).resolve().parent
    source_root = source_dir.parents[4]
    source_commit = None
    source_dirty = None
    if (source_root / ".git").exists() and source_dir == (
        source_root / "src/lava/lib/dl/slayer"
    ):
        revision = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        source_commit = revision.stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source_root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        source_dirty = bool(status.stdout.strip())
    report = {
        "task": "noisy dual-rail XOR workflow check, not an MNIST benchmark",
        "architecture": [4, 16, 2],
        "seed": args.seed,
        "epochs_run": args.epochs,
        "selected_epoch": best_epoch,
        "selection": "validation accuracy, then validation loss, evaluated every 10 epochs",
        "learning_rate": 0.01,
        "optimizer": "torch.optim.Adam",
        "loss": "official slayer.loss.SpikeRate(0.25, 0.02, reduction='mean')",
        "train_samples": len(train_y),
        "validation_samples": len(validation_y),
        "test_samples": len(test_y),
        "bins": 48,
        "initial_validation_accuracy": initial_validation_accuracy,
        "selected_validation_accuracy": best_key[0],
        "official_test_accuracy": test_accuracy,
        "lacuna_test_accuracy": lacuna_accuracy,
        "weight_change_l2_by_layer": weight_changes,
        "training_seconds": training_seconds,
        "execution_path": execution_path,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "official_source_path": str(source_dir),
        "official_source_commit": source_commit,
        "official_source_dirty": source_dirty,
        "lacuna_library_sha256": hashlib.sha256(
            library_path.read_bytes()
        ).hexdigest(),
        "workflow_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "importer_sources_sha256": {
            filename: hashlib.sha256(
                (
                    Path(__file__).resolve().parents[1]
                    / "src/lacuna/importers"
                    / filename
                ).read_bytes()
            ).hexdigest()
            for filename in ("dense_lif.py", "slayer.py")
        },
        "import": imported.metadata,
        "validation_comparison": validation_comparison,
        "test_comparison": test_comparison,
        "history": history,
    }
    report["files_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in output.iterdir()
        if path.is_file()
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "official_test_accuracy": test_accuracy,
                "lacuna_test_accuracy": lacuna_accuracy,
                "all_layer_test_spikes_match": test_comparison[
                    "exact_spike_match_on_batch"
                ],
                "weight_change_l2_by_layer": weight_changes,
                "training_seconds": training_seconds,
            }
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output")
    run(parser.parse_args())
