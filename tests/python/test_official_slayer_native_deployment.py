"""Native image deployment checks without a training framework dependency."""

import json
from pathlib import Path
import struct
import subprocess

import pytest

from lacuna import Engine, RunOptions, SpikeTrain
from lacuna.importers.dense_lif import DenseLIFLayer, build_dense_lif
from validation.official_slayer_native_deployment import (
    compare_outputs,
    compile_runner,
    input_episode,
    input_header,
    native_replay,
    output_episode,
    output_header,
    sha256,
    verify_artifact,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def native_runner(tmp_path_factory):
    directory = tmp_path_factory.mktemp("native-replay")
    library = Path(Engine().core._lib._name).resolve()
    return compile_runner(directory, library)


def fixture_streams(directory):
    deployment = build_dense_lif(
        (
            DenseLIFLayer(((0.7, -0.2), (0.4, 1.1)), tau_m=3.0, threshold=1.0),
            DenseLIFLayer(((0.8, 0.9),), tau_m=6.0, threshold=1.0),
        )
    )
    events = [
        [(0.0, 0), (1.0, 0), (1.0, 1), (3.0, 1)],
        [],
        [(0.0, 1), (2.0, 1)],
        [(0.0, 0), (1.0, 0)],
    ]
    options = RunOptions(output_capacity=100, queue_capacity=4096)
    nodes = len(deployment.network.graph.nodes)
    image = directory / "network.lcbin"
    inputs = directory / "inputs.bin"
    expected = directory / "expected.bin"
    with (
        Engine().compile(deployment.network) as simulation,
        inputs.open("wb") as target,
        expected.open("wb") as reference,
    ):
        simulation.save_compiled_graph_image(image)
        input_header(
            target,
            episodes=len(events),
            nodes=nodes,
            initial=[0.0] * nodes,
            duration=5.0,
            options=options,
        )
        output_header(reference, episodes=len(events), nodes=nodes, states=nodes)
        for sequence in events:
            input_episode(target, sequence)
            stimulus = {
                port: SpikeTrain(
                    times=tuple(t for t, channel in sequence if channel == index),
                    values=1.0,
                )
                for index, port in enumerate(deployment.input_ports)
            }
            result = simulation.run(5.0, inputs=stimulus, options=options)
            output_episode(reference, result)
    return image, inputs, expected


def test_native_image_replays_ordered_spikes_and_raw_state(native_runner, tmp_path):
    image, inputs, expected = fixture_streams(tmp_path)
    actual = tmp_path / "actual.bin"
    evidence = native_replay(
        native_runner, image, inputs, actual, tmp_path / "process.json"
    )
    rows = compare_outputs(expected, actual)
    assert evidence["returncode"] == 0
    assert sha256(expected) == sha256(actual)
    assert len(rows) == 4
    assert all(row["spikes_bitwise_equal"] for row in rows)
    assert all(row["final_states_bitwise_equal"] for row in rows)
    assert all(row["last_update_times_bitwise_equal"] for row in rows)
    assert rows[1]["spikes"] == 0
    assert sum(row["spikes"] for row in rows) > 0


@pytest.mark.parametrize("damage", ["truncate", "magic", "trailing", "nan", "node"])
def test_native_replay_rejects_malformed_events(native_runner, tmp_path, damage):
    image, inputs, _ = fixture_streams(tmp_path)
    payload = bytearray(inputs.read_bytes())
    if damage == "truncate":
        payload = payload[:-1]
    elif damage == "magic":
        payload[0] = 0
    elif damage == "trailing":
        payload.extend(b"extra")
    else:
        # Header is 48 bytes, followed by five initial states and a count.
        first = 48 + 5 * 8 + 4
        if damage == "nan":
            payload[first : first + 8] = struct.pack("<d", float("nan"))
        elif damage == "node":
            payload[first + 8 : first + 12] = struct.pack("<I", 9999)
    inputs.write_bytes(payload)
    process = subprocess.run(
        [str(native_runner), str(image), str(inputs), str(tmp_path / "output.bin")],
        capture_output=True,
    )
    assert process.returncode != 0
    assert process.stderr


def test_native_replay_rejects_corrupt_image(native_runner, tmp_path):
    image, inputs, _ = fixture_streams(tmp_path)
    image.write_bytes(b"not a compiled graph")
    process = subprocess.run(
        [str(native_runner), str(image), str(inputs), str(tmp_path / "output.bin")],
        capture_output=True,
    )
    assert process.returncode != 0


def test_comparison_detects_state_bit_difference(tmp_path):
    _, _, expected = fixture_streams(tmp_path)
    actual = tmp_path / "actual.bin"
    data = bytearray(expected.read_bytes())
    count = struct.unpack_from("<Q", data, 20)[0]
    data[28 + count * 12] ^= 1
    actual.write_bytes(data)
    rows = compare_outputs(expected, actual)
    assert rows[0]["spikes_bitwise_equal"]
    assert not rows[0]["final_states_bitwise_equal"]


@pytest.mark.parametrize("damage", ["artifact", "library"])
def test_artifact_hashes_guard_replay(tmp_path, damage):
    library = tmp_path / "library"
    library.write_bytes(b"runtime")
    artifact = tmp_path / "network.json"
    artifact.write_bytes(b"network")
    report = {
        "files_sha256": {"network.json": sha256(artifact)},
        "lacuna_library_sha256": sha256(library),
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    verify_artifact(tmp_path, library)
    (artifact if damage == "artifact" else library).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch|runtime library differs"):
        verify_artifact(tmp_path, library)
