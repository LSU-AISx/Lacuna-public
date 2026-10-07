from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pytest

from lacuna import (
    AdaptiveLIF,
    Engine,
    LIF,
    LocallyConnected,
    NativeEventEncoder,
    NetworkBuilder,
    NeuronPolarity,
    TTFSEncoder,
)
from lacuna.errors import ResolutionError
from lacuna.ffi import CoreEvaluator, StateInspectionRequest


def _network():
    builder = NetworkBuilder("visualized")
    neurons = builder.population(
        "neurons",
        3,
        LIF(),
        polarity=(
            NeuronPolarity.EXCITATORY,
            NeuronPolarity.INHIBITORY,
            NeuronPolarity.EXCITATORY,
        ),
    )
    builder.connect(
        neurons,
        neurons,
        pattern=LocallyConnected(radius=1),
        weight=(2.0, 1.0, 3.0, 2.0),
        delay=0.5,
    )
    builder.input("spike", neurons[0], encoder=NativeEventEncoder())
    builder.input(
        "encoded", neurons[1], encoder=TTFSEncoder(0.1, 0.8, amplitude=20.0)
    )
    builder.input("drive", neurons[2], parameter="drive")
    builder.output("readout", neurons[2])
    return builder.build()


def test_live_visualizer_projects_graph_injects_and_samples_selected_state(
    core: CoreEvaluator,
) -> None:
    with Engine(core._lib._name).compile(_network()) as compiled:
        with compiled.visualizer(
            3.0, step=1.0, sample_interval=0.1
        ) as viewer:
            graph = viewer.graph_payload()
            assert graph["name"] == "visualized"
            assert len(graph["nodes"]) == 3
            assert len(graph["edges"]) == 4
            assert [node["polarity"] for node in graph["nodes"]] == [
                "EXCITATORY",
                "INHIBITORY",
                "EXCITATORY",
            ]
            assert [edge["weight"] for edge in graph["edges"]] == [
                2.0,
                -1.0,
                -3.0,
                2.0,
            ]
            assert [edge["magnitude"] for edge in graph["edges"]] == [
                2.0,
                1.0,
                3.0,
                2.0,
            ]
            assert {item["injection_kind"] for item in graph["inputs"]} == {
                "spike",
                "presentation",
                "drive",
            }
            assert graph["outputs"] == [
                {"id": "readout", "node": 2, "decoder": None}
            ]
            assert graph["nodes"][0]["is_input"] is True
            assert graph["nodes"][0]["is_output"] is False
            assert graph["nodes"][0]["input_ports"] == ["spike"]
            assert graph["nodes"][0]["layout_role"] == "input"
            assert graph["nodes"][2]["is_input"] is True
            assert graph["nodes"][2]["is_output"] is True
            assert graph["nodes"][2]["output_ports"] == ["readout"]
            assert graph["nodes"][2]["layout_role"] == "input_output"
            assert all(0.0 <= item["x"] <= 1.0 for item in graph["nodes"])
            assert all(0.0 <= item["y"] <= 1.0 for item in graph["nodes"])

            selected = viewer.select(0)
            assert selected["state_names"] == ["v"]
            assert viewer.inject(
                {
                    "port": "spike",
                    "kind": "spike",
                    "amplitude": 20.0,
                    "count": 1,
                }
            )["scheduled"] == 1
            first = viewer.advance()
            assert first["frontier"] == 1.0
            assert first["spikes"] == [{"t": 0.0, "node": 0}]
            assert len(first["states"]) == 10
            assert first["states"][0]["t"] == 0.0
            assert first["states"][-1]["t"] == pytest.approx(0.9)
            assert all(item["t"] < 1.0 for item in first["states"])
            assert all(item["node"] == 0 for item in first["states"])

            assert viewer.inject(
                {
                    "port": "encoded",
                    "kind": "presentation",
                    "value": 1.0,
                    "duration": 1.0,
                }
            )["scheduled"] == 1
            assert viewer.inject(
                {
                    "port": "drive",
                    "kind": "drive",
                    "value": 18.0,
                    "duration": 0.5,
                }
            )["scheduled"] == 2
            second = viewer.advance()
            assert second["frontier"] == 2.0
            assert any(item["node"] == 1 for item in second["spikes"])


def test_live_visualizer_samples_every_state_of_complex_neuron(
    core: CoreEvaluator,
) -> None:
    builder = NetworkBuilder("adaptive-visualized")
    neuron = builder.neuron("adaptive", AdaptiveLIF())
    builder.input("kick", neuron, encoder=NativeEventEncoder())
    network = builder.build()

    with Engine(core._lib._name).compile(network) as compiled:
        with compiled.visualizer(
            1.0, step=0.5, sample_interval=0.1
        ) as viewer:
            graph = viewer.graph_payload()
            assert graph["nodes"][0]["state_names"] == ["v", "w"]
            assert viewer.select(0)["state_names"] == ["v", "w"]
            payload = viewer.advance()

    assert payload["states"]
    assert all(sample["names"] == ["v", "w"] for sample in payload["states"])
    assert all(len(sample["values"]) == 2 for sample in payload["states"])


def test_incremental_high_level_run_accepts_live_inspection_requests(
    core: CoreEvaluator,
) -> None:
    with Engine(core._lib._name).compile(_network()) as compiled:
        with compiled.start_run(2.0) as run:
            first = run.advance(
                1.0,
                inspections=(
                    StateInspectionRequest(0.25, 0),
                    StateInspectionRequest(0.75, 2),
                ),
            )
            final = run.finish(
                inspections=(StateInspectionRequest(2.0, 1),)
            )
    assert [(item.t, item.node) for item in first.states] == [
        (0.25, 0),
        (0.75, 2),
    ]
    assert [(item.t, item.node) for item in final.states] == [(2.0, 1)]


def test_visualizer_serves_tokenized_local_application(core: CoreEvaluator) -> None:
    with Engine(core._lib._name).compile(_network()) as compiled:
        with compiled.visualizer(2.0) as viewer:
            try:
                viewer.start(open_browser=False)
            except PermissionError:
                pytest.skip("local socket binding is unavailable in this sandbox")
            with urlopen(viewer.url, timeout=2.0) as response:
                html = response.read().decode("utf-8")
                policy = response.headers["Content-Security-Policy"]
            assert "Lacuna Network Visualizer" in html
            assert "State variables and spike times" in html
            assert "spikesByNode" in html
            assert "Spike at t =" in html
            assert "inter-spike rate" in html
            assert "spikes/unit" in html
            assert "maintainPlaybackBuffer" in html
            assert "recordingPausedForBuffer" in html
            assert "state.recorder.pause()" in html
            assert "__LACUNA_BASE_PATH__" not in html
            assert "connect-src 'self'" in policy

            request = Request(
                viewer.url + "api/select",
                data=json.dumps({"node": 1}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2.0) as response:
                payload = json.loads(response.read())
            assert payload == {"selected": 1, "state_names": ["v"]}

            close_request = Request(
                viewer.url + "api/close",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(close_request, timeout=2.0) as response:
                close_payload = json.loads(response.read())
            viewer.wait()
            assert close_payload == {"closed": True}
            assert viewer._closed is True
            assert viewer._thread is not None
            assert viewer._thread.is_alive() is False


def test_visualizer_rejects_incompatible_port_input(core: CoreEvaluator) -> None:
    with Engine(core._lib._name).compile(_network()) as compiled:
        with compiled.visualizer(2.0) as viewer:
            with pytest.raises(ResolutionError, match="does not accept native spikes"):
                viewer.inject({"port": "encoded", "kind": "spike"})
