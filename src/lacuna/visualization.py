"""Interactive browser visualization over a live compiled Lacuna network."""

from __future__ import annotations

import json
import math
import secrets
import socket
import threading
import webbrowser
from dataclasses import fields, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from typing import Mapping, Sequence, TYPE_CHECKING
from urllib.parse import urlparse

from .codec import NativeEventEncoder
from .errors import ResolutionError
from .ffi import StateInspectionRequest
from .graph import DriveInput, InputMode, ScalarInput, SpikeInput

if TYPE_CHECKING:
    from .simulation import CompiledNetwork, IncrementalSimulationRun, SimulationResult


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{context} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{context} must be finite") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{context} must be finite")
    return result


def _positive(value: object, context: str) -> float:
    result = _finite(value, context)
    if result <= 0.0:
        raise ResolutionError(f"{context} must be positive")
    return result


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ResolutionError(f"{context} must be an integer >= {minimum}")
    return value


def _encoder_parameters(encoder: object) -> dict[str, object]:
    if not is_dataclass(encoder):
        return {}
    result: dict[str, object] = {}
    for item in fields(encoder):
        if item.name == "kind":
            continue
        value = getattr(encoder, item.name)
        result[item.name] = value.value if hasattr(value, "value") else value
    return result


def _topology_layout(
    node_ids: Sequence[int], edges: Sequence[object]
) -> dict[int, tuple[float, float]]:
    """Produce a deterministic topology-aware seed layout in normalized space."""

    nodes = tuple(node_ids)
    if not nodes:
        return {}
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        if edge.pre in adjacency and edge.post in adjacency:
            adjacency[edge.pre].add(edge.post)
            adjacency[edge.post].add(edge.pre)

    unseen = set(nodes)
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        queue = [start]
        unseen.remove(start)
        component = []
        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda item: (-len(item), min(item)))

    columns = max(1, math.ceil(math.sqrt(len(components))))
    rows = math.ceil(len(components) / columns)
    positions: dict[int, tuple[float, float]] = {}
    for component_index, component in enumerate(components):
        cell_x = component_index % columns
        cell_y = component_index // columns
        width = 1.0 / columns
        height = 1.0 / rows
        if len(component) == 1:
            local = {component[0]: (0.5, 0.5)}
        else:
            root = min(
                component,
                key=lambda node: (-len(adjacency[node]), node),
            )
            layer = {root: 0}
            queue = [root]
            while queue:
                node = queue.pop(0)
                for neighbor in sorted(adjacency[node]):
                    if neighbor in component and neighbor not in layer:
                        layer[neighbor] = layer[node] + 1
                        queue.append(neighbor)
            for node in component:
                layer.setdefault(node, max(layer.values(), default=0) + 1)
            grouped: dict[int, list[int]] = {}
            for node, depth in layer.items():
                grouped.setdefault(depth, []).append(node)
            maximum = max(grouped)
            local = {}
            for depth, members in sorted(grouped.items()):
                members.sort()
                x = 0.5 if maximum == 0 else 0.12 + 0.76 * depth / maximum
                for index, node in enumerate(members):
                    y = (index + 1) / (len(members) + 1)
                    local[node] = (x, y)
        for node, (x, y) in local.items():
            positions[node] = (
                cell_x * width + (0.08 + 0.84 * x) * width,
                cell_y * height + (0.08 + 0.84 * y) * height,
            )
    return positions


def _role_aware_layout(
    node_ids: Sequence[int],
    edges: Sequence[object],
    *,
    input_nodes: frozenset[int],
    output_nodes: frozenset[int],
) -> dict[int, tuple[float, float]]:
    """Place input and output interfaces around a topology-derived core."""

    positions = _topology_layout(node_ids, edges)
    input_only = tuple(sorted(input_nodes - output_nodes))
    output_only = tuple(sorted(output_nodes - input_nodes))
    both = tuple(sorted(input_nodes & output_nodes))
    core = tuple(
        node
        for node in sorted(node_ids)
        if node not in input_nodes and node not in output_nodes
    )

    for index, node in enumerate(input_only):
        positions[node] = (0.07, (index + 1) / (len(input_only) + 1))
    for index, node in enumerate(output_only):
        positions[node] = (0.93, (index + 1) / (len(output_only) + 1))
    for index, node in enumerate(both):
        positions[node] = (0.50, (index + 1) / (len(both) + 1))
    for node in core:
        x, y = positions[node]
        positions[node] = (0.22 + 0.56 * x, 0.07 + 0.86 * y)
    return positions


class NetworkVisualizer:
    """Serve and control one live incremental run of a compiled network.

    The visualizer borrows the compiled network. Closing it ends only the live
    visualization run and local HTTP server. The compiled network remains reusable.
    """

    def __init__(
        self,
        compiled: "CompiledNetwork",
        duration: float,
        *,
        step: float = 1.0,
        sample_interval: float = 0.1,
        seed: int = 0,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        from .simulation import CompiledNetwork

        if not isinstance(compiled, CompiledNetwork):
            raise ResolutionError("NetworkVisualizer requires a CompiledNetwork")
        self.compiled = compiled
        self.duration = _finite(duration, "visualization duration")
        if self.duration <= 0.0:
            raise ResolutionError("visualization duration must be positive")
        self.step_size = _positive(step, "visualization step")
        self.sample_interval = _positive(
            sample_interval, "visualization sample interval"
        )
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**64:
            raise ResolutionError("visualization seed must be an unsigned 64-bit integer")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ResolutionError("the visualizer server must bind to a loopback host")
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ResolutionError("visualizer port must lie in [0, 65535]")
        self.host = host
        self.port = port
        self.seed = seed
        self._lock = threading.RLock()
        self._selected_node: int | None = None
        self._pending_spikes: list[SpikeInput] = []
        self._pending_drives: list[DriveInput] = []
        self._pending_scalars: list[ScalarInput] = []
        self._run: IncrementalSimulationRun = compiled.start_run(
            self.duration, seed=seed
        )
        self._token = secrets.token_urlsafe(24)
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def frontier(self) -> float:
        """Return the current simulation frontier."""

        return self._run.frontier

    @property
    def finished(self) -> bool:
        """Return whether the configured horizon has been reached."""

        return self._run.finished

    @property
    def url(self) -> str:
        """Return the authenticated local visualizer URL."""

        if self._server is None:
            raise RuntimeError("visualizer server has not been started")
        address, port = self._server.server_address[:2]
        host = "127.0.0.1" if address in {"0.0.0.0", "::"} else address
        return f"http://{host}:{port}/{self._token}/"

    def __enter__(self) -> "NetworkVisualizer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _known_nodes(self) -> frozenset[int]:
        return frozenset(node.id for node in self.compiled.network.graph.nodes)

    def select(self, node: int | None) -> dict[str, object]:
        """Select one node for live state and spike recording."""

        with self._lock:
            if node is not None and (
                not isinstance(node, int)
                or isinstance(node, bool)
                or node not in self._known_nodes()
            ):
                raise ResolutionError(f"unknown visualization node {node!r}")
            self._selected_node = node
            return {
                "selected": node,
                "state_names": []
                if node is None
                else list(self.compiled._model_names(node)),
            }

    def _port(self, identifier: object):
        if not isinstance(identifier, str):
            raise ResolutionError("input port must be a string")
        for port in self.compiled.network.graph.input_ports:
            if port.id == identifier:
                return port
        raise ResolutionError(f"unknown input port '{identifier}'")

    def inject(self, request: Mapping[str, object]) -> dict[str, object]:
        """Schedule a native spike, encoded presentation, or direct drive pulse."""

        if not isinstance(request, Mapping):
            raise ResolutionError("injection request must be an object")
        with self._lock:
            if self.finished:
                raise ResolutionError("the visualization run is already finished")
            port = self._port(request.get("port"))
            kind = request.get("kind")
            start = self.frontier + _finite(request.get("offset", 0.0), "input offset")
            if start < self.frontier or start > self.duration:
                raise ResolutionError("input time lies outside the remaining run")
            scheduled = 0
            if kind == "spike":
                if port.mode is not InputMode.SPIKE or not isinstance(
                    port.encoder, NativeEventEncoder
                ):
                    raise ResolutionError(
                        f"input port '{port.id}' does not accept native spikes"
                    )
                amplitude = _finite(request.get("amplitude", 1.0), "spike amplitude")
                count = _integer(request.get("count", 1), "spike count", minimum=1)
                if count > 4096:
                    raise ResolutionError("one injection is limited to 4096 spikes")
                interval = _finite(request.get("interval", 1.0), "spike interval")
                if count > 1 and interval <= 0.0:
                    raise ResolutionError("multi-spike interval must be positive")
                events = [
                    SpikeInput(start + index * interval, port.id, amplitude)
                    for index in range(count)
                    if start + index * interval <= self.duration
                ]
                self._pending_spikes.extend(events)
                self._pending_spikes.sort(key=lambda item: (item.t, item.port))
                scheduled = len(events)
            elif kind == "presentation":
                if isinstance(port.encoder, NativeEventEncoder):
                    raise ResolutionError(
                        f"input port '{port.id}' does not have a scalar encoder"
                    )
                value = _finite(request.get("value", 1.0), "presentation value")
                if not 0.0 <= value <= 1.0:
                    raise ResolutionError("presentation value must lie in [0, 1]")
                duration = _positive(
                    request.get("duration", self.step_size), "presentation duration"
                )
                end = min(self.duration, start + duration)
                if end <= start:
                    raise ResolutionError("presentation has no time remaining")
                if any(
                    item.port == port.id and item.t_start == start
                    for item in self._pending_scalars
                ):
                    raise ResolutionError(
                        f"input port '{port.id}' already has a presentation at this time"
                    )
                self._pending_scalars.append(ScalarInput(start, end, port.id, value))
                self._pending_scalars.sort(key=lambda item: (item.t_start, item.port))
                scheduled = 1
            elif kind == "drive":
                if port.mode is not InputMode.DRIVE or not isinstance(
                    port.encoder, NativeEventEncoder
                ):
                    raise ResolutionError(
                        f"input port '{port.id}' does not accept direct drive updates"
                    )
                value = _finite(request.get("value", 0.0), "drive value")
                baseline = _finite(request.get("baseline", 0.0), "drive baseline")
                duration = _finite(request.get("duration", 0.0), "drive duration")
                if duration < 0.0:
                    raise ResolutionError("drive duration must be nonnegative")
                events = [DriveInput(start, port.id, value)]
                if duration > 0.0 and start + duration <= self.duration:
                    events.append(DriveInput(start + duration, port.id, baseline))
                self._pending_drives.extend(events)
                self._pending_drives.sort(key=lambda item: (item.t, item.port))
                scheduled = len(events)
            else:
                raise ResolutionError(
                    "input kind must be 'spike', 'presentation', or 'drive'"
                )
            return {
                "scheduled": scheduled,
                "port": port.id,
                "frontier": self.frontier,
            }

    @staticmethod
    def _take_events(values: list, boundary: float, *, inclusive: bool, time) -> tuple:
        selected = []
        remaining = []
        for item in values:
            event_time = time(item)
            if event_time < boundary or (inclusive and event_time <= boundary):
                selected.append(item)
            else:
                remaining.append(item)
        values[:] = remaining
        return tuple(selected)

    def _inspection_requests(
        self, boundary: float, *, inclusive: bool
    ) -> tuple[StateInspectionRequest, ...]:
        node = self._selected_node
        if node is None:
            return ()
        values = []
        time = self.frontier
        tolerance = 8.0 * math.ulp(max(1.0, abs(boundary)))
        while time < boundary or (inclusive and time <= boundary + tolerance):
            if not inclusive and boundary - time <= tolerance:
                break
            selected_time = (
                boundary
                if inclusive and abs(time - boundary) <= tolerance
                else time
            )
            values.append(StateInspectionRequest(selected_time, node))
            time += self.sample_interval
        if inclusive and (not values or values[-1].t != boundary):
            values.append(StateInspectionRequest(boundary, node))
        return tuple(values)

    @staticmethod
    def _result_payload(
        result: "SimulationResult", *, frontier: float, finished: bool
    ) -> dict[str, object]:
        return {
            "frontier": frontier,
            "finished": finished,
            "spikes": [
                {"t": item.t, "node": item.node} for item in result.spikes
            ],
            "states": [
                {
                    "t": item.t,
                    "node": item.node,
                    "names": list(item.names),
                    "values": list(item.values),
                    "generation": item.generation,
                    "clamped": item.clamped,
                }
                for item in result.states
            ],
            "decoded": [
                {
                    "port": getattr(item, "port", None),
                    "window": getattr(item, "window", None),
                    "value": getattr(item, "value", None),
                    "valid": getattr(item, "valid", None),
                }
                for item in result.decoded
            ],
        }

    def advance(self, until: float | None = None) -> dict[str, object]:
        """Advance to the next frame or an explicit boundary."""

        with self._lock:
            if self.finished:
                return {
                    "frontier": self.frontier,
                    "finished": True,
                    "spikes": [],
                    "states": [],
                    "decoded": [],
                }
            boundary = min(
                self.duration,
                self.frontier + self.step_size
                if until is None
                else _finite(until, "visualization advance boundary"),
            )
            if boundary <= self.frontier:
                raise ResolutionError("advance boundary must exceed the frontier")
            final = boundary == self.duration
            spikes = self._take_events(
                self._pending_spikes,
                boundary,
                inclusive=final,
                time=lambda item: item.t,
            )
            drives = self._take_events(
                self._pending_drives,
                boundary,
                inclusive=final,
                time=lambda item: item.t,
            )
            scalars = self._take_events(
                self._pending_scalars,
                boundary,
                inclusive=final,
                time=lambda item: item.t_start,
            )
            inspections = self._inspection_requests(boundary, inclusive=final)
            if final:
                result = self._run.finish(
                    spike_inputs=spikes,
                    drive_inputs=drives,
                    scalar_inputs=scalars,
                    inspections=inspections,
                )
            else:
                result = self._run.advance(
                    boundary,
                    spike_inputs=spikes,
                    drive_inputs=drives,
                    scalar_inputs=scalars,
                    inspections=inspections,
                )
            return self._result_payload(
                result, frontier=self.frontier, finished=self.finished
            )

    def reset(self) -> dict[str, object]:
        """Restart network activity with the original seed and horizon."""

        with self._lock:
            if not self._run.closed:
                self._run.close()
            self._run = self.compiled.start_run(self.duration, seed=self.seed)
            self._pending_spikes.clear()
            self._pending_drives.clear()
            self._pending_scalars.clear()
            self._selected_node = None
            return {"frontier": 0.0, "finished": False}

    def graph_payload(self) -> dict[str, object]:
        """Return graph structure and current visual state for the client."""

        network = self.compiled.network
        graph = network.graph
        input_ports_by_node: dict[int, list[str]] = {}
        for port in graph.input_ports:
            input_ports_by_node.setdefault(port.node, []).append(port.id)
        output_ports_by_node: dict[int, list[str]] = {}
        for port in graph.output_ports:
            output_ports_by_node.setdefault(port.node, []).append(port.id)
        input_nodes = frozenset(input_ports_by_node)
        output_nodes = frozenset(output_ports_by_node)
        positions = _role_aware_layout(
            tuple(node.id for node in graph.nodes),
            graph.edges,
            input_nodes=input_nodes,
            output_nodes=output_nodes,
        )
        labels: dict[int, str] = {}
        for label, node in sorted(network.neuron_labels.items()):
            labels.setdefault(node, label)
        population_by_node: dict[int, str] = {}
        for population in network.population_records:
            for node in population.node_ids:
                population_by_node.setdefault(node, population.name)
        reservoir_by_node: dict[int, str] = {}
        for reservoir in network.reservoir_records:
            for node in reservoir.node_ids:
                reservoir_by_node.setdefault(node, reservoir.name)
        nodes = []
        for node in graph.nodes:
            x, y = positions[node.id]
            is_input = node.id in input_nodes
            is_output = node.id in output_nodes
            if is_input and is_output:
                layout_role = "input_output"
            elif is_input:
                layout_role = "input"
            elif is_output:
                layout_role = "output"
            elif node.id in reservoir_by_node:
                layout_role = "reservoir"
            else:
                layout_role = "hidden"
            nodes.append(
                {
                    "id": node.id,
                    "label": labels.get(node.id, str(node.id)),
                    "model": node.model,
                    "polarity": node.polarity.value,
                    "state_names": list(self.compiled._model_names(node.id)),
                    "population": population_by_node.get(node.id),
                    "reservoir": reservoir_by_node.get(node.id),
                    "is_input": is_input,
                    "is_output": is_output,
                    "input_ports": sorted(input_ports_by_node.get(node.id, ())),
                    "output_ports": sorted(output_ports_by_node.get(node.id, ())),
                    "layout_role": layout_role,
                    "x": x,
                    "y": y,
                }
            )
        node_by_id = {node.id: node for node in graph.nodes}
        edge_payload = [
            {
                "id": edge.id,
                "pre": edge.pre,
                "post": edge.post,
                "weight": edge.weight * node_by_id[edge.pre].polarity.sign,
                "magnitude": abs(edge.weight),
                "delay": edge.delay,
                "synapse": edge.synapse,
            }
            for edge in graph.edges
        ]
        inputs = [
            {
                "id": port.id,
                "node": port.node,
                "mode": port.mode.value,
                "parameter": port.parameter,
                "encoder": port.encoder.kind.name,
                "encoder_parameters": _encoder_parameters(port.encoder),
                "injection_kind": (
                    "presentation"
                    if not isinstance(port.encoder, NativeEventEncoder)
                    else "drive"
                    if port.mode is InputMode.DRIVE
                    else "spike"
                ),
            }
            for port in graph.input_ports
        ]
        outputs = [
            {
                "id": port.id,
                "node": port.node,
                "decoder": (
                    None if port.decoder is None else type(port.decoder).__name__
                ),
            }
            for port in graph.output_ports
        ]
        return {
            "name": network.name,
            "duration": self.duration,
            "step": self.step_size,
            "sample_interval": self.sample_interval,
            "semantic_sha256": network.semantic_sha256,
            "nodes": nodes,
            "edges": edge_payload,
            "inputs": inputs,
            "outputs": outputs,
            "frontier": self.frontier,
            "finished": self.finished,
        }

    def _handler(self):
        viewer = self
        base = f"/{self._token}"

        class Handler(BaseHTTPRequestHandler):
            server_version = "LacunaVisualizer/1"

            def log_message(self, format, *args) -> None:
                return

            def _send(self, status: int, content_type: str, payload: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; script-src 'unsafe-inline'; "
                    "style-src 'unsafe-inline'; connect-src 'self'; "
                    "img-src data: blob:; media-src blob:",
                )
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.end_headers()
                self.wfile.write(payload)

            def _json(self, status: int, payload: object) -> None:
                self._send(
                    status,
                    "application/json; charset=utf-8",
                    json.dumps(payload, allow_nan=False).encode("utf-8"),
                )

            def _request(self) -> object:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ResolutionError("invalid request length") from exc
                if length < 0 or length > 1_048_576:
                    raise ResolutionError("request body is too large")
                if length == 0:
                    return {}
                try:
                    return json.loads(self.rfile.read(length))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ResolutionError("request body must be JSON") from exc

            def do_GET(self) -> None:
                path = urlparse(self.path).path.rstrip("/")
                if path == base:
                    template = files("lacuna").joinpath("_visualizer.html").read_text(
                        encoding="utf-8"
                    )
                    html = template.replace(
                        "__LACUNA_BASE_PATH__", json.dumps(base)
                    ).encode("utf-8")
                    self._send(200, "text/html; charset=utf-8", html)
                    return
                if path == f"{base}/api/graph":
                    self._json(200, viewer.graph_payload())
                    return
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                path = urlparse(self.path).path.rstrip("/")
                try:
                    request = self._request()
                    if path == f"{base}/api/step":
                        if not isinstance(request, Mapping):
                            raise ResolutionError("step request must be an object")
                        self._json(200, viewer.advance(request.get("until")))
                    elif path == f"{base}/api/select":
                        if not isinstance(request, Mapping):
                            raise ResolutionError("selection request must be an object")
                        self._json(200, viewer.select(request.get("node")))
                    elif path == f"{base}/api/inject":
                        if not isinstance(request, Mapping):
                            raise ResolutionError("injection request must be an object")
                        self._json(200, viewer.inject(request))
                    elif path == f"{base}/api/reset":
                        self._json(200, viewer.reset())
                    elif path == f"{base}/api/close":
                        self._json(200, {"closed": True})
                        # ThreadingHTTPServer runs this handler outside the
                        # serve_forever thread, so shutdown is safe here.  Do it
                        # synchronously: once the client receives this response,
                        # the blocking viewer.wait() is guaranteed to be ending.
                        viewer.close()
                    else:
                        self._json(404, {"error": "not found"})
                except (ResolutionError, ValueError, RuntimeError) as exc:
                    self._json(400, {"error": str(exc)})
                except Exception as exc:
                    self._json(500, {"error": f"visualizer failure: {exc}"})

        return Handler

    def start(self, *, open_browser: bool = True) -> "NetworkVisualizer":
        """Start the local server and optionally open a browser."""

        with self._lock:
            if self._closed:
                raise RuntimeError("visualizer is closed")
            if self._server is not None:
                return self
            server_type = ThreadingHTTPServer
            if self.host == "::1":
                class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
                    address_family = socket.AF_INET6

                server_type = IPv6ThreadingHTTPServer
            self._server = server_type((self.host, self.port), self._handler())
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="lacuna-visualizer",
                daemon=True,
            )
            self._thread.start()
        if open_browser:
            webbrowser.open(self.url)
        return self

    def wait(self) -> None:
        """Block until the visualizer session closes."""

        if self._thread is None:
            raise RuntimeError("visualizer server has not been started")
        try:
            while self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self.close()

    def close(self) -> None:
        """Stop the server and release the simulation run."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            server = self._server
            self._server = None
            if not self._run.closed:
                self._run.close()
        if server is not None:
            server.shutdown()
            server.server_close()


__all__ = ["NetworkVisualizer"]
