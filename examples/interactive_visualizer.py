#!/usr/bin/env python3
"""Launch a conventional input-reservoir-readout network in the visualizer."""

from __future__ import annotations

import os
from pathlib import Path

from lacuna import (
    AdaptiveLIF,
    BurstEncoder,
    Engine,
    FixedInDegree,
    FixedOutDegree,
    NativeEventEncoder,
    NetworkBuilder,
    NeuronPolarity,
    PoissonRateEncoder,
    TTFSEncoder,
    Uniform,
)


def find_core_library() -> Path:
    """Find the just-built Lacuna shared library on macOS, Linux, or Windows."""

    configured = os.environ.get("LACUNA_CORE_LIB")
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        raise SystemExit(f"LACUNA_CORE_LIB does not point to a file: {path}")

    candidates = tuple(sorted(Path("build").glob("liblacuna_core.*"))) + tuple(
        sorted(Path("build").glob("lacuna_core.dll"))
    )
    if not candidates:
        raise SystemExit(
            "The C core has not been built. Run:\n"
            "  cmake -S . -B build\n"
            "  cmake --build build"
        )
    return candidates[0]


def build_demo_network():
    builder = NetworkBuilder(
        "input-reservoir-readout-demo",
        metadata={"purpose": "visualizer demonstration"},
    )
    input_neurons = builder.population(
        "input",
        5,
        AdaptiveLIF(
            name="input_adaptive_lif",
            drive=0.0,
            adaptation_increment=0.8,
            refractory=1.0,
        ),
        polarity=NeuronPolarity.EXCITATORY,
    )
    reservoir = builder.reservoir(
        "reservoir",
        40,
        AdaptiveLIF(
            name="reservoir_adaptive_lif",
            drive=13.0,
            adaptation_increment=0.8,
        ),
        connectivity=FixedOutDegree(5, seed=20, exclude_self=True),
        weight=Uniform(2.5, 5.0, seed=21),
        delay=Uniform(1.0, 6.0, seed=22),
        seed=20,
        inhibitory_fraction=0.2,
    )
    readout = builder.population(
        "readout",
        3,
        AdaptiveLIF(
            name="readout_adaptive_lif",
            drive=11.0,
            adaptation_increment=0.5,
        ),
        polarity=NeuronPolarity.EXCITATORY,
    )

    builder.connect(
        input_neurons,
        reservoir,
        pattern=FixedOutDegree(14, seed=30),
        weight=Uniform(18.0, 22.0, seed=31),
        delay=Uniform(0.8, 2.5, seed=32),
    )
    builder.connect(
        reservoir,
        readout,
        pattern=FixedInDegree(14, seed=40),
        weight=Uniform(4.0, 7.0, seed=41),
        delay=Uniform(1.0, 4.0, seed=42),
    )

    # Each input neuron owns its encoder; activity then enters the recurrent core
    # through an ordinary feed-forward projection.
    builder.input("native-kick", input_neurons[0], encoder=NativeEventEncoder())
    builder.input(
        "poisson-rate",
        input_neurons[1],
        encoder=PoissonRateEncoder(0.02, 0.20, amplitude=20.0),
    )
    builder.input(
        "ttfs",
        input_neurons[2],
        encoder=TTFSEncoder(1.0, 20.0, amplitude=20.0),
    )
    builder.input(
        "burst",
        input_neurons[3],
        encoder=BurstEncoder(0.05, 0.50, duration=20.0, amplitude=20.0),
    )
    builder.input("direct-current", input_neurons[4], parameter="drive")
    builder.outputs("readout-spike", readout)
    return builder.build()


def main() -> None:
    network = build_demo_network()
    report = network.validate()
    library = find_core_library()

    print(
        f"Opening {network.name}: {report.node_count} neurons, "
        f"{report.edge_count} synapses"
    )
    print("Try these controls:")
    print("  1. Click and drag a neuron, then click Step.")
    print("  2. Select native-kick and inject amplitude 20.")
    print("  3. Select poisson-rate, ttfs, burst, or direct-current.")
    print("  4. For a clear adaptation test, select input[4], then inject")
    print("     direct-current at value 25 for duration 250.")
    print("  5. Compare the rising w trace with the falling inter-spike rate.")
    print("  6. Use Save screenshot or Start recording under Capture.")
    print("Close the visualizer with its Close session button or press Ctrl-C here.")

    with Engine(library).compile(network) as compiled:
        with compiled.visualizer(
            500.0,
            step=2.0,
            sample_interval=0.2,
        ) as viewer:
            viewer.start(open_browser=True)
            print(f"Visualizer URL: {viewer.url}")
            viewer.wait()


if __name__ == "__main__":
    main()
