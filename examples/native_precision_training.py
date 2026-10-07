"""Train in native precision and export the learned network for that profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from lacuna import Engine, LIF, NetworkBuilder, PairSTDP, SpikeTrain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("float64", "float32", "float16"), default="float32")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/native-precision"))
    args = parser.parse_args(argv)
    builder = NetworkBuilder("native-precision-training")
    pre = builder.neuron("pre", LIF(name="pre_lif"))
    post = builder.neuron("post", LIF(name="post_lif"))
    builder.connect(pre, post, weight=0.5, delay=0.5, plasticity=PairSTDP(learning_rate=0.01))
    pre_input = builder.input("pre_stimulus", pre)
    post_input = builder.input("post_stimulus", post)
    builder.output("response", post)
    network = builder.build()
    engine = Engine(precision=args.precision)
    with engine.compile(network) as simulation:
        result = simulation.run(30.0, inputs={
            pre_input: SpikeTrain(times=(1.0, 11.0, 21.0), values=20.0),
            post_input: SpikeTrain(times=(3.0, 13.0, 23.0), values=20.0),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / args.precision
    learned = result.save_learned_network(stem.with_suffix(".json"))
    # Graph images contain initial weights, so export the learned snapshot.
    with engine.compile(learned) as simulation:
        simulation.save_compiled_graph_image(stem.with_suffix(".lcg"))
    print(f"Precision: {result.precision.value}")
    print(f"Weight: {network.graph.edges[0].weight} -> {result.weights[0]}")
    print(f"Spikes: {len(result.spikes)}")
    print(f"Learned network: {stem.with_suffix('.json')}")
    print(f"Deployment image: {stem.with_suffix('.lcg')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
