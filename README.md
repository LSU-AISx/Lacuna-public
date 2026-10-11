# Lacuna

Lacuna is a spiking neural network engine that runs supported neuron dynamics
analytically between events and uses numerical stepping when an analytical
solution is unavailable. A single C runtime supports both modes, including
mixed networks, synaptic learning, recording, and deployment without Python.

## Get started

Lacuna requires Python 3.10 or newer, a C compiler, and CMake. From a checkout:

```sh
python3 -m pip install -e .
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

The full Python test suite also needs the reduced-precision native builds;
see [numerical precision](docs/numerical_precision.md).

Create and run a small network with the Python API:

```python
from lacuna import Engine, LIF, NetworkBuilder, SpikeTrain

builder = NetworkBuilder("example")
source = builder.neuron("source", LIF())
target = builder.neuron("target", LIF())
builder.connect(source, target, weight=2.0, delay=1.0)
stimulus = builder.input("stimulus", source)
builder.outputs("readout", target)

with Engine().compile(builder.build()) as simulation:
    result = simulation.run(
        100.0,
        inputs={stimulus: SpikeTrain((1.0, 2.0, 3.0), 20.0)},
    )

print(result.spikes)
```

`Engine()` uses native float64 precision by default. Select
`Engine(precision="float32")` or `Engine(precision="float16")` before
compilation to use a reduced-precision native runtime; see
[numerical precision](docs/numerical_precision.md) for build and deployment
requirements.

## What is supported

- Event-driven analytical propagation for supported integrate-and-fire,
  adaptive, filtered-synapse, and escape-noise dynamics; stepped execution for
  supported nonlinear ODE models.
- Sparse recurrent networks, delayed connections, external spike and drive
  inputs, and deterministic same-time event handling.
- Static synapses, pair and triplet STDP, and reward-modulated STDP.
- Spike, state, and causal-trace recording; versioned graph and compiled-graph
  artifacts for standalone C deployment.
- Import of trained dense and convolutional LIF networks, including fixed
  spiking pooling, into the ordinary runtime.

The precise model boundaries and current limitations are in the
[design specification](lacuna_design_spec.md).

## Guides and examples

- [Model authoring and numerical precision](docs/numerical_precision.md)
- [Bounded numerical execution and trajectory reuse](docs/bounded_numerical_execution.md)
- [Importing trained networks](docs/imported_networks.md)
- [Standalone compiled-graph images](docs/compiled_graph_images.md)
- [MNIST reward-modulated STDP example](examples/mnist_modulated_stdp.py)
- [Official SLAYER deployment adapter](docs/official_slayer_deployment.md) and
  [AlexNet MNIST paper example](docs/official_slayer_alexnet_mnist.md)

The R-STDP example uses a signed, class-specific feedback signal after the
decision window; it is an experiment in the learning API, not a claim of
state-of-the-art classification performance.

Lacuna is distributed under the [BSD 3-Clause License](LICENSE).

Generative AI was used to assist in writing this README and other documentation,
however, the developers have reviewed the content for accuracy.
