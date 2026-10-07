# Importing trained LIF networks

Lacuna supports deploying compatible externally trained networks through
`lacuna.importers`. Training stays in the source framework. Conversion produces
an ordinary Lacuna network of LIF neurons and static delta connections, using
the existing C scheduler and graph storage formats.

The public interface has two levels. Framework-neutral layer descriptions
accept weights and continuous-time neuron parameters from any training tool
that implements the same dynamics. An optional official Lava-DL SLAYER adapter
extracts those descriptions from a restricted source model and compares the
source and converted networks on representative inputs.

This feature has the same experimental status as the rest of Lacuna. Support
applies to the model contract below, not arbitrary SNNs or checkpoint files.

## Supported models

| Interface | Supported configuration |
| --- | --- |
| `build_dense_lif` | Feedforward dense LIF layers with signed static weights |
| `build_feedforward_lif` | Dense and spatial LIF layers with grouped or depthwise convolution |
| `import_slayer_dense` | Compatible official CUBA Dense blocks |
| `import_slayer_feedforward` | Compatible official CUBA Conv, Pool, Flatten, and Dense blocks |

The target neurons have zero rest, reset, drive, and refractory duration,
positive membrane time constants and thresholds, and delta-input synapses.
Inputs are binary spike events. Each input emits at most once per timestamp.
Signed weights use explicit mixed-sign neurons without changing Lacuna's
default Dale typing for other networks.

Convolutional kernels expand into connections between individual neurons.
Their coefficients are shared during source training, but the converted graph
stores static edge weights. Memory therefore scales with expanded connectivity,
not only with the number of learned coefficients. Pooling uses this same
representation with a separate fixed kernel for each channel.

The SLAYER adapter supports CPU float32 inference in an official
`torch.nn.Sequential` model. Set `current_decay=1.0`, use fixed shared neuron
parameters, and explicitly disable weight hooks, weight normalization, delays,
and output delay shifts. Spatial pooling requires fixed weights, nonoverlapping
kernel-sized strides, zero padding, unit dilation, and evenly divisible input
dimensions. Its output is a spike train after neuronal integration, not a
numeric average or maximum. Recurrence, skip connections, persistent state,
trained neuron parameters, tensor max/average pooling, and sampled SRM models
are outside this adapter's supported profile.

See the [complete SLAYER contract](official_slayer_deployment.md#model-contract)
before designing a model for conversion. Unsupported configurations are
rejected rather than silently substituted.

## Framework-neutral deployment

The standard Lacuna installation includes the importers. The following example
requires no Torch or SLAYER installation. Here the example weights represent
coefficients obtained from an external training tool, in `[output, input]`
order:

```python
from lacuna import Engine, SpikeTrain
from lacuna.importers import DenseLIFLayer, build_dense_lif

converted = build_dense_lif(
    [DenseLIFLayer([[1.25, -0.5]], tau_m=20.0, threshold=1.0)]
)
converted.network.save("trained-network.json")

with Engine().compile(converted.network) as simulation:
    simulation.save_compiled_graph_image("trained-network.lcbin")
    result = simulation.run(
        3.0,
        inputs={
            converted.input_ports[0]: SpikeTrain((1.0, 2.0)),
            converted.input_ports[1]: SpikeTrain((2.0,)),
        },
    )
```

`Conv2dLIFLayer` describes spatial cross-correlation weights in
`[output, input_within_group, kernel_row, kernel_column]` order.
Pass these layers to `build_feedforward_lif` with an explicit `(C, H, W)`
input shape. A following `DenseLIFLayer` uses channel, row, column flattening
order. Deployment objects expose `input_ports` and `layer_nodes` so that callers
can map inputs and inspect spikes without guessing neuron identifiers.

## Converting a trained SLAYER model

Use the [isolated, pinned source environment](official_slayer_deployment.md#reproduce-the-first-training-and-deployment-check)
on the training or conversion host. Ordinary Lacuna installation does not
install Torch, Lava-DL, or the Lava multiprocessing runtime. The validated
upstream revision is `d825ac506eb9bb91ea2b945116fb45a40e775451`, using
Python 3.10 and Torch 2.9.0. Other upstream versions are not covered by these
validation results.

First recreate the compatible architecture and load its trained weights using
the source framework. The importer accepts this live model, not a standalone
checkpoint whose architecture would have to be inferred. After training,
convert and validate it as follows. This example assumes a spatial input of
one 28 by 28 channel:

```python
from lacuna import Engine
from lacuna.importers import (
    import_slayer_feedforward,
    validate_slayer_feedforward,
)

model.cpu().float().eval()
converted = import_slayer_feedforward(
    model,
    input_shape=(1, 28, 28),
    timestep=1.0,
    acknowledge_quantization=True,
)
report = validate_slayer_feedforward(
    model,
    converted,
    validation_spikes,
    source_batch_size=64,
)
print(report["exact_spike_match_on_batch"])
print(report["prediction_agreement"])

# Save after reviewing whether the measured disagreement is acceptable.
converted.deployment.network.save("trained-network.json")
with Engine().compile(converted.deployment.network) as simulation:
    simulation.save_compiled_graph_image("trained-network.lcbin")
```

`validation_spikes` must be a CPU binary tensor with shape
`[batch, channel, row, column, time]`. For dense-only models, use
`import_slayer_dense` and `validate_slayer_dense` with inputs shaped
`[batch, input, time]`. Use the same source batch size as source evaluation for
convolutional comparisons. Input bin `k` maps to time `k * timestep`, starting
at zero, with no additional input encoding in the importer.

Conversion snapshots weights and parameters. Later source training does not
update the exported network. Reconvert after training changes, and keep the
source architecture, checkpoint, import metadata, representative inputs, and
validation report alongside the network artifacts. The existing experiment
scripts demonstrate this provenance workflow without adding a training engine
to Lacuna.

## Numerical agreement and deployment

The source performs scaled integer-state truncation and float32 accumulation.
Lacuna propagates binary64 state analytically between events. Matching the
decay equations therefore does not guarantee identical spike trains, especially
near threshold. The required quantization acknowledgment makes this limitation
explicit. Validation reports every neural layer's spike differences, including
pooling neurons, as well as final predictions. Applications must choose their
own acceptance criterion. There is no universal acceptable error threshold.

The JSON file stores an editable network that can be recompiled on the host.
The binary image stores the compiled deployment representation and can be
loaded by the existing C runtime without Python, Torch, SLAYER, or the symbolic
compiler on the target. See [compiled graph images](compiled_graph_images.md)
for loader compatibility and resource requirements. Imported layers do not
require a special C executor.

Validation evidence includes [dense MNIST](official_slayer_mnist.md),
[convolutional MNIST](official_slayer_conv_mnist.md),
[convolutional and pooling MNIST](official_slayer_pool_mnist.md), and the
[numerical and native deployment checks](official_slayer_conversion_validation.md).
The [compact spiking AlexNet example](official_slayer_alexnet_mnist.md) extends
the training-and-conversion workflow to five convolutional and three dense layers.
These reports preserve observed disagreements and delimit what each test
establishes. They do not claim general bitwise equivalence or full-dataset
accuracy from a subset pilot.
