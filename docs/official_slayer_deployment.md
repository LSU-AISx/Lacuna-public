# Official SLAYER training and ordinary Lacuna deployment

For the public import API and a deployment-oriented quick start, see
[importing trained networks](imported_networks.md). This guide gives the
SLAYER-specific model contract and reproducible source setup.

Training runs in the unmodified Lava-DL implementation of SLAYER. Lacuna imports
the resulting static weights and neuron parameters. There is no SLAYER trainer,
surrogate-gradient implementation, sampled SRM executor, or new C execution path
inside Lacuna.

The supported source is a CPU float32 `torch.nn.Sequential` containing
compatible official CUBA `Dense`, `Conv`, `Pool`, and `Flatten` blocks. The original
dense-only interface remains available. This is a restricted model importer,
not a converter for arbitrary checkpoints or the original sampled-SRM examples
from `slayerPytorch`.

## Model contract

Use binary inputs with shape `[batch, channel, time]` for dense models or
`[batch, channel, row, column, time]` for spatial models, zero initial states, and
at most one spike per input neuron per bin. Every neuron has zero rest, reset,
drive, and refractory clamp. Weights may have either sign and remain static
during Lacuna execution. Both hidden and output weights can be trained by
SLAYER before export.

Set `current_decay=1.0` to remove the persistent synaptic current. This gives a
delta-input LIF neuron. Set a fixed shared `voltage_decay` strictly between zero
and one. The importer reads its actual quantized value `d/4096`, then assigns

\[
\tau_m=-\Delta t/\log(1-d/4096).
\]

The target uses the effective float32 source threshold. In real arithmetic,
these parameters give the same membrane decay at the input timestamps. With
zero drive and positive threshold, no additional spikes can occur between
delta deposits. Lacuna can therefore use its existing analytical event-driven
LIF execution without introducing a simulation grid.

Explicitly set `pre_hook_fx=None`, `weight_norm=False`, `delay=False`, and
`delay_shift=False` on every source block. The last setting matters because
the official default shifts block outputs by one bin. This first importer
rejects shifted outputs and learned delays instead of guessing their mapping.
Normalization, dropout, persistent or nonzero initial state, trained neuron
parameters, masks, graded spikes, bias, recurrent/skip connections, and other
layer families are also rejected. Conservative bounds reject parameters that
could overflow the source integer-state calculation.

## Numerical compatibility is measured

Official CUBA neurons use scaled integer state calculations even on CPU. They
round weighted inputs and decay toward zero at every bin. Lacuna does not
reproduce that rounding. It analytically propagates binary64 state between
events. Increasing the official neuron's `scale` reduces the state quantum
without modifying its implementation, but does not prove exact equivalence.

Export therefore requires `acknowledge_quantization=True`.
`validate_slayer_dense()` and `validate_slayer_feedforward()` execute the actual source and target networks on the
same supplied inputs. It checks spike identity and bin time at every layer,
not only output classification, and reports mismatches and off-grid spikes.
This is validation of those inputs, not a guarantee for every possible input.
The tests deliberately include a near-threshold mismatch to ensure it remains
visible. Final-output count ties use the lowest index, including silent outputs.

The exporter rejects a changed source model during subsequent validation using
a fingerprint of weights, parameters, source files, and the Torch version. The
deployment is an independent snapshot, so later source training cannot modify
it. Re-export and revalidate after retraining.

## Separation from the core

`lacuna.importers.DenseLIFLayer` and `build_dense_lif()` form a framework-neutral
construction interface using weights in `[output, input]` order and continuous
LIF parameters. Other training tools can populate this interface without
depending on SLAYER. `Conv2dLIFLayer` and `build_feedforward_lif()` extend this
interface to spatial kernels with explicit stride, padding, dilation, and groups.
The optional source adapter lives in
`lacuna.importers.slayer`. Importing ordinary Lacuna does not import Torch or
Lava-DL, and neither dependency is needed on the deployment target.

The constructed network consists of ordinary LIF neurons and signed static
delta edges. Input relay neurons turn unit-valued named input events into
outgoing spikes. JSON saving and binary-image export use the existing APIs.
No external weight-update API or special runtime has been added.

## Convolutional deployment

`import_slayer_feedforward(model, input_shape=(C, H, W),
acknowledge_quantization=True)` imports a compatible convolutional network.
Provide the spatial input shape explicitly. The source must include an
official CUBA `Flatten` block between its last spatial layer and first dense
block. This preserves channel-major, row-major ordering without introducing
additional neurons or computation. The importer checks shapes rather than
running a dummy input through the source during export.

Kernels use cross-correlation, as in the source implementation, with no kernel
reversal. Supported geometry includes rectangular kernels, strides, dilation,
zero padding, and grouped or depthwise convolution. Temporal convolution,
transposed convolution, max pooling, recurrence, and skip connections remain
outside this importer profile. The neuron and numerical restrictions above
apply to every convolutional layer as well as every dense layer.

SLAYER shares and trains the kernel coefficients across spatial positions.
Export copies those coefficients onto the corresponding static Lacuna edges,
omitting zero weights and connections into padding. It does not create a dense
expanded weight matrix. Runtime storage still scales with the expanded
connections, not just the number of unique kernel coefficients. This is
inference deployment, not ongoing tied-weight training inside Lacuna.

`validate_slayer_feedforward()` checks every convolutional, pooling, and dense spike
layer, flattening spatial coordinates only for reporting. An official
`Flatten` block does not add a second copy of its input spike stream. Source
inference is batched to bound convolution workspace, and each test sample
starts from an independent initial state in both engines.
The `source_batch_size` option records and controls that source batch size.
Use the same batch geometry as the source evaluation when checking a trained
model, since float32 convolution implementations need not be bitwise invariant
to changing batch size.

## Spiking pooling deployment

The feedforward importer also accepts official CUBA `Pool` blocks with fixed
kernel weights. Each feature channel is pooled independently. Supported kernels
are square or rectangular, with stride equal to kernel size, zero padding,
unit dilation, and spatial dimensions divisible by the kernel dimensions.
Overlapping windows, uneven spatial dimensions, padding, dilation, and trainable
pool kernels are rejected. The usual neuron, delay, and quantization restrictions
still apply. This bounded geometry avoids upstream pooling's special padding
and channel-folding behavior.

For example, this pools each nonoverlapping 2×2 region using quarter-weight
synapses into a CUBA neuron:

```python
pool = slayer.block.cuba.Pool(
    neuron_params,
    kernel_size=2,
    stride=2,
    weight_scale=0.25,
    pre_hook_fx=None,
    weight_norm=False,
    delay=False,
    delay_shift=False,
)
```

Here `neuron_params` follows the same delta-LIF contract as the other layers.
The combined synaptic input is an average over the four positions, but the
block output remains the spike train of a neuron with its own membrane state,
threshold, and reset. It is neither ordinary numeric average pooling nor
maximum selection. `weight_scale=1.0` instead gives unit-weight summed input.
The importer copies the actual fixed kernel values, not an assumed default.

Pooling lowers through the existing framework-neutral `Conv2dLIFLayer` as one
independent kernel per channel. It therefore becomes ordinary LIF neurons and
static delta edges, with no C engine change. Metadata identifies the source
layer as `pool2d` and the lowered connectivity as `conv2d`. Pooling neurons have
their own entries in all-layer spike validation and are retained in JSON and
binary deployment images. Pooling weights remain fixed during source training,
while gradients pass through the official block to the surrounding learned
layers.

The [pooling MNIST validation](official_slayer_pool_mnist.md) provides an example
with two convolutional layers, two fixed spiking pools, and a learned readout.

## Reproduce the first training and deployment check

Run from the Lacuna checkout with Python 3.10. Keep this environment separate
from the normal Lacuna environment. The official repository is archived, so
the tested source revision is pinned.

```bash
python3.10 -m venv artifacts/official-slayer-env
source artifacts/official-slayer-env/bin/activate
python -m pip install -r integrations/official_slayer/requirements.txt

git clone https://github.com/lava-nc/lava-dl.git artifacts/lava-dl-official
git -C artifacts/lava-dl-official checkout d825ac506eb9bb91ea2b945116fb45a40e775451

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
export PYTHONPATH="$PWD/src:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"
# On Linux, select the shared library explicitly.
# export LACUNA_CORE_LIB="$PWD/build/liblacuna_core.so"

python -m pytest tests/python/test_dense_lif_import.py tests/python/test_official_slayer_import.py
python examples/train_official_slayer_lif.py --epochs 200 --seed 0
```

Skip cloning if that checkout already exists. Confirm its revision and leave
the official sources unmodified. Source imports are intentional here: they
avoid installing the unrelated Lava multiprocessing runtime. The example does
not replace or monkeypatch any upstream neuron or gradient routine.

The example trains a 4→16→2 network on noisy dual-rail XOR using official
`SpikeRate` loss and PyTorch Adam at learning rate 0.01. All weights in both
layers are optimized. The inputs are 48-bin independent spike realizations:
128 training examples, 64 validation examples, and 128 held-out test examples.
Checkpoint selection uses validation accuracy, with validation loss as a
tiebreaker, checked at epoch 1 and every 10 epochs. The test set does not select
the checkpoint.

Each run creates a fresh `artifacts/official-slayer-lif/<timestamp>/` directory
containing `report.json`, `official_checkpoint.pt`, `validation_inputs.pt`,
`network.json`, and `network.lcbin`. The report stores all-layer comparisons,
training history, layer weight changes, actual source provenance, and artifact
hashes. This small experiment tests the workflow, not MNIST accuracy.

A [WDBC pilot](official_slayer_wdbc.md) applies the same workflow to real-valued
features with training-only scaling and a held-out patient split. It records
both classification results and the observed spike-level transfer differences.

The [MNIST subset pilot](official_slayer_mnist.md) trains both layers of a
784→64→10 dense LIF network and compares all-layer spikes and classifications
on 1,000 validation and 1,000 test images. The replay helper sizes its event
and input buffers from the graph to support the larger fan-out.

The [convolutional MNIST pilot](official_slayer_conv_mnist.md) trains two spatial
layers and a dense output layer, then deploys them through the same C engine.

The [extended conversion validation](official_slayer_conversion_validation.md)
adds three fresh training seeds per architecture, a 48-case numerical stress
matrix, and standalone C replay of both trained binary images. It records
the observed disagreements as well as the successful deployment checks.

## Minimal use after training

```python
from lacuna import Engine
from lacuna.importers import import_slayer_dense, validate_slayer_dense

model.eval()
converted = import_slayer_dense(
    model, timestep=1.0, acknowledge_quantization=True
)
report = validate_slayer_dense(model, converted, validation_spikes)
print(report["exact_spike_match_on_batch"])

converted.deployment.network.save("trained-network.json")
with Engine().compile(converted.deployment.network) as simulation:
    simulation.save_compiled_graph_image("trained-network.lcbin")
```

Review the comparison before deployment. A matching classification score alone
does not establish matching spike trains. The binary image loads through the
ordinary C graph loader and requires no training framework on the target.

## Upstream sources

- [Official Lava-DL SLAYER repository](https://github.com/lava-nc/lava-dl/tree/d825ac506eb9bb91ea2b945116fb45a40e775451)
- [CUBA neuron](https://github.com/lava-nc/lava-dl/blob/d825ac506eb9bb91ea2b945116fb45a40e775451/src/lava/lib/dl/slayer/neuron/cuba.py)
- [Integer leaky dynamics](https://github.com/lava-nc/lava-dl/blob/d825ac506eb9bb91ea2b945116fb45a40e775451/src/lava/lib/dl/slayer/neuron/dynamics/leaky_integrator.py)
- [Dense block and output delay](https://github.com/lava-nc/lava-dl/blob/d825ac506eb9bb91ea2b945116fb45a40e775451/src/lava/lib/dl/slayer/block/base.py)
