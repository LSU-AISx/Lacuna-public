# Spiking AlexNet-style MNIST example

This example trains a compact AlexNet-style SNN with official Lava-DL SLAYER,
then imports the selected checkpoint into ordinary Lacuna LIF neurons and
delta synapses. It keeps the five-convolution, three-dense-layer organization
and pooling positions of
[AlexNet](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html),
but it is an MNIST adaptation, not the original ImageNet architecture or a
reproduction of its reported accuracy.

The original large RGB input, channel widths, 4,096-unit dense layers, ReLU,
local response normalization, max pooling, dropout, and softmax are not used.
The adapted forward network contains only compatible spiking LIF blocks and
static delta connections. Fixed spiking sum pooling replaces max pooling.
Classification uses the spike-count readout of ten output neurons.

## Architecture

MNIST pixels retain their original resolution. The input encoder adds a silent
two-pixel border, allowing three nonoverlapping pooling stages without uneven
dimensions. This padding is input preparation, not a hidden network operation.

| Block | Configuration | Output shape |
| --- | --- | --- |
| Input | Binary spikes, padded 28×28 image | 1×32×32 |
| Conv 1 | 8 kernels, 5×5, padding 2 | 8×32×32 |
| Pool 1 | Fixed unit-weight 2×2, stride 2 | 8×16×16 |
| Conv 2 | 16 kernels, 5×5, padding 2 | 16×16×16 |
| Pool 2 | Fixed unit-weight 2×2, stride 2 | 16×8×8 |
| Conv 3 | 24 kernels, 3×3, padding 1 | 24×8×8 |
| Conv 4 | 24 kernels, 3×3, padding 1 | 24×8×8 |
| Conv 5 | 16 kernels, 3×3, padding 1 | 16×8×8 |
| Pool 3 | Fixed unit-weight 2×2, stride 2 | 16×4×4 |
| Flatten | Channel, row, column order | 256 |
| Dense 1 | Spiking LIF | 64 |
| Dense 2 | Spiking LIF | 32 |
| Readout | Spiking LIF | 10 |

All eight Conv/Dense weight tensors are trained. The three four-entry pooling
kernels remain fixed. There are eleven neural stages because each pool also
contains LIF neurons. Flattening changes indexing without adding a neural stage.
The configuration has 34,248 learned weights and expands to 20,842 neurons,
including input relays, and at most 1,573,216 edges. Zero weights are omitted
during import. Kernel sharing reduces source parameters, not the expanded
Lacuna edge storage.

## Training and validation

The default run uses 10,000 training images, 1,000 validation images, and 1,000
images from the separate canonical test partition. A seeded random permutation
selects disjoint training and validation images without forcing equal counts
per class. The same procedure supports the full 55,000/5,000/10,000 split.
The script does not download MNIST implicitly, and verifies the cached files
through the existing loader.

The input encoder produces 32 binary time bins with spike probability
`0.4 * pixel / 255` per bin. Every neuron uses `current_decay=1`,
`voltage_decay=512/4096`, threshold 1, and constructor state scale 4096.
Initial states are zero. Bias, normalization, weight hooks, persistent states,
learned delays, and output delay shifts are disabled.

Training uses official `SpikeRate(true_rate=0.2, false_rate=0.02)` loss, Adam
at learning rate 0.001, batch size 16, ten epochs, and one CPU thread. Model
seed 0 and split seed 137 are defaults. Source training, validation, and
conversion remain in CPU float32. The target runs the ordinary C evaluator.

A bounded development preflight found that default surrogate parameters left
the first four convolutional tensors unchanged after an optimizer step in
this deeper network. The example instead uses the public SLAYER parameters
`tau_grad=0.01` and `scale_grad=10`. This changes the training gradient, not the
neuron's forward equations. Initialization remains the official implementation
with `weight_scale=2`. The preflight verifies finite, nonzero gradients and
measurable updates in all eight trained tensors, with unchanged pooling
weights. This check is not evidence of convergence or optimal hyperparameters.

Checkpoint selection uses validation accuracy, then lower validation loss,
then the earlier epoch. Test images are evaluated only after selection. The
selected checkpoint is reloaded before conversion. Every epoch records loss,
validation metrics, gradient magnitudes, and per-layer weight changes. The
report does not hide unchanged learned tensors or internal spike disagreements.

The comparison retains all eleven neural stages. It processes bounded chunks
aligned to the source batch size, limiting retained activation tensors while
preserving source inference batch boundaries. Counts and predictions are
concatenated, and mismatch locations are mapped back to their full-split sample
indices. This adds repeated graph preparation cost but avoids retaining a full
large split's intermediate spike tensors at once.

## Running the example

Use the [isolated official SLAYER environment](official_slayer_deployment.md#reproduce-the-first-training-and-deployment-check)
and build the C runtime from the same Lacuna checkout. From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"

python examples/train_official_slayer_alexnet_mnist.py --preflight-only
```

The preflight reports an expanded graph estimate and a measured synthetic
forward/backward step. Its projected training time excludes validation,
conversion, replay, and runtime variability. It is not a performance benchmark.

A bounded training-and-conversion pilot is:

```bash
python examples/train_official_slayer_alexnet_mnist.py \
  --train-samples 2000 --validation-samples 200 --test-samples 200
```

Running without sample-count overrides uses the default 10,000/1,000/1,000
subsets. To use every canonical training image across training and validation,
and every canonical test image:

```bash
python examples/train_official_slayer_alexnet_mnist.py \
  --train-samples 55000 --validation-samples 5000 --test-samples 10000
```

The larger run requires substantially more compute. These commands describe
available protocols, not claims that all of them have been run.

`--training-only-indices` accepts a JSON integer list or a development report
containing `input.indices`. It reserves any images used during configuration
development for training and excludes them from validation. The selected
indices and evidence hash are saved in the protocol.

## Artifacts and resume

Each invocation creates a new directory under
`artifacts/official-slayer-alexnet-mnist`. The protocol freezes source files,
runtime-library identity, dataset hashes, splits, parameters, and selection
rules. Artifacts include the epoch history, optimizer checkpoints, selected
checkpoint, spike inputs, JSON network, binary deployment image, and separate
validation/test comparison reports.

Epoch checkpoints use separate filenames. An atomic manifest update points to
the latest completed checkpoint only after it is saved, retaining the previous
completed epoch if an interruption occurs before publication. Resume restores
the model, Adam state, selected checkpoint, history, and random state. Code,
data, and runtime changes are rejected on resume.

`--max-training-seconds` pauses at an epoch boundary after the specified
training-session budget. It does not export or evaluate test data while epochs
remain. Pass `--resume` with the printed artifact directory to continue. The
budget defaults to zero, meaning no time limit, including on a resumed run.
An interrupted export can resume from completed training. A completed report
cannot be overwritten by resume.

The binary image can also be checked with
`validation/official_slayer_native_deployment.py --artifact` followed by the
completed directory. That utility compares JSON-compiled execution with a
standalone C loader. Neither Python nor SLAYER runs in the native child.

Matching source and target equations do not guarantee bitwise spike agreement:
official SLAYER state truncation and float32 accumulation differ from Lacuna's
analytical binary64 propagation. Source/target comparisons report the actual
disagreements. Native JSON-versus-binary equivalence is a separate check and
does not establish bitwise SLAYER equivalence.

## Measured local pilot

The completed pilot on 2026-09-12 used 2,000 training images, 200 validation
images, and 200 images from the canonical test partition. It trained all eight
learned layers for ten epochs with the configuration above. This is a small
conversion example, not a full-MNIST accuracy result or a multi-seed study.

Epoch 10 was selected using validation only. Both epochs 9 and 10 classified
196 of 200 validation images correctly, and epoch 10 had lower validation
loss. The held-out test set was not used to select the checkpoint.

| Split | Official SLAYER | Converted Lacuna | Matching predictions |
| --- | --- | --- | --- |
| Validation | 196/200 (98.0%) | 197/200 (98.5%) | 199/200 (99.5%) |
| Test | 193/200 (96.5%) | 192/200 (96.0%) | 199/200 (99.5%) |

The selected weights differed from initialization in every learned layer.
All three pooling kernels remained unchanged. The imported network contained
20,842 neurons and 1,573,216 edges and used `compiled_sparse` execution.
The implementation and runtime were unchanged during training and replay,
as checked against the frozen protocol hashes.

Spike agreement was not exact. The following counts compare binary spike
presence at each sample, neuron, and time bin. A spike shifted between bins
can therefore produce two mismatched bins. No off-grid spikes were found.

| Neural stage | Validation mismatched bins | Test mismatched bins |
| --- | ---: | ---: |
| Conv 1 | 44 | 17 |
| Pool 1 | 29 | 10 |
| Conv 2 | 1,032 | 433 |
| Pool 2 | 496 | 232 |
| Conv 3 | 4,591 | 2,191 |
| Conv 4 | 8,202 | 4,590 |
| Conv 5 | 7,032 | 4,518 |
| Pool 3 | 4,961 | 3,290 |
| Dense 1 | 2,758 | 1,850 |
| Dense 2 | 2,358 | 1,601 |
| Readout | 530 | 424 |

The source and target differ in finite-precision accumulation and per-bin
state truncation. Threshold decisions can diverge and affect later layers.
Consequently, close classification agreement should not be interpreted as
identical internal dynamics. The per-layer reports retain spike totals and
first mismatch locations in addition to these counts.

The measured artifact directory is
`artifacts/official-slayer-alexnet-mnist/20260912T172047310802Z`.
Its `report.json` SHA-256 is
`2afdb607de9dcc775358f3f0b81153f179e860205704f60d4de0ae10cc55c190`.
The selected `official_checkpoint.pt` SHA-256 is
`f17d61fda0d3a88bbcadc85a856f293f72ca00e0a887f4b71d5f07d0e58fbc64`.
The binary deployment image `network.lcbin` SHA-256 is
`4e0787f4a9cf6fc0ce656d5efa4bd5f5e5186c0eac4af1cc56018496854c3529`.
Generated checkpoints and graphs are local artifacts, not bundled in Git.

The measured invocation was:

```bash
python examples/train_official_slayer_alexnet_mnist.py \
  --train-samples 2000 --validation-samples 200 --test-samples 200 \
  --training-only-indices artifacts/alexnet-preflight/train-32x16.json
```

That development report reserved the following canonical training indices.
For reproduction without the report, save this list as a JSON file and pass
its path to `--training-only-indices`. This reproduces the data split, although
the evidence-file hash will differ from the original development report.

```json
[
  55659, 16011, 49605, 5335, 54106, 42192, 9086, 42415,
  32449, 47227, 822, 11716, 44789, 34217, 21540, 1846,
  895, 8101, 15748, 58554, 56833, 30778, 32400, 7312,
  50328, 49748, 5338, 7600, 38074, 7386, 45755, 51070
]
```

The run used Python 3.10.16, Torch 2.9.0, and unmodified official Lava-DL
commit `d825ac506eb9bb91ea2b945116fb45a40e775451`. The protocol records dataset,
frontend, example, and runtime hashes. The observed training-and-validation
time was 684.96 seconds, followed by 777.23 seconds for conversion and
cross-engine comparison. These are local workflow durations, not controlled
performance measurements or estimates for a full-data run.

### Native deployment check

The existing standalone C validator replayed all 400 validation and test
images from the saved `network.lcbin`. Its child process linked only to the
Lacuna runtime and system library, with no Python or SLAYER execution.
Every ordered spike, raw final state, and last-update time matched the
JSON-compiled Lacuna replay bit-for-bit. This covered 16,579,347 spikes,
including input relays, with zero failed samples. The original per-image
output counts, predictions, and per-layer spike totals were reproduced too.

The native evidence is stored in
`artifacts/official-slayer-native-deployment/20260912T235105675606Z`.
Its `report.json` SHA-256 is
`a440cce52fb3e531e06e6391a5707094fe1721701b183a796ffce0b9f60ee6df`.
This establishes deployment-image equivalence on the tested host and runtime.
It does not establish cross-platform bit identity or remove the measured
SLAYER-to-Lacuna differences above.

### Regression checks

All 48 example-specific tests passed, including real eleven-stage conversion,
gradient flow through every trained layer, fixed pooling weights, checkpoint
integrity, interrupted-versus-uninterrupted training, and chunked comparison
equivalence. The existing importer suite passed 282 tests. The ordinary Python
suite without the optional Torch environment passed 805 tests with 10 skips,
and the C test executable passed. No importer or C runtime source changes were
needed for this example.
