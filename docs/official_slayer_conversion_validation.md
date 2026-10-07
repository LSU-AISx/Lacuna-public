# SLAYER-to-Lacuna conversion validation

The additional deployment, numerical, and repeated-training checks completed
on September 12, 2026. They support conversion of the restricted official
Lava-DL SLAYER LIF/delta profile described in the
[deployment guide](official_slayer_deployment.md). They do not establish
universal spike equivalence or support for arbitrary SLAYER checkpoints.

Training used the unmodified official source at
`d825ac506eb9bb91ea2b945116fb45a40e775451` with Torch 2.9.0 on CPU.
These checks added validation utilities and tests, not a training runtime,
neuron executor, or precision mode to Lacuna.

## Repeated training and transfer

Six fresh runs used seeds 0, 1, and 2 for each architecture. Every run trained
on 10,000 MNIST images for 10 epochs, selected its checkpoint using 1,000
validation images, and evaluated 1,000 held-out canonical test images.
The image split was fixed with seed 137. Model seeds changed initialization,
training order, and spike realizations, not just initial weights.

The dense model was 784→64→10. The CNN used a 1→4 convolution with a 5×5
kernel and stride 2, a 4→8 convolution with a 3×3 kernel and stride 2,
then flattening and a 200→10 dense output. Every weight layer was trained.
Both architectures used Adam at learning rate 0.003, training batches of
128, and 32 binary input bins with firing probability `0.4 * pixel / 255`.
Source replay batch sizes were 64 for dense models and 128 for CNNs.

All runs used constructor `scale=4096`, corresponding to an internal source
state scale of 262,144 and quantum of approximately 3.815e-6. This is not
the same as an internal state scale of 4,096.

| Architecture | Seed | Selected epoch | SLAYER test accuracy | Lacuna test accuracy | Different test predictions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense | 0 | 9 | 86.1% | 86.1% | 0 / 1,000 |
| Dense | 1 | 10 | 86.1% | 86.1% | 0 / 1,000 |
| Dense | 2 | 9 | 85.7% | 85.7% | 0 / 1,000 |
| Convolutional | 0 | 10 | 82.9% | 82.9% | 0 / 1,000 |
| Convolutional | 1 | 9 | 84.1% | 84.1% | 0 / 1,000 |
| Convolutional | 2 | 10 | 83.1% | 83.1% | 1 / 1,000 |

For CNN seed 2, the differing test image had label 9. SLAYER predicted 0 and
Lacuna predicted 4, so both were incorrect and aggregate accuracy was unchanged.
Validation predictions differed on 2, 1, and 0 images for CNN seeds 0, 1, and 2.
The corresponding SLAYER/Lacuna validation correct counts were 817/818,
835/836, and 810/810. Dense validation predictions matched in all three runs.

Every trained model had some internal spike-bin differences. No target spikes
occurred off the input grid. Identical accuracy therefore must not be described
as identical spike trains or identical decisions on every input.

No hyperparameter search or test-based seed selection was performed. The six
runs share image indices, so they are not six independent test sets. Their
seed variation is descriptive, not an estimate of population accuracy with a
confidence interval. These are conversion checks on MNIST subsets, not a
full-data accuracy study or a performance benchmark.

## Numerical stress matrix

The fixed matrix crossed two source scales, four architectures, and six input
patterns for 48 cases. Architectures covered two-layer dense neurons with an
exact binary decay, three slow-decay layers, six layers spanning source decay
integers 1 through 4095, and grouped convolutions followed by a dense layer.
Inputs covered silence, bursts, sparse and dense activity, signed cancellation,
and near-threshold deposits. Each case had four samples lasting 256 or 1,024
bins, with input intervals of 0.1, 0.25, or 1.0 depending on the architecture.

All 48 cases completed without an execution error. All 22 predeclared exact
controls passed. These controls have independently predictable silence or
identity spike sequences. The remaining stress cases report differences
without imposing a tolerance chosen after seeing the results.

| Source constructor scale | Internal state quantum | Cases with identical all-layer spikes |
| ---: | ---: | ---: |
| 64 | 1 / 4,096 | 15 / 24 |
| 4,096 | 1 / 262,144 | 23 / 24 |

Across the full matrix, 38 cases matched exactly, 26,250 spike bins differed,
and no off-grid target spikes occurred. The sole finer-scale mismatch was the
six-layer dense network under dense input, with 32 differing spike bins.
Coarser quantization produced much larger differences in deliberately
near-threshold cases. In the coarse grouped-convolution near-threshold case,
SLAYER emitted no spikes while Lacuna emitted 2,304, 1,024, and 1,024 spikes
in its three layers.

This exposes a real model-compatibility limit. Source state truncation and
float32 accumulation are not the same dynamics as ordinary unquantized
binary64 Lacuna propagation. Fine source quantization reduced disagreement
in this matrix, but cannot guarantee identical threshold decisions. Synthetic
argmax agreement is not a classification result and is not used as evidence
of transfer quality here.

## Standalone C deployment

The previously saved dense and CNN pilot artifacts were each replayed on
their complete 1,000 validation and 1,000 test episodes. A standalone C
program loaded the existing binary image and ran the ordinary Lacuna core.
Its ordered spikes, raw final states, and last-update times were compared
bit-for-bit with a fresh JSON-compiled Lacuna run on the same inputs.

| Trained artifact | Episodes | Ordered spikes checked | Final state values checked |
| --- | ---: | ---: | ---: |
| Dense MNIST | 2,000 | 2,927,628 | 1,716,000 |
| Convolutional MNIST | 2,000 | 4,202,289 | 3,140,000 |

All 7,129,917 ordered spikes, including input relays, matched. All 4,856,000
final state values and equally many last-update times matched. Original
per-image output counts, predictions, and per-layer totals were reproduced.
There were no failed samples.

The host-side preparation script uses Torch to read saved input tensors.
The replay child itself uses only the compiled C executable, Lacuna library,
and system libraries. It does not import Python, Torch, or SLAYER and does
not run the model compiler. Dependency inspection and exact output streams
are preserved with the results.

This checks same-host binary deployment on macOS ARM64 with the same runtime
library. It is not a microcontroller test, cross-platform bitwise guarantee,
or claim of bitwise agreement with SLAYER.

## Reproduction and evidence

Activate the isolated official SLAYER environment described in the deployment
guide, build Lacuna, and run from the checkout root. These commands create
fresh output directories and do not overwrite prior studies.

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"

python validation/official_slayer_numerical_matrix.py
python validation/official_slayer_transfer_campaign.py
```

For native replay, pass one or more actual trained artifact directories:

```bash
python validation/official_slayer_native_deployment.py \
  --artifact artifacts/official-slayer-mnist/20260912T140703852778Z \
  --artifact artifacts/official-slayer-conv-mnist/20260912T142912966153Z
```

Those two paths identify the local pilot artifacts checked here. On another
machine, transfer them or substitute freshly generated artifact directories.
Native validation requires the same library hash recorded by the source
artifact, so a different build should first generate its own pilot artifacts.
Use `--library` to select a library explicitly if needed.
On Linux, append `--library build/liblacuna_core.so` to the replay command.

The preserved reports are:

- `artifacts/official-slayer-transfer-campaign/20260912T150322997790Z/report.json`
- `artifacts/official-slayer-numerical-matrix/20260912T150724884781Z/report.json`
- `artifacts/official-slayer-native-deployment/20260912T150813478398Z/report.json`

Their SHA-256 values, in the same order, are:

```text
770428c9074bf208c229e5527d7f9df36a3960ad0f589154abc8bd53fa807114
f6893fc3b759bb0fe972300e6a2fb48d30d57f3b86b3e808ecbf2f8b184e1b24
b4b5155136051d8010710aa613b4d27535080235ba8835f85d429cdbd1f47029
```

Saved provenance includes source revisions, runtime and importer hashes,
dataset and split identities, checkpoints, input spikes, per-layer comparisons,
and artifact hashes. The numerical matrix was repeated after adding target
library provenance. All 48 comparison records were identical between those
two runs. The earlier matrix artifacts remain available rather than overwritten.

The focused importer, pilot, and validation suite passed 372 tests in the
official-source environment. The ordinary no-Torch Python suite passed 723
tests with 7 skips, and the C core test executable passed. These suites overlap
and their counts should not be added together. New tests cover the fixed
campaign protocol, analytically expected spike bins, malformed native replay
inputs, artifact integrity, and exact binary deployment comparisons.

## Supported conclusion

Compatible official SLAYER-trained dense and convolutional LIF/delta models
can be converted into ordinary Lacuna networks and deployed through its
existing C runtime without SLAYER on the target. The observed classification
agreement was high, but conversion does not reproduce source quantization.
Validate each exported model on representative inputs and retain spike-level
and decision-level disagreements, especially for coarse quantization or
threshold-sensitive applications.
