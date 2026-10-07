# Official SLAYER pooling import and MNIST validation

Official CUBA `Pool` layers can be imported as ordinary Lacuna LIF neurons
with fixed delta connections. This is an importer extension, not a C pooling
operator or a new execution path. It reuses the framework-neutral depthwise
convolution representation. See the
[deployment contract](official_slayer_deployment.md#spiking-pooling-deployment)
for supported geometry and source settings.

The first supported geometry uses fixed square or rectangular kernels, stride
equal to the kernel size, zero padding, unit dilation, and evenly divisible
spatial dimensions. The actual fixed kernel values are preserved independently
for each channel. Tensor max/average pooling, overlapping windows, irregular
dimensions, and trainable pooling kernels remain outside this profile.

## Validation procedure

The pooling tests compare source synaptic outputs and imported edge deposits
against independently calculated window sums. Distinct channels and asymmetric
signed kernels check channel isolation and kernel orientation. Hand-specified
spike sequences check membrane integration, reset, simultaneous input, and
spatial flattening. A Conv–Pool–Conv–Pool–Flatten–Dense test compares every
neural layer, including both pools, against the real official implementation.

Other checks cover JSON and compiled-binary replay, malformed geometry,
unsupported neuron features, and changed source fingerprints. A deliberately
near-threshold case retains the expected source/target quantization disagreement
instead of treating matching equations as a bitwise guarantee.

The focused importer, pilot, and validation suite passed 448 tests, including
58 pooling importer cases and 18 pooling-pilot cases. The ordinary no-Torch
Python suite passed 723 tests with 9 skips, and the C core test executable
passed. These suites overlap and their counts should not be added together.
One upstream deprecation warning comes from deliberately exercising rejected
weight normalization.

## Trained pilot protocol

The example uses a fixed-seed architecture with two convolutional layers, two
spiking pooling layers, and a dense readout:

| Layer | Configuration | Output shape |
| --- | --- | --- |
| Input | Binary image spikes | 1×28×28 |
| Conv | 4 filters, 3×3, stride 1, padding 1 | 4×28×28 |
| Pool | Fixed 2×2 sum pooling, stride 2 | 4×14×14 |
| Conv | 8 filters, 3×3, stride 1, padding 1 | 8×14×14 |
| Pool | Fixed 2×2 sum pooling, stride 2 | 8×7×7 |
| Flatten | Channel, row, column order | 392 |
| Dense | 10 spiking outputs | 10 |

Each pooling block contains ordinary CUBA neurons, so it performs spiking
integration after the local sum. Its output is not the numeric window sum.
The three learned tensors contain 36, 288, and 3,920 weights. The two four-entry
pooling kernels remain fixed at 1.0. The expanded Lacuna graph has 6,674 neurons
including input relays and 86,720 edges before any zero-weight omission.

The protocol uses 10,000 training, 1,000 validation, and 1,000 canonical test
images, with balanced sampling and split seed 137. Training uses model seed 0,
10 epochs, batch size 128, and Adam at learning rate 0.003. Inputs contain
32 bins with Bernoulli probability `0.4 * pixel / 255`. Neurons have
`current_decay=1`, `voltage_decay=512/4096`, threshold 1, zero initial state,
and constructor `scale=4096`, corresponding to internal state scale 262,144.
Quantization hooks, output shifts, learned delays, and normalization are off.

Checkpoint selection uses validation accuracy, then lower validation loss,
then the earlier epoch. Test images do not select the checkpoint. Source replay
uses the same batch size as source validation. The persisted checkpoint is
reloaded before export, and all five neural layers are compared on all saved
validation and test inputs.

Before training, a synthetic activity check found that quarter-weight pooling
at both stages left the second pool silent in this initialization. The pilot
therefore uses official default unit-weight sum pooling. This was an activity
preflight, not an accuracy search. Quarter-weight pooling remains supported by
the importer and independently tested. Source training must change each learned
weight tensor and leave both pooling kernels unchanged.

The script freezes the protocol, source files, dataset identities, and runtime
library before training and checks them again after training and replay. Saved
artifacts include the selected checkpoint, spike inputs and labels, all-layer
comparisons, weight-change audits, network JSON, and compiled deployment image.
The existing standalone C validation utility can replay the resulting image
without SLAYER or Python in the native process.

## Measured training and transfer results

The run on September 12, 2026 completed all ten epochs and selected epoch 4.
Its saved checkpoint was reloaded before conversion. All frozen protocol,
source, dataset, library, and artifact hash checks passed. The final exported
graph has 6,674 neurons and 86,720 edges and uses `compiled_sparse` execution.

| Split | Images | Official SLAYER correct | Lacuna correct | Matching predictions |
| --- | ---: | ---: | ---: | ---: |
| Validation | 1,000 | 935 (93.5%) | 934 (93.4%) | 999 / 1,000 |
| Test | 1,000 | 932 (93.2%) | 932 (93.2%) | 1,000 / 1,000 |

The one differing validation prediction was sample 973, a digit 4. SLAYER
predicted 4 and Lacuna predicted 9. Test classifications matched, but neither
split had identical internal spike trains:

| Neural layer | Differing validation spike bins | Differing test spike bins |
| --- | ---: | ---: |
| First convolution | 27 | 38 |
| First pooling layer | 20 | 26 |
| Second convolution | 234 | 257 |
| Second pooling layer | 184 | 219 |
| Dense readout | 86 | 77 |

No off-grid spikes were observed. The importer preserves the existing warning
that official fixed-point state truncation and float32 accumulation need not
produce the same threshold decisions as analytical binary64 Lacuna execution.
The comparisons retain every layer's spike totals and first mismatch, not just
classification scores.

The selected checkpoint's weight-change L2 norms relative to initialization
were approximately 7.03e-9, 0.0046913, and 3.35979 for the first convolution,
second convolution, and dense readout, respectively. Both fixed pooling kernels
were bit-identical to their initial values. All three learned tensors were
included in optimization and changed, but the first convolution's movement
was negligible. This is not evidence of substantial feature learning at every
layer.

This is one fixed-seed conversion pilot on MNIST subsets, not a full-dataset
accuracy result, architecture comparison, hyperparameter search, or estimate of
seed variability. Training took 1,061.59 seconds and export plus cross-engine
comparison took 206.31 seconds on the local host. These are workflow durations,
not controlled simulator benchmarks.

Artifacts are stored in
`artifacts/official-slayer-pool-mnist/20260912T153515458136Z/`.
The report SHA-256 is:

```text
2d115bab166dcce0dbcf7b1b92b8e3f1b0b9f8d99df50b5d90a2e2348641367e
```

## Standalone C deployment validation

The saved binary image was loaded and replayed by the existing standalone C
utility for all 2,000 validation and test episodes. Its output matched the
JSON-compiled Lacuna reference bit for bit, including 8,815,113 ordered spikes
with input relays, 13,348,000 final state values, and their last-update times.
No episode failed, and the original Lacuna per-image spike counts and
classifications were reproduced.

The native process linked only the Lacuna runtime and system library. Python
and Torch prepared inputs and the comparison reference on the host, but neither
was present in the native replay process. This establishes binary deployment
equivalence on the same host and runtime. It does not establish bitwise
agreement with SLAYER or execution on a different processor.

The deployment report is stored in
`artifacts/official-slayer-native-deployment/20260912T155707979424Z/`.
Its SHA-256 is:

```text
4e8709b03d60de8d7929cc26afca40121af1f7b28655debed89263cecbd654a9
```

## Reproduce

Use the isolated official SLAYER environment and pinned source checkout from
the deployment guide. Run these commands from the Lacuna checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"

python -m pytest -o addopts='' -q \
  tests/python/test_official_slayer_pool_import.py \
  tests/python/test_official_slayer_pool_mnist.py

python examples/train_official_slayer_pool_mnist.py
```

Replay a completed pilot through the standalone C loader with:

```bash
python validation/official_slayer_native_deployment.py \
  --artifact artifacts/official-slayer-pool-mnist/20260912T153515458136Z
```

The example creates a fresh directory under
`artifacts/official-slayer-pool-mnist/`. It does not overwrite earlier results.
If training completed but export was interrupted, `--resume` accepts that
directory only when its frozen code, data, and selected checkpoint still match.
It does not resume a completed report or restart partially completed training.
