# Official SLAYER convolutional MNIST deployment pilot

This fixed, small pilot trained a convolutional LIF network in the unmodified
official SLAYER implementation and deployed it through ordinary Lacuna
LIF/delta execution. It tested the new spatial importer, not an accuracy
search or the maximum attainable MNIST performance.

## Architecture and procedure

Each 28 by 28 grayscale image used 784 input neurons. The first convolution
had four 5 by 5 filters with stride two, producing 4 by 12 by 12 neurons.
The second had eight 3 by 3 filters over the four input channels with stride
two, producing 8 by 5 by 5 neurons. An official Flatten block preserved
channel, row, column order for a fully connected layer of ten output LIF
neurons. There was no pooling, normalization, bias, recurrence, or layer delay.

All three weight layers trained in SLAYER. They contained 100, 288, and 2,000
unique weights respectively. Export expanded these 2,388 coefficients into
23,600 signed static delta connections. The deployed graph contained 1,570
neurons, including its 784 input relays. Flatten did not add any runtime
operation or neuron. No C runtime change was needed.

The pilot reused the dense pilot's balanced split and encoding protocol:
10,000 training and 1,000 disjoint validation images from canonical MNIST
training data, plus 1,000 images from the separate canonical test pool.
Split seed was 137 and model seed was zero. The actual indices and data
hashes are retained in the protocol artifact. The verified local dataset was
not downloaded again.

Each pixel produced a binary spike independently in each of 32 bins with
probability `0.4 * pixel / 255`. Training used fresh spike realizations each
epoch. Validation and test inputs used fixed separate seeds and were replayed
unchanged in both engines. Ten epochs used Adam at learning rate 0.003,
batch size 128, and official SpikeRate loss with target rates 0.2 and 0.02.
Checkpoint selection used validation accuracy, then lower validation loss and
earlier epoch. No test predictions selected a checkpoint or changed settings.

Every neural layer used current decay 1, voltage decay 512/4096, threshold 1,
constructor `scale=4096` (internal state scale 262,144), and fixed shared
neuron parameters. Rest, reset, drive, and
refractory duration were zero. Source inference used batches of 128 both
during validation selection and during cross-engine replay.

## Measured results

Epoch 10 was selected at 817/1,000 validation accuracy (81.7%). Official
SLAYER and Lacuna each classified 829/1,000 test images correctly (82.9%).
All 1,000 test predictions agreed between the engines. This is lower than the
86.1% measured for the prior dense pilot on the same test subset, so it is not
an accuracy improvement. The convolutional pilot also used substantially
fewer unique trainable weights than that dense model's 50,816.

Validation predictions agreed on 998/1,000 images, not all 1,000. Lacuna
classified 818 correctly versus SLAYER's 817. On validation sample 569, with
true digit 7, SLAYER predicted 5 and Lacuna predicted 7. On sample 864, with
true digit 2, they predicted 0 and 4 respectively. These differences did not
change checkpoint selection, which used only the official source model.

Spike trains were not identical. Test-batch differences were:

| Neural layer | SLAYER spikes | Lacuna spikes | Differing spike bins |
| --- | ---: | ---: | ---: |
| First convolution | 529,880 | 529,883 | 51 |
| Second convolution | 241,007 | 241,010 | 35 |
| Dense output | 11,746 | 11,744 | 26 |

Validation had 34, 113, and 62 differing bins respectively. A shifted spike
can account for two differing bins, so these numbers are not counts of
distinct displaced spikes. Matching predictions do not imply exact spike
equivalence. The known float32 accumulation and fixed-point source state
operations remain different from Lacuna's analytical binary64 propagation.
The report preserves mismatches rather than suppressing them.

Selected weights changed from initialization with L2 norms 0.00997634,
1.05999506, and 5.46682072 for the three layers. The first convolution changed
much less than the later layers. This observation alone does not establish
why this small architecture performed below the dense pilot.

Training and validation selection took 78.60 seconds locally on one CPU
thread. Export, both cross-engine comparison batches, and binary generation
took 56.73 seconds. These are workflow timings, not controlled simulator
benchmarks. Lacuna used `compiled_sparse` execution.

## Artifacts and reproduction

The completed run is
`artifacts/official-slayer-conv-mnist/20260912T142912966153Z/`.
It contains `report.json`, `protocol.json`, `history.json`,
`training_summary.json`, `official_checkpoint.pt`, `validation_inputs.pt`,
separate validation/test comparison reports, `network.json`, and
`network.lcbin`. Input tensors, labels, actual source revision, library hash,
workflow source hashes, and artifact hashes are retained.

The report SHA-256 is
`1d69037ca486a6864c45a1feda048f072227b86e1b33cdcfb3671ca067994de0`.
All recorded artifact and workflow hashes were verified after completion.
The official source was clean at
`d825ac506eb9bb91ea2b945116fb45a40e775451`.

After the isolated setup in [official SLAYER deployment](official_slayer_deployment.md),
run from the Lacuna checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"
python examples/train_official_slayer_conv_mnist.py \
  --train-samples 10000 --validation-samples 1000 --test-samples 1000 \
  --epochs 10 --bins 32 --batch-size 128 --learning-rate 0.003 \
  --seed 0 --split-seed 137
```

Runs use new output directories. If training completes but export fails,
`--resume` accepts that run directory and reuses its selected checkpoint
without retraining, after checking training code, source environment, data,
architecture, and checkpoint identity. It refuses completed reports.

A separate 20-training-image, 10-validation-image, 10-test-image smoke run
verified the end-to-end workflow before this pilot. Its accuracy is not used
as a performance result. Geometry tests independently cover asymmetric
kernels, stride, padding, dilation, multiple channels, groups, depthwise
convolution, CHW flattening, and small-network layer-level spike agreement.
See the deployment guide for the supported model restrictions.
