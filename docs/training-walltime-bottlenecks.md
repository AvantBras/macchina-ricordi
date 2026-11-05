# Training wall-time: historical measurements and profiling cautions

> **Historical evidence, not a current tuning guide.** These figures came from the
> author's September 2026 experiments with fresh audio crops. The model, source
> corpus, profiling logs and original benchmark scripts are not supplied here.
> They cannot establish the bottleneck of a new run. For current training options,
> see [planning and launching training](musicgen.md#3-plan-or-launch-training).

## What was measured

The original notes reported the following on the author's workstation:

| Measurement | Reported result |
|---|---|
| Random-file FLAC decode with soundfile | 14.7 ms/read |
| Random-file MP3 decode with ffmpeg | 27.5 ms/read |
| Reshape runs at batch 2–4 | about 2.75–2.80 updates/s |
| Nano run at batch 8 | 1.37 updates/s |

These are different measurements and runs, not a controlled throughput sweep.
The decode comparison does not imply that converting a dataset makes an entire
training run proportionally faster.

## The correction: updates/s is not samples/s

The earlier version of this note treated slower updates and longer epochs as
proof that batch 8 harmed throughput. That conclusion was not justified.
For the configuration under discussion, an epoch had a fixed 2,000 updates:
a larger batch therefore processed more audio per epoch.

At batch 4, 2.75–2.80 updates/s corresponds to 11.0–11.2 samples/s. At batch 8,
1.37 updates/s corresponds to 10.96 samples/s: roughly the same sample throughput,
not a twofold slowdown. The runs were not a matched benchmark, so even that small
difference is not evidence of a batch-size penalty. Compare samples/s or audio
seconds/s at matched segment duration, not epoch duration or updates/s alone.

The original timing observations also do not prove that CPU decoding, GPU codec
encoding, transfers, or language-model compute was the dominant stage. Profile
those stages rather than selecting a cause from GPU utilization or VRAM occupancy.

## Fresh audio and token pools have different costs

Without `--pool`, training draws fresh crops and encodes their audio during
training. With `train run --pool`, the training crops are encoded into a token
pool before the language-model run; validation still uses its audio path. Pool
construction has an upfront cost, and repeated training draws from a finite set
of cached crops. See the token-pool subsection of [the training guide](musicgen.md#3-plan-or-launch-training)
for sizing, reuse and the memorization trade-off.

Neither path is universally faster for every model, dataset, schedule or machine.
Include pool construction when measuring a first run, and distinguish that result
from reuse of an existing pool.

## How to measure a new run

- Hold model, segment duration, data distribution and work performed constant.
- Compare samples/s or audio seconds/s as well as updates/s, peak memory and total
  elapsed time. Record the update count and batch size behind every epoch time.
- Profile data loading/decoding, host-to-device transfer, codec encoding and
  language-model forward/backward work separately, accounting for overlap.
- Record GPU model, driver/runtime, power limit, CPU, workers, storage and cache
  state. Change one variable at a time; do not infer useful batch size from free
  VRAM alone.
- Measure rather than assuming that larger batches help or hurt. Small models,
  fixed update budgets and cached tokens make comparisons especially easy to misread.

The [PyTorch profiler guide](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
and [data-loading tutorial](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html)
provide tools for making these measurements. This note makes no claim that a
fresh profiling run has been performed for this public checkout.
