# MusicGen

The MusicGen tooling stays inside this repo, but runs from a dedicated repo-managed runtime:

- main project env: `.venv`
- MusicGen env: `.venv-musicgen`
- pinned AudioCraft checkout: `.audiocraft`

Training is **a recipe trained on a workspace** (see the [glossary](glossary.md)). Recipes are how to train:
the model's shape, codec, loader and schedule. They never name data; the workspace you pass decides that. All
of them are unconditional: `micro-unconditional` and the `reshape-*` family. The default is `reshape-nano`. Text
conditioning was removed from `main` on 2026-10-05; the recipes here are all unconditional and work on audio alone.

## Recipes

Recipes live in [`musicgen/pyproject.toml`](../musicgen/pyproject.toml) as Dora overrides:

```toml
[tool.macchina_ricordi.recipes.base]          # abstract: only there to be extended
abstract = true
codec = "encodec_32khz"               # a name from [tool.macchina_ricordi.codecs]
overrides = ["solver=musicgen/musicgen_base_32khz", "dataset.batch_size=3", "optim.epochs=5", ...]

[tool.macchina_ricordi.recipes.reshape-deep]
extends = "deep-stack"                # inherits everything, then:
overrides = ["transformer_lm.dim=768", "transformer_lm.num_heads=12", "transformer_lm.num_layers=16",
             "dataset.batch_size=2"]  # replaces the inherited batch size
```

`extends` starts from another recipe; a line replaces the inherited line with the same key. `abstract`
recipes (`base`, `deep-stack`, `nano-family`, `pico-family`) are not offered on their own. A recipe that sets `dset=` is
refused: the data comes from `--workspace-dir`, and training adds `dset=audio/<the workspace's egs name>`
itself. Each recipe renders exactly the override list its profile had before 2026-10-05, so Dora signatures,
and therefore `train run` continuing an existing XP, are unchanged. The one exception is the nano family
(`reshape-nano*`), which switched gradient checkpointing off on 2026-10-24: same gradients, ~15% faster. Re-running
an older nano `train run` command now starts a new XP, so continue one of those with
`train resume <sig> --epochs N`. Pick one with `--recipe` (`--profile` still works).

The four `reshape-pico*` recipes are approximately 3.7M parameters: `reshape-pico` (128 × 8),
`reshape-pico-wide` (192 × 2, 2× MLP), `reshape-pico-deep` (96 × 19), and
`reshape-pico-slim` (128 × 12, 2× MLP). They inherit the nano family's schedule, batch size and
checkpoint policy. With `--pool`, both families allow up to eight uses per crop (four for larger models).

The optional `reshape-deep-25hz` recipe is local-only: its `lowfps_25hz` codec
points to a placeholder for a compatible codec checkpoint that is not shipped.
Do not select it until you have supplied that checkpoint and configured its path
in `musicgen/pyproject.toml`. The default `reshape-nano` uses the public EnCodec codec.

## 1. Set Up The Runtime

```bash
make setup-musicgen
python -m macchina_ricordi.cli runtime check
```

You can override the discovered runtime paths when needed:

```bash
python -m macchina_ricordi.cli runtime check \
  --musicgen-python /path/to/python \
  --audiocraft-dir /path/to/audiocraft
```

### Maintaining the runtime

The MusicGen runtime is locked with `musicgen/uv.lock`, separate from the root project's `uv.lock`
(see [Security and trust boundaries](../SECURITY.md#accepted-maintenance-policy) for the policy).
Routine Python and GitHub Actions version updates are accepted on a low-frequency monthly cadence
and arrive as grouped Dependabot PRs; each entry updates its own lockfile (`uv.lock` at the root,
`musicgen/uv.lock` here). A lockfile advisory is advisory only and does not block other work. Lint is
correctness-only (`ruff check`, the `F`/`B` rules); formatting is optional.

GPU-stack upgrades (torch and the CUDA-index wheel set) are never auto-merged. Validate one locally
in an isolated worktree before adopting it in your working environment. The block below is a
reproducible **template**: both paths are placeholders, so nothing runs until you set them to files
you actually have.

```bash
# Replace both with absolute paths you own (a template, not real sample paths):
MODEL=/absolute/path/to/your/trusted/bundle         # a portable bundle you exported (models.md#export)
DATASET=/absolute/path/to/prepared-smoke-workspace  # a small workspace `data build` made; not a production set

# 1. build the runtime from the locked project (isolated worktree), then health-check it
make setup-musicgen
uv run --no-sync ricordi runtime check
# 2. generate from a known-trained model, independently of any new training
uv run --no-sync ricordi play generate "$MODEL" --duration-seconds 2
# 3. train one tiny, bounded epoch from scratch (reshape-nano; needs clips >= 2 s for the 2 s segment)
uv run --no-sync ricordi train run --recipe reshape-nano --dataset "$DATASET" --epochs 1 \
  --override optim.updates_per_epoch=2 --override dataset.train.num_samples=4 \
  --override dataset.valid.num_samples=2 --override dataset.generate.num_samples=2 \
  --override dataset.num_workers=0 --override dataset.batch_size=2 \
  --override dataset.segment_duration=2 --override checkpoint.save_every=1
# 4. set SMOKE_XP to the absolute XP path step 3 printed (do not invent a signature), then generate
SMOKE_XP=/absolute/path/to/worktree/.dora/xps/REPLACE_WITH_PRINTED_SIG
uv run --no-sync ricordi play generate --xp-path "$SMOKE_XP" --checkpoint 1 \
  --duration-seconds 2 --extend-stride 1
```

Step 4 generates independently from the model that step 3 trained; it does not warm-start, so it
assumes no foreign checkpoint shape. A lockfile check proves resolution, not GPU compatibility: these
four steps are the local gate, and there is no automated GPU CI, so a change that only passes the
lockfile check has not been shown to work on the target GPU.

## 2. Build A Dataset

A dataset is built once, in one step, into `datasets/<name>/`: the clips gathered from
source folders (whole files hard-linked, or cut into fixed-length slices), split into train
and valid (hard links again, so no audio is copied), and AudioCraft's manifests. The GUI's
Corpus page does the same with a preview ([datasets](gui.md#datasets)).

The `examples/sample_data/...` paths in the commands below are placeholders for source folders **you
provide** — this checkout ships no sample audio, so point `data build` at folders you gathered yourself.

```bash
# --dry-run prints the per-source table and writes nothing
python -m macchina_ricordi.cli data build noisy examples/sample_data/midwife "examples/sample_data/Barn Owl" --dry-run
# whole files
python -m macchina_ricordi.cli data build noisy examples/sample_data/midwife "examples/sample_data/Barn Owl"
# 10 s slices, 32 kHz mono FLAC, with sped-up and slowed copies of the train clips
python -m macchina_ricordi.cli data build noisy_10s examples/sample_data/midwife --slice 10 --speed-ratios 0.9,1.1

python -m macchina_ricordi.cli data list            # every dataset, built ones and legacy workspaces
python -m macchina_ricordi.cli data show noisy_10s  # how it was built, the models trained on it, its weights
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset noisy_10s --dry-run
```

`datasets/noisy_10s/` then holds `clips/` and `corpus.json` (every clip's origin), `data/train`
and `data/valid`, `egs/` (the manifests), `logs/` (the prepare record and the fingerprint) and
`dataset.json`, which describes the whole build: sources, slicing, split, counts, slice length,
fingerprint. `build_status.json` says whether the build finished.

**A dataset is never rebuilt.** `data build` refuses a name that exists; a different cut, or
the same sources with more files, is a new dataset under a new name. Anywhere a command reads a
dataset, its name works (`--dataset noisy_10s`, alias `--workspace-dir`), and so does a path.

Clips are named `<source>__<file>[__<n>]`, so they stay attributable (weights and the
GUI group by that prefix). Useful flags: `--valid-fraction` and `--seed` (split),
`--max-silence-fraction` (drop clips that are mostly silence), `--speed-ratios` and
`--speed-mode` (copies of **train** clips only, see [Augmentation](augmentation.md)),
`--exclude FILE` (repeatable).

Every build also looks for the same recording in two sources, say an album ripped once as
mp3 and once as FLAC, which neither a hard link nor the file size reveals: files with the
same title (the tag, else the name without its track number) and lengths within 2 s whose
loudness envelopes line up (10 s from 30% in, correlation ≥ 0.9). Without `--skip-duplicates`
they are listed; with it, one copy is kept, lossless over lossy, then the larger file. On this
library the same track in two formats scored ~1.0 and different clips sharing a name and
length (`set_01.wav`) at most ~0.6.

```bash
python -m macchina_ricordi.cli data build stott "examples/sample_data/2014 - Faith in Strangers" \
  "examples/sample_data/Andy Stott - Faith in Strangers (2014) - FLAC" --skip-duplicates \
  --exclude "examples/sample_data/2014 - Faith in Strangers/09 - Missing.mp3" --dry-run
```

If your source material is long recordings you want cut at quiet gaps rather than at fixed
lengths, segment them first ([Segmentation](segmentation.md)) and build from the output.

### Legacy workspaces, and the two lower-level steps

Before datasets, the two halves were separate: `data corpus` gathered clips into
`training_datasets/<name>/clips`, and `data prepare` copied a clip folder into a workspace
(`.workspace_<name>/`). Those workspaces are frozen. They still list, train and take weights
like datasets (their name is the folder name, dot included: `--dataset .workspace_x`), but
`data prepare` refuses one that is already prepared, since preparing wipes its `data/`,
`egs/` and `logs/` (run records included); `--overwrite` does it anyway. Both commands remain
for a clip folder you assembled yourself:

```bash
python -m macchina_ricordi.cli data corpus noisy_10s examples/sample_data/midwife --slice 10
python -m macchina_ricordi.cli data prepare training_datasets/noisy_10s/clips --workspace-dir .workspace_noisy_10s
```

### Audit dataset slice lengths

A workspace's audio clip length `S` and the recipe's `dataset.segment_duration`
`D` must satisfy `D <= S`. When `D > S`, the AudioCraft loader disables random-crop
augmentation and silence-pads every example up to `D`, so the model learns "music
then silence" as structure. `prepare` now records each workspace's clip-length
stats, and training **refuses** a run where `D` exceeds the median clip (pass
`--allow-segment-padding` to override, which records the decision).

To review the whole back-catalog at a glance:

```bash
python -m macchina_ricordi.cli data audit          # table for every workspace
python -m macchina_ricordi.cli data audit --json    # machine-readable rows
python -m macchina_ricordi.cli data audit --workspace-dir .workspace_mega_10s
```

The `safe_D` column is the largest `segment_duration` a workspace can train at
without padding; `short` counts slicing-remainder fragments worth filtering.

### Rebalance the training distribution (anti-wells)

Unconditional models can collapse whole acoustic modes ("wells"): on
`mega_10s`, clusters covering ~19% of the data received ~0% of generations
while the dominant percussive cluster was over-produced 2–3×. AudioCraft
draws training files in proportion to their `weight` when weight sampling is on
(MusicGen's base config turns it off), so the fix is per-segment weights. They are
kept as a named **weights overlay**: a weighted copy of the train manifest under
`artifacts/weights/<workspace>/<name>/`. The workspace itself is never written, so
one workspace can carry several weightings and a run uses one only when asked:

```bash
# Look at the clusters first (flattens the cluster prior; no generations needed):
python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_mega_10s --dry-run

# Save weights under a name:
python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_mega_10s --save-as flat_a05

# Coverage mode: upweight what a model under-produces, judged from its gens:
python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_mega_10s \
  --gens-dir artifacts/sweeps/uncond_deep90_20260608T080840Z --save-as cov_deep90_r1

# Compute now, look, save later (what the GUI does):
python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_mega_10s --plan-out /tmp/mega_plan.json
python -m macchina_ricordi.cli data rebalance --workspace-dir .workspace_mega_10s \
  --apply-plan /tmp/mega_plan.json --save-as cov_deep90_r1

# What is saved, and train with one:
python -m macchina_ricordi.cli data weights list --workspace-dir .workspace_mega_10s
python -m macchina_ricordi.cli train run --recipe reshape-deep --workspace-dir .workspace_mega_10s \
  --weights cov_deep90_r1
python -m macchina_ricordi.cli data weights delete cov_deep90_r1 --workspace-dir .workspace_mega_10s
```

Weights are clipped to `[1/cap, cap]` (default 4) and normalised to mean 1, so
the epoch scale is unchanged. The overlay's `overlay.json` records the decision.
`train run --weights` points the train split at the overlay, re-enables weight
sampling for it, and suffixes the auto dataset label with `_rw<digest>` so the
weighted run gets a fresh Dora signature. The valid split is never reweighted
(valid ppl stays comparable across runs: judge rebalanced models by generation
coverage, not valid loss). An overlay goes stale if the workspace's manifest
changes; training then refuses it. The wells protocol (alpha 0.5, a fresh sweep
each round) is now a list of overlays side by side: `cov_r1`, `cov_r2`, ...

Workspaces weighted in place before overlays existed (a `rebalance.json` next to
the train manifest) still train with those weights; `data rebalance --reset`
clears them. None is weighted that way as of 2026-10-05.

### Mix by source folder

Clusters are one way to weight; `data mix` is the other. It gives each source folder a share of the train draws, typed as relative numbers, and keeps the result as an overlay like `data rebalance` does, so `--weights` and the `_rw` label work the same way:

```bash
python -m macchina_ricordi.cli data mix --dataset noisy_10s album_a=3 album_b=1 --dry-run
python -m macchina_ricordi.cli data mix --dataset noisy_10s album_a=3 album_b=1 --save-as a3_b1
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset noisy_10s --weights a3_b1
```

A clip's chance of being drawn is its weight, so each clip is weighted `segments × share ÷ clips in its folder`: the folder's share of draws comes out exact and the average weight stays 1. Weights are not clipped. A clip's folder is the part of its name before `__` (clips are named `<folder>__<file>` when a dataset is built), and every folder with train clips must be listed. The shares are exact under the default per-file sampling. With *in proportion to length* sampling each clip's weight also scales with its length, so the long files of one folder count for more.

## 3. Plan Or Launch Training

Dry-run first: it plans and prints without launching training or recording a training job.
It can still refresh AudioCraft dataset configuration, build a dataset fingerprint,
and fetch a referenced checkpoint; it is not a read-only operation.

```bash
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset my_dataset --dry-run
```

Then launch the real run:

```bash
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset my_dataset
```

Useful training flags:

- `--epochs 30` instead of the recipe's (the same as `--override optim.epochs=30`)
- `--override key=value` appends Dora overrides
- `--dataset` (required; alias `--workspace-dir`) is the dataset to train on, by name or folder
- `--weights <name>` trains on a saved weights overlay (above)
- `--run-label <label>` sets an explicit Dora `label=...` override for a new run
- `--no-auto-dataset-label` disables the default dataset-fingerprint label and preserves legacy Dora signature behavior
- `--gpu-watchdog` enables the built-in GPU failure watchdog
- `--from <model>` (alias `--init-from-checkpoint`) warm-starts a new run from another model: a
  [model reference](models.md) (`deep@best`, a sig, a label), a `checkpoint.th` path, or a
  [portable bundle](models.md#export) directory such as `models/comelady`. It is a fresh run — a new
  optimizer and schedule — on the dataset you name with `--dataset`; no source corpus beyond the snapshot is
  needed. See [Checkpoint Reuse](musicgen-checkpoint-reuse.md) for the full flow.
- `--override +dataset.train.sample_on_duration=true` draws training crops in proportion to
  file length (every second of audio equally often) instead of every file equally often, which
  is the MusicGen default. It matters for workspaces of whole tracks of very different lengths,
  barely for clips cut to one length. The train split only, so validation stays comparable.
  Per-clip cluster weights are a separate switch: `--weights` (above).
- `--pool` trains from a token pool (below) instead of fresh crops; `--pool-seed N` draws a different pool.

### Train from a token pool (`--pool`, opt-in)

By default every training step draws fresh crops and runs the frozen EnCodec encoder on them, which is two thirds of
a nano step. With `--pool`, the crops are encoded once, before training, and each comes back a fixed number of
times. A nano run is about 8× faster. The default stays fresh crops until blind A/Bs say the pool sounds the same
(stress test).

```bash
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset my_dataset --epochs 60 --pool --dry-run
```

- **Size:** the larger of one crop per second of training audio and the run's crops ÷ 8 (÷ 4 above the nano
  family), rounded up to 1,000. The dry run prints the size, the repeats, the build time and the disk it needs.
- **Where:** `artifacts/token_pools/<dataset>-<fingerprint>-<segment>-b<batch>-<crops>-<epochs>ep/`. A run with
  the same dataset, segment, batch and epochs reuses it. A build happens inside the training job, before
  training, and writes `<name>.partial` until it is complete.
- **Disk:** each crop is stored once (`tokens.npy`, memory-mapped by AudioCraft's `TokenPoolLoader`), with its info
  (`infos.pt`) and the draw order (`order.npy`): about 8.5 KB per crop of 20 s, 2.1 GB for a 250-epoch nano run.
  `train run` refuses to start when that would leave under 10 GB free. Pools built before 2026-10-25 store every
  epoch's batches as zips under `cache/` (16× larger) and still work; delete one once nothing will train from it.
- **Refusals:** a pool built for another dataset fingerprint, weights overlay, segment, codec, batch size or fewer
  epochs; `--from` a model that trained on a pool of the same crops (same dataset, segment and seed, even if
  its epochs differ: pass `--pool-seed 1`); and `train resume` past the
  epochs the pool holds. Continue such a model with `train run --from <model>@last`, with or without `--pool`.
- In the GUI: the Atelier's *Crops* section ([gui](gui.md#training)). While a pool builds, the job's state is
  `building`.
- `model show` and `model list` say whether a model trained on fresh crops or a pool.
- Only training reads the pool. Validation and the end-of-run samples still encode audio, as before.

### Every training is a job

A real `train run` records itself in a job folder, `artifacts/train_jobs/<id>/`, never in the dataset:

| File | What |
|---|---|
| `job.json` | what was asked: `train run`'s arguments (or, for a resume, the model and its epoch target) |
| `job_status.json` | `queued` → `starting` → `running` → `completed` / `failed` / `stopped`, or `cancelled`; the process, the model (`xp_sig`) and how it ended |
| `train_plan.json` | the resolved plan: the Dora command, the dataset fingerprint, warm-start details |
| `train_run_status.json` | the Dora process, its exit, failure hints (and `gpu_watchdog_events.jsonl` with `--gpu-watchdog`) |
| `train.log` | everything the job printed, when the queue or the GUI started it |

Runs from before jobs left `logs/train_plan.json` and `logs/train_run_status.json` in their workspace; those stay
where they are and are still read as a fallback (verification).

### Queue trainings

To chain trainings (from an agent, or before going to bed), queue them. `train queue add` takes exactly
`train run`'s arguments, and one worker runs the queue in order, one GPU job at a time, waiting while anything
else trains:

```bash
python -m macchina_ricordi.cli train queue add --dataset bz --recipe reshape-deep --epochs 30 --run-label bz-deep
python -m macchina_ricordi.cli train queue add --dataset bz --recipe reshape-deep --from bz-deep@best --run-label bz2
python -m macchina_ricordi.cli train queue resume bz-deep --epochs 60   # then train bz-deep on, to 60 epochs
python -m macchina_ricordi.cli train queue run          # the worker; returns when the queue is empty
python -m macchina_ricordi.cli train queue list --json  # queued, running and done, with the model each made
python -m macchina_ricordi.cli train queue show <id>    # one job, with the tail of its log
python -m macchina_ricordi.cli train queue cancel <id>  # a queued job is dropped; a running one is stopped
```

- **Checked when added:** the dataset, the weights, and the model `--from` names. A model that does not exist yet
  is accepted when it is the run label of a job queued before (`--from bz-deep@best` above); it is resolved when
  the job starts, so the job fails at once if that earlier job failed.
- **A failed job does not stop the queue.** The worker records it and goes on to the next.
- **The worker:** `queue run` runs in the foreground and exits when nothing is left; `--watch` keeps it waiting
  for new jobs. Only one runs at a time (`add --start` starts one in the background unless one is running; the GUI
  does the same). Jobs run detached: stopping the worker leaves the current job training.
- **Waiting for a run to reach an epoch, then resuming it** (what `train handoff` does) is
  `train queue resume <model> --epochs N`, queued behind it.

The GUI's **Atelier** page (`python -m macchina_ricordi.cli gui`) uses the same queue: preview, add, watch curves and the
log, stop or cancel. See [the GUI](gui.md#training).

## 4. Run The CPU Sidecar

```bash
python -m macchina_ricordi.cli train sidecar --once
```

The sidecar watches checkpoints and runs CPU-only generate/evaluate stages against the newest unseen epoch. When it
can resolve the run's terminal epoch from `.hydra/overrides.yaml`, it skips the terminal `generate` pass to avoid
duplicating the final GPU-side generation.

## 5. Generate Audio

```bash
python -m macchina_ricordi.cli play generate \
  --duration-seconds 20 \
  --num-samples 2
```

Generated audio is a [session](sessions.md):
`artifacts/sessions/YYYY-MM-DD/generate_<run_id>/` (`continue_<run_id>` for a continuation) holds
`session.json`, `takes.jsonl`, `run.json` and the takes under `takes/`. `--output-prefix` adds a label
(`generate_<run_id>_<label>`). `--output-layout dated-run-dir` or `flat` writes the layouts used before October
2026.
Add `--listen` to saved generation runs when you also want JSON and Markdown
listening dossiers next to each generated WAV.

The newer generation surface is organized by mode:

```bash
# Promptless generation
python -m macchina_ricordi.cli play generate \
  --xp-path .dora/xps/<sig> \
  --duration-seconds 20 \
  --num-samples 4

# Audio-prompted continuation
python -m macchina_ricordi.cli play continue \
  --prompt-audio seed.wav \
  --prompt-duration-seconds 2.5 \
  --duration-seconds 30 \
  --num-samples 6
```

For trained Dora runs, point generation at an XP folder:

```bash
python -m macchina_ricordi.cli play generate midwife@best --duration-seconds 30
# the same, spelled the older way:
python -m macchina_ricordi.cli play generate --xp-path .dora/xps/<sig> --checkpoint latest --duration-seconds 30
```

Generation metadata records the model's recipe, read from its `macchina_ricordi_train_metadata.json`; `--recipe`
overrides it. See [Models](models.md) for every way to name a model.

To audition parameter settings without reloading the XP for every try, run a sweep:

```bash
python -m macchina_ricordi.cli play sweep \
  --mode continue \
  --xp-path midwife \
  --checkpoint latest \
  --prompt-audio seed.wav \
  --duration-seconds 30 \
  --num-samples 2 \
  --temperature-values 1.0,1.4,1.8,2.2 \
  --top-k-values 80,250 \
  --cfg-coef-values 1.0,3.0 \
  --seed-values 10,11
```

A sweep is a [session](sessions.md) too: `artifacts/sessions/<date>/sweep_<run_id>[_<prefix>]/` holds
`session.json`, `takes.jsonl`, `sweep_plan.json`, `sweep_report.json`,
`index.csv`, `index.html`, and one `combo_000/` style folder per parameter combination. Add `--score` to include the
existing arrhythmic/percussive score in the index, `--listen` to write listening dossier sidecars and add compact
listening columns to the sweep index, or `--dry-run` to only write the expanded sweep plan.

Useful generation flags:

- `--num-samples` controls how many WAVs are produced (`--batch-size` is kept as an alias).
- `--duration-seconds` controls the final generated sample length.
- `--prompt-start-seconds` and `--prompt-duration-seconds` choose the audio-prompt window for `continue`.
- `--temperature`, `--top-k`, `--top-p`, `--cfg-coef`, and `--use-sampling/--no-use-sampling` control sampling.
- `--min-p` enables confidence-adaptive min-p sampling (replaces top-k/top-p when > 0); meant for high
  temperatures — see [musicgen-creative.md](musicgen-creative.md#min-p).
- `--extend-stride` controls chunk overlap for outputs longer than the trained segment duration; it must be shorter than
  the model segment duration. The default is `8.0`, which is safe for 10s micro-unconditional XPs.
- `--checkpoint latest|final|<epoch>` selects a checkpoint when `--xp-path` is used.
- `--checkpoint-path` can point directly at a `checkpoint.th` or `checkpoint_N.th`. Standalone merged or perturbed
  checkpoints are supported when their directory contains `merged_from.json` or `perturbed_from.json` sidecars that point
  back to a source XP; pass `--xp-path` explicitly if no sidecar is available.
- `--listen/--no-listen` controls optional listening dossier sidecars for saved WAVs. `--listen` cannot be combined
  with `--no-save-output`.

MusicGen `--listen` uses the deterministic listening dossier path only. It does
not run semantic probes or captioner witnesses during generation. For those
slower optional passes, generate the WAVs first and then run `listen index
artifacts/runs --semantic-probes`, `--captioner-witness`, or both.

Beyond one-shot generation: `play stream` is the live instrument loop, with
`play monitor` as its read-only dashboard
([musicgen-stream.md](musicgen-stream.md)); `play duet` renders two models
playing together and `play variations` re-performs a piece over its frozen
structural skeleton ([musicgen-creative.md](musicgen-creative.md)).

## 6. Share Or Receive Models

A trained model leaves this machine as a **portable bundle** — a plain directory (weights in Safetensors,
plus a `config.json`) that another checkout reads directly. There is no hub transfer and no account:
export the folder, move it however you like, and the other side drops it into `models/<name>/` (or points at
the path).

```bash
# Export the raw weights of a@best as a portable bundle
python -m macchina_ricordi.cli model export a@best --bundle --out-dir models --name a
# Alternative weight states within that checkpoint (choose one export, not all three)
python -m macchina_ricordi.cli model export a@best --bundle --out-dir models --name a --weights-state ema
python -m macchina_ricordi.cli model export a@best --bundle --out-dir models --name a --weights-state best

# On the receiving side: a bundle under models/ is discovered automatically; one anywhere else is registered
python -m macchina_ricordi.cli model list
python -m macchina_ricordi.cli model show /somewhere/else/a
```

Then every model command treats it like a trained model — generation, the stream, warm-start
(`--from models/a --dataset your_own`), and `.nrbox` box export. See
[the portable bundle](models.md#the-portable-bundle) for the full workflow, and the retired `model push`/`pull`
and `hf://` note there.

## Notes

- `train run --dry-run` is the fastest way to confirm dataset paths, workspace resolution, and the rendered Dora command.
- Training now adds a deterministic dataset-aware `label=nr_<workspace>_<fingerprint>` by default, so different prepared
  workspaces do not accidentally reuse the same Dora XP signature. Use `--run-label` for a human-chosen signature label.
- Generation metadata JSONs (`--save-output`) now include a `model` block summarising the training provenance of the
  checkpoint used: label, workspace, dataset fingerprint, and init-from-checkpoint lineage (including merge method/alpha
  or perturb sigma/scope/seed when present). This makes every generated WAV traceable back to the exact model and dataset
  that produced it.
- For browsing all models, merges, perturbations, and generations at once, use [the GUI](gui.md); `model catalog` prints the same list in the terminal or as JSON — see [Model Lifecycle](musicgen-admin.md#catalog).
