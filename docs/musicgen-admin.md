# Model Lifecycle

Resuming, deriving, tracing and tidying models. These were the `musicgen admin` commands; they now sit in the
group of the noun they act on (see [the command line](cli.md)): `train` for training a model further, `model`
for deriving and tracing one, `play` for replays, `tidy` for disk.

## Command Summary

| Command | Purpose |
| --- | --- |
| `train resume` | Resume an existing Dora XP in place with a higher epoch target. |
| `train handoff` | Wait for an epoch checkpoint, then launch an in-place resume. |
| `tidy prune` | Enumerate and optionally delete old XP folders under `.dora/xps/`. |
| `tidy clean` | Delete intermediate epoch checkpoints from XPs while keeping the last N. |
| `tidy archive` | Copy matching checkpoints to another filesystem with checksum verification. |
| `model merge` | Interpolate two local checkpoint files into a new warm-start checkpoint. |
| `model perturb` | Add controlled Gaussian noise to a checkpoint for degrade-then-heal experiments. |
| `model export … --bundle` | Snapshot a model as a portable bundle to hand over or host. |
| `train watch` | Watch an XP's `history.json` and emit events when training plateaus. |
| `play replay` | Re-run a saved generation and verify byte-identity against the original audio. |
| `model curve` | Render a per-epoch arrhythmicity score curve from the sidecar's `scores.jsonl`. |
| `tidy backfill` | Retroactively write `macchina_ricordi_train_metadata.json` into existing XP folders. |
| `model lineage` | Trace warm-start, merge, and perturb ancestry for one XP signature. |
| `model catalog` | Browse trained models, merges, and generation runs (terminal or JSON). |

## Resume An Existing XP

```bash
python -m macchina_ricordi.cli train resume <sig> --epochs 5
```

This resumes an existing Dora experiment in place and writes `resume_status.json` into the XP folder.
You can launch this from the main `.venv`; if AudioCraft is not importable there, the command delegates the
actual resume to the repo-managed `.venv-musicgen` runtime. Use `--resume-python` only when you need to
override that runtime.

Before loading the XP, the resume points the shared dataset config
(`.audiocraft/config/dset/audio/<dataset_name>.yaml`) back at the data the XP was trained on, read from its own
`.hydra/config.yaml`. Every workspace prepared under the same dataset name shares that one file, and Dora rebuilds the XP's
config from it, so without this step a resume would train on whichever workspace was trained last. The outcome,
including what the file pointed at before, is written to `dataset_config_status.json` in the XP folder. If the
XP's training data no longer exists, the resume refuses rather than continuing on other data. `train sidecar-cpu`
does the same before each generate or evaluate stage.

## Wait, Then Resume

```bash
python -m macchina_ricordi.cli train handoff \
  <sig> \
  --wait-for-epoch 4 \
  --epochs 8 \
  --active-pid 12345
```

This waits for a target checkpoint to appear, optionally waits for the current PID to stop, then launches
the in-place resume helper. It writes `handoff_resume_status.json` plus a resume stdout log in the XP folder.
Like `train resume`, the watcher can be launched from `.venv` and will delegate to `.venv-musicgen` when needed.

The queue does the same without a watcher process: `train queue resume <model> --epochs 8` waits behind the
running job and then resumes (see [MusicGen](musicgen.md#queue-trainings)). Prefer it; `train handoff` stays
for runs started outside the queue.

## Prune XP Folders

`.dora/xps/` accumulates every training run you launch. Each XP can be tens of
GB. `prune` gives you a safe way to trim them.

```bash
# Dry run (default) — prints a keep/remove table, deletes nothing
python -m macchina_ricordi.cli tidy prune --keep-last 3

# Actually delete everything that did not match any keep rule
python -m macchina_ricordi.cli tidy prune --keep-last 3 --commit
```

The same deletion, limited to XPs you pick and with references shown, is on the GUI's
Réserve page ([housekeeping](gui.md#housekeeping)).

**Without keep markers, the defaults delete nearly everything.** `--keep-last 3` keeps the
three most recently updated XPs and nothing else, so `prune --commit` on a zoo with no `.keep`
files removes every other XP. Read the dry-run table first, and protect what matters with
`.keep` files or `--keep-sig`.

### Keep rules (an XP is kept if *any* applies)

- `--keep-last N` — the N most recently updated XPs (by `checkpoint.th` mtime). Default `N=3`.
- `--keep-labeled` / `--no-keep-labeled` — protect XPs that contain a
  `logs/keep.json` or a top-level `.keep` marker file. Default on. Create one by hand:
  ```bash
  touch /path/to/.dora/xps/<sig>/.keep
  ```
- `--keep-sig <sig>` (repeatable) — explicitly protect a specific signature.
- `--keep-younger-than 7d` — protect XPs whose `checkpoint.th` is newer than
  the given duration (`Nh` / `Nd` / `Nw`).

### Safety knobs

- The default is **dry-run**. Nothing is deleted unless you pass `--commit`.
- XPs a training job is running (or whose `train_run_status.json` says `status == "running"`) are reported as
  `skip_running` and left alone. Override with `--force` if you're sure.
- Every invocation (dry or committed) writes a full JSON audit trail to
  `<dora-dir>/prune_report_<timestamp>.json`. Override the path with
  `--report-path`.

## Clean Intermediate Checkpoints

When an XP finishes training, it often keeps every epoch checkpoint
(`checkpoint_5.th`, `checkpoint_10.th`, ...), each ~2.3 GB. `tidy clean`
removes the older ones while keeping the last N by epoch number.

```bash
# Dry run (default) — prints a keep/remove table, deletes nothing
python -m macchina_ricordi.cli tidy clean --keep-last 2

# Actually delete the older epoch checkpoints
python -m macchina_ricordi.cli tidy clean --keep-last 2 --commit

# Limit to specific XPs
python -m macchina_ricordi.cli tidy clean --sig abc123 --sig def456

# Override running guard
python -m macchina_ricordi.cli tidy clean --force --commit

# Also keep each XP's best epoch checkpoint (lowest valid ce)
python -m macchina_ricordi.cli tidy clean --keep-last 1 --keep-best
```

### What it keeps besides the last N

- **Referenced checkpoints** (on by default; `--no-protect-referenced` to turn off): any
  `checkpoint_N.th` whose path `xps/<sig>/checkpoint_N.th` appears in the project's text
  records outside its own XP folder: a child XP's warm start, a merge or perturbation, a
  generation run, a sweep, a session, a workspace log, or a note under `docs/`. It is a text
  scan of `artifacts/` and `docs/`, plus a `research/` folder when one is present, the
  workspaces (minus their `data/`, `egs/` and `clips/`) and the other XP folders, about a
  second on this repo. Cleanup reports do not count.
- **The best epoch checkpoint** with `--keep-best`: the one whose epoch has the lowest valid
  ce in `history.json`, among the epoch checkpoints on disk. A tie goes to the later epoch.

The table's `protected` column says which checkpoint was kept and why.

The same cleanup, previewed per XP, is on the GUI's Réserve page
([housekeeping](gui.md#housekeeping)).

### What it never touches

- `checkpoint.th` (the final checkpoint) is never a candidate.
- `checkpoint_merged.th` is never a candidate.
- Only files matching `checkpoint_<integer>.th` are considered.

### Safety

- Dry-run by default (`--commit` to apply).
- XPs a training job is running (or with `train_run_status.json` saying `status=running`) are skipped
  (override with `--force`).
- Every invocation writes `checkpoint_cleanup_report.json` into each
  XP folder it scanned, **the dry run included**, so a dry run over the whole zoo touches every
  XP folder. Limit it with `--sig`, or preview from the GUI, whose preview writes nothing and
  whose commit reports only in the XPs that lost a file.

## Archive Checkpoints To Another HDD

`tidy archive` copies matching checkpoints into immutable, versioned
bundles on another filesystem. It never deletes or modifies the source XP.
The command is a dry-run unless `--commit` is supplied.

```bash
# Inspect matching latest checkpoints; copy nothing
python -m macchina_ricordi.cli tidy archive \
  --destination /media/you/archive-disk/macchina_ricordi-checkpoints \
  --label euro_voices \
  --profile micro-unconditional \
  --min-age 1h \
  --require-metadata

# Apply the same plan
python -m macchina_ricordi.cli tidy archive \
  --destination /media/you/archive-disk/macchina_ricordi-checkpoints \
  --label euro_voices \
  --profile micro-unconditional \
  --min-age 1h \
  --require-metadata \
  --commit
```

The destination's nearest existing parent must already be mounted. By
default, the command refuses a destination on the same filesystem as the
Dora directory, which protects against accidentally archiving to the system
disk after an HDD is unmounted. Use `--allow-same-filesystem` only when that
guard is intentional.

### Selection rules

- `--checkpoint latest` considers `checkpoint.th` (the default).
- `--checkpoint epochs` considers `checkpoint_N.th` files.
- `--checkpoint all` considers both kinds.
- Repeated `--label`, `--profile`, and `--sig` values are ORed within their
  own field; different fields are ANDed.
- `--min-age 1h` uses the checkpoint modification time.
- XPs reporting `status == running` are always skipped.
- `--require-metadata` skips XPs without valid
  `macchina_ricordi_train_metadata.json`.

The GUI's Réserve page runs the same archive, with a preview table and progress
([housekeeping](gui.md#housekeeping)); it never lifts the same-filesystem guard.

Each archived bundle contains the checkpoint, its SHA-256 manifest, and the
available identity/config companions (`macchina_ricordi_train_metadata.json`,
`history.json`, `.hydra/`, and related small configuration files). Bundles
are stored below `<destination>/<sig>/versions/<sha256-prefix>/`, so a later
checkpoint update creates a new version instead of overwriting the old one.
An `archive_index.json` makes repeated timer runs idempotent. Every invocation
also writes a JSON audit report under the Dora directory; use `--report-path`
to relocate it.

### Run automatically with systemd

Example user-service and timer templates live in `systemd/`. Copy them to
`~/.config/systemd/user/`, create the referenced environment file, edit the
paths for this machine, then enable the timer:

```bash
mkdir -p ~/.config/systemd/user ~/.config/macchina_ricordi
cp systemd/macchina-ricordi-checkpoint-archive.service.example ~/.config/systemd/user/macchina-ricordi-checkpoint-archive.service
cp systemd/macchina-ricordi-checkpoint-archive.timer.example ~/.config/systemd/user/macchina-ricordi-checkpoint-archive.timer
cp systemd/macchina-ricordi-checkpoint-archive.env.example ~/.config/macchina_ricordi/checkpoint-archive.env
editor ~/.config/macchina_ricordi/checkpoint-archive.env
systemctl --user daemon-reload
systemctl --user enable --now macchina-ricordi-checkpoint-archive.timer
```

The timer runs every 15 minutes and is harmless when there are no new
matches. If the HDD is not mounted, the service fails without creating a
fallback directory on the system filesystem.

## Merge Checkpoints

`merge` blends two local AudioCraft checkpoint files into a third checkpoint.
It is meant for model-soup / SLERP exploration: warm-start a new run from the
blend, or generate from it with a compatible XP config.

**Safety warning:** only use trusted checkpoint files with this command.
PyTorch checkpoint loading can execute arbitrary code when reading untrusted
inputs, so do not run `model merge` on downloaded `.th` files unless
you trust their source.

```bash
python -m macchina_ricordi.cli model merge \
  /path/to/a/checkpoint.th \
  /path/to/b/checkpoint.th \
  /path/to/merged/checkpoint.th \
  --alpha 0.5 \
  --method linear
```

The GUI runs this from a form, with a shape check before anything loads: see [the GUI](gui.md#merge-and-perturb).

### What it writes

- `/path/to/merged/checkpoint.th` - merged weight roots only.
- `/path/to/merged/merged_from.json` - lineage, source hashes, output hash,
  merged roots, tensor counts, and warnings.

The merge touches only compatible weight roots present in both inputs:
`model`, `best_state.model`, and `fsdp_best_state.model`. Optimizer,
scheduler, and scaler state are intentionally left out, so the result is not
an in-place training resume checkpoint.

### Useful flags

- `--alpha` - interpolation amount. `0` keeps checkpoint A, `1` keeps checkpoint B.
- `--method linear|slerp` - linear averaging by default; SLERP is available for
  spherical interpolation.
- `--dry-run` - validate paths and print the helper command without writing.
- `--force` - overwrite the destination checkpoint if it already exists.
- `--musicgen-python` - use a specific torch-capable runtime instead of the
  repo-managed `.venv-musicgen`.

## Perturb Checkpoints

`perturb` copies a local AudioCraft checkpoint into a new warm-start checkpoint
while adding reproducible Gaussian noise to selected floating tensors. It is
meant for deliberate damage: perturb a model, warm-start from the perturbed
checkpoint, then let training heal or mutate the result.

**Safety warning:** only use trusted checkpoint files with this command.
PyTorch checkpoint loading can execute arbitrary code when reading untrusted
inputs, so do not run `model perturb` on downloaded `.th` files unless
you trust their source.

```bash
python -m macchina_ricordi.cli model perturb \
  /path/to/source/checkpoint.th \
  /path/to/perturbed/checkpoint.th \
  --sigma 0.02 \
  --layer-scope attention-projections \
  --seed 1234
```

### What it writes

- `/path/to/perturbed/checkpoint.th` - perturbed weight roots only.
- `/path/to/perturbed/perturbed_from.json` - lineage, source hash, output hash,
  sigma, layer scope, seed, perturbed roots, tensor counts, and warnings.

Like `merge`, the perturb command touches only warm-startable weight roots:
`model`, `best_state.model`, and `fsdp_best_state.model`. Optimizer,
scheduler, and scaler state are intentionally left out, so the result is not
an in-place training resume checkpoint.

The source may be an XP reference or a **portable bundle** ([Models](models.md#the-portable-bundle)). When the
destination is a bundle directory, the output is a new portable bundle that keeps the source's configuration
and records its parent (the source's digest) in its own `config.json`; the child is self-contained, so it
never needs the parent path around to warm-start from it.

### Useful flags

- `--sigma` - required Gaussian noise scale. Start small; `0.005` to `0.02`
  is a gentle range for exploration.
- `--layer-scope all|attention|attention-projections|lm-head` - select a
  built-in tensor subset. Default is `all`.
- `--layer-scope contains:<fragment>` - target checkpoint keys containing a
  custom fragment when the built-ins are too broad.
- `--seed` - reproducible noise seed. Default is `0`.
- `--dry-run` - validate paths and print the helper command without writing.
- `--force` - overwrite the destination checkpoint if it already exists.
- `--musicgen-python` - use a specific torch-capable runtime instead of the
  repo-managed `.venv-musicgen`.

## Watch For Plateau

Advisory loss-plateau detector. Runs as a standalone watcher process (pair it
with a separate `train run` terminal, or chain it after `train handoff`).

```bash
# Emit events but never act
python -m macchina_ricordi.cli train watch <sig>

# Stop the training process group if no improvement for 4 consecutive epochs
python -m macchina_ricordi.cli train watch <sig> \
  --metric valid.loss \
  --patience-epochs 4 \
  --min-improvement 1e-4 \
  --action stop \
  --stop-pid <train-pid>
```

### What it does

- Polls `<xp>/history.json` at `--poll-seconds` (default: 60s).
- Tracks the best value of `--metric` (dotted path, default `valid.loss`).
- Reports `plateau_detected` when `epochs_since_best >= --patience-epochs`.
- On `--action=stop`, sends `SIGTERM` to the process group identified by
  `--stop-pid`, waits `--grace-seconds`, then falls back to `SIGKILL` if still
  alive.

### Useful flags

- `--once` — take a single sample, write status, and exit. Useful in scripts.
- `--events-path` / `--status-path` — relocate the JSONL/JSON sidecars outside
  the XP folder.
- `--min-improvement` — change the sensitivity for what counts as progress.

Every run writes `<xp>/plateau_events.jsonl` (append-only) and
`<xp>/plateau_watch_status.json` (latest snapshot).

## Audit Reproducibility

Given a generate or continue [session](sessions.md) (its id or folder) or its `run.json`, `replay`
re-runs the exact same request and byte-compares each output WAV against the
originals.

```bash
python -m macchina_ricordi.cli play replay generate_20261006T090503Z
python -m macchina_ricordi.cli play replay artifacts/runs/2026-04-22/musicgen_20260422T100000Z/run.json
```

Streams, duets and variations cannot be replayed (a stream round depends on what played before it), and
neither can sweep takes: the sweep helper runs the model in fp32 and replay runs the generate engine
(fp16, CUDA graphs), so their bytes would never match.

### What it does

1. Loads `run_id`, `backend`, `model_id`, `device`, `sample_rate`, and the full
   `request` (prompts, batch_size, duration_seconds, max_new_tokens, seed).
2. Rebuilds a backend with matching overrides and replays the generation.
3. Saves the replay run under `artifacts/replays/replay_<run_id>_<ts>/` using
   the same `save_generation_run` path your original `generate` used.
4. Hashes every source WAV and every replay WAV, reports per-sample match or
   MISMATCH, and writes `replay_report.json` under the replay directory.

### Useful flags

- `--output-dir <dir>` — change the replay root (default: `artifacts/replays`).
- `--strict` — exit with code `1` on any mismatch (handy for CI reproducibility
  checks before publishing a checkpoint).
- `--json` — emit a machine-readable summary instead of the plain-text table.

### Notes

- Replay requires all source WAVs referenced by the metadata to still exist.
- `replay_report.json` carries full SHA-256s for every source and replay file,
  so you can diff across runs later.

## Score Curve

Once the CPU sidecar has scored at least one epoch's worth of generated
samples, `model curve` prints how arrhythmic/percussive your model is
becoming over training time.

```bash
# Median score per epoch + Unicode sparkline
python -m macchina_ricordi.cli model curve <sig>

# Mean of a different RewardComponents field, machine-readable
python -m macchina_ricordi.cli model curve <sig> \
  --metric onset_density --aggregate mean --json
```

### Where the data comes from

`train sidecar` runs CPU `generate` stages against each new
epoch checkpoint. With `--score-samples` (default on), each newly produced
sample under `<xp>/samples/<epoch>/*.wav` is scored with
`macchina_ricordi.scoring.reward.score_arrhythmic_percussive` and appended as one
row to `<xp>/sidecar_cpu/scores.jsonl`. The renderer is a pure offline
reader of that JSONL.

### Default text output

```
scores_path: /.../<sig>/sidecar_cpu/scores.jsonl
sig: <sig>
metric: score
aggregate: median
samples_total: 24 (failed: 0)

 epoch    n    median       min       max
     3    6    0.6120    0.4010    0.7880
     4    6    0.6740    0.5020    0.8120
     5    6    0.7010    0.5180    0.8440
     6    6    0.7380    0.6010    0.8790

▁▃▅█
```

### Useful flags

- `--metric <field>` — any `RewardComponents` field (default `score`). Lets
  you chart e.g. `onset_density`, `tempo_instability`,
  `percussive_onset_strength_p90`.
- `--aggregate mean|median|p90|max|min` — how to collapse the per-sample
  values for each epoch (default `median`).
- `--scores-path <path>` — read a JSONL from somewhere other than
  `<xp>/sidecar_cpu/scores.jsonl`.
- `--json` — emit a stable machine-readable summary; useful for piping into
  notebooks or CI checks.
- `--no-sparkline` — drop the trailing bar string when piping into `less`
  or terminals without UTF-8.

### Opting out at scoring time

If you want to disable scoring for a particular sidecar run (e.g. running
the sidecar against a foreign environment that lacks `librosa`), pass
`--no-score-samples`:

```bash
python -m macchina_ricordi.cli train sidecar --once --no-score-samples
```

Existing rows in `scores.jsonl` are preserved; subsequent runs with scoring
re-enabled will only score sample IDs not already present.

## Backfill XP Metadata

Newer training runs automatically write a `macchina_ricordi_train_metadata.json` into
the XP folder capturing the dataset fingerprint, effective label, command, and
warm-start lineage. Older XPs predate this file. `tidy backfill` reconstructs
it from the Hydra config and overrides that Dora baked at training time.

```bash
# Dry-run (default) — prints a table of what would be done
python -m macchina_ricordi.cli tidy backfill

# Write metadata files for every XP that lacks one
python -m macchina_ricordi.cli tidy backfill --commit

# Overwrite existing metadata (e.g. after fixing a stale datasource path)
python -m macchina_ricordi.cli tidy backfill --commit --force
```

### What it reconstructs

For each XP under `.dora/xps/<sig>/`:

- Reads `.hydra/overrides.yaml` for the effective label, `continue_from`, and
  training overrides.
- Reads `.hydra/config.yaml` for the resolved datasource paths, then walks up to
  infer the workspace directory.
- If `continue_from` points to a local merged or perturbed checkpoint, loads
  `merged_from.json` / `perturbed_from.json` sidecars and inlines lineage
  details (method/alpha or sigma/scope/seed plus source paths and SHA256s).
- If the workspace still exists, loads its `dataset_fingerprint.json` for
  train/valid counts and aggregate hashes.
- Assembles a `macchina_ricordi_train_metadata.json` matching the format written by
  current training launches.

### Limitations

The resolved `config.yaml` reflects the *current* state of the shared Dora
dataset config, not necessarily what it was at training time. If the config was
changed between runs, the inferred workspace may be wrong. Fix the generated
metadata file by hand or pass `--force` to regenerate after inspection.

## Lineage

`model lineage` traces the ancestry of one Dora XP signature using the metadata that
training, merge, perturb, and backfill commands already write.

```bash
# Compact terminal tree
python -m macchina_ricordi.cli model lineage <sig>

# Stable graph JSON for scripts
python -m macchina_ricordi.cli model lineage <sig> --json
```

The command is read-only. It walks backwards through the same provenance every
other view reads ([the zoo](models.md#the-zoo-one-graph)): warm-start
`init_from_checkpoint` metadata, the `merged_from` / `perturbed_from` sidecars
(the newest operation first when a record holds both), epoch checkpoints like
`checkpoint_150.th` resolved back to their XP, and parent/child override diffs
for known XP-to-XP warm-starts. Missing ancestors are shown as unknown nodes
with warnings instead of aborting. `model show` lists the other direction:
children and descendants.

V1 is deliberately sig-first: the target must be an XP signature under
`.dora/xps/`. Starting directly from an arbitrary checkpoint path or a standalone
merge/perturb directory is future work.

## Catalog

`catalog` scans the project for trained models, merged checkpoints, generation
runs, sweep summaries, stream sessions, model portraits, and the Git-tracked
experiment ledger, then lists them in the terminal or as JSON. For browsing, use
[the GUI](gui.md), which reads the same data.

```bash
# Plain-text table (terminal)
python -m macchina_ricordi.cli model catalog

# Filter by workspace name
python -m macchina_ricordi.cli model catalog --dataset midwife

# Machine-readable JSON
python -m macchina_ricordi.cli model catalog --json

# Show existing listening summaries from generated WAV sidecars
python -m macchina_ricordi.cli model catalog --listen
```

The catalog's models, parents and experiments are the zoo's
([models.md](models.md#the-zoo-one-graph)); a model with a name is listed by it.
Generation runs, sweeps, streams, and the other sessions (duets, variations) are
the sessions `session list` shows ([sessions](sessions.md)), with their listening
reports read beside each take. The merge and perturbation files are the zoo's
derived checkpoints. `catalog --json`
also includes the canonical lineage graph for each model when it can be built.
Generation metadata now also includes an inline `model` block summarising the
training provenance of the checkpoint used.
`catalog --listen` is read-only: it surfaces `*_listening_report.json` sidecars
that already exist next to generated WAVs, and it never backfills missing
listening reports.
The ledger is parsed with the standard-library TOML reader. Malformed entries
produce `ledger_warnings` in JSON and a terminal warning section while the rest
of the catalog still loads. Catalog scans read metadata, CSVs, JSONL logs, and
Markdown; they do not load checkpoint tensors or audio.
If a sidecar contains optional `semantic_probes` or `captioner_witness` fields,
catalog treats them as part of the existing listening report. To create those
fields, run `listen index ... --semantic-probes` or `listen index ...
--captioner-witness` before opening the catalog.

## Artifact Layout

Everything `play` makes is a [session](sessions.md) under
`artifacts/sessions/<date>/<kind>_<run id>/`; sessions from before October 2026
stay in `artifacts/runs`, `sweeps`, `streams`, `duets` and `variations`. Derived checkpoints live under
`artifacts/checkpoints/merged` or `artifacts/checkpoints/perturbed`; the older
`merged_checkpoints/` stays where it is, as do `.dora/xps/`, the prepared workspaces
and `training_datasets/` (AGENTS.md, "The repo root", maps every folder). (The one-time `artifacts migrate`
command that moved older outputs into this layout was removed on 2026-10-05.)

## When To Use What

- `train resume` — directly extend a completed or interrupted run on the same dataset.
- `train handoff` — staged handoff where a current run must finish an epoch first.
- `prune` — reclaim disk after a long experimentation streak.
- `tidy clean` — after a long training run completes, reclaim disk by
  removing old epoch snapshots while keeping the final checkpoint and the last
  few epochs.
- `merge` — blend compatible checkpoint weights into a new artistic starting point.
- `perturb` — add reproducible checkpoint noise before a degrade-then-heal
  warm-start run.
- `train watch` — unattended training: get a signal (or a clean stop) when
  progress flatlines.
- `replay` — before publishing or sharing a checkpoint, confirm its
  generations reproduce exactly.
- `model curve` — track how arrhythmic/percussive a run is becoming over
  epochs without leaving the terminal.
- `tidy backfill` — after upgrading macchina_ricordi or backfilling old experiments,
  write `macchina_ricordi_train_metadata.json` into every XP folder so lineage and
  dataset provenance are self-describing.
- `model lineage` — inspect the ancestry of one XP signature from the terminal or as
  JSON.
- `catalog` — get an overview of all models, merges, perturbations, and
  generations; use the HTML dashboard for visual exploration or `--json` for
  scripting.

## Useful Global Flags

- `--resume-python` lets you choose the interpreter used for the actual resume command; by default the
  repo-managed `.venv-musicgen` runtime is used when delegation is needed.
- `--resume-log-path` writes the spawned resume stdout/stderr somewhere explicit.
- `--status-path` stores watcher state outside the XP folder if needed.
- `--grace-seconds` and `--poll-seconds` tune how conservative the watchers are.
