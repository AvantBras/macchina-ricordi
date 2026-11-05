# Reusing An Existing Checkpoint For New Training

This note documents the **warm-start from snapshot** workflow: taking an existing trained model —
another XP's checkpoint, a derived checkpoint, or a portable bundle you received — and launching a
new training run from it, on a dataset you name. It is intentionally separate from the "resume the
same XP in place" workflow.

Two concepts coexist in this repo:

- **resume**: continue the same Dora experiment (same dataset, same XP signature)
  using `train resume` / `train handoff`.
- **warm-start / init from checkpoint**: start a *new* Dora experiment (new XP
  signature, new dataset) that loads old weights as initialization. This is what
  `train run --from` (alias `--init-from-checkpoint`) provides.

## TL;DR

```bash
# Warm-start from a running XP's checkpoint, onto a dataset you prepared
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset my_own \
  --from midwife@best

# Or from a locally merged checkpoint file
python -m macchina_ricordi.cli model merge \
  /path/to/a/checkpoint.th \
  /path/to/b/checkpoint.th \
  /path/to/merged/checkpoint.th
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset my_own \
  --from /path/to/merged/checkpoint.th

# Or from a portable bundle you downloaded and extracted into models/
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset my_own \
  --from models/comelady
```

A warm start is always a **new model learning**: the snapshot becomes initial weights with a fresh
optimizer and schedule, on the dataset you name with `--dataset`. No source corpus beyond the snapshot is
required — a bundle is self-contained, so the parent XP's files are never needed.

## What `--from` Does

1. Resolves the source snapshot:
   - a **model reference** (`midwife@best`, a sig, a label) — the newest checkpoint unless you
     pick one with `@`;
   - a **local checkpoint path** (`…/checkpoint.th`, or a merged/perturbed output file);
   - a **portable bundle directory** (`models/comelady`, or its path anywhere). The bundle carries
     its own weights and configuration, so its parent's files do not matter.
   The old `hf://` checkpoint URI is gone; model transfer is now the portable bundle
   ([Models](models.md#export)), moved by hand.
2. Strips any pre-existing `continue_from=...` entry from the selected recipe's
   training overrides and appends `continue_from=<resolved-local-path>`. No
   current recipe sets a default `continue_from`, so the entry is simply
   injected; one passed with `--override` is refused (pick one source).
3. Records a full source-metadata block inside the job's `train_plan.json` (`artifacts/train_jobs/<id>/`) under `init_from_checkpoint`:
   - `origin`: `local` or `bundle`
   - `user_input`: raw CLI argument
   - `resolved_local_path`: on-disk checkpoint path passed to Dora
   - `source_resume_metadata`: slim summary of the source's `resume-metadata`
     bundle if one was present (dataset fingerprint, environment, training overrides),
     plus `merged_from` / `perturbed_from` lineage when the checkpoint has local
     sidecars.
   - `dataset_fingerprint_matches_current_workspace`: `true`, `false`, or `null`
     (when no comparison was possible).
   - `warnings`: list of human-readable advisories.
4. When the source is a non-pretrained legacy `.th` checkpoint (for example
   `--from /.../checkpoint.th`), the wrapper sets
   `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for the training subprocess so legacy
   AudioCraft checkpoints remain loadable with modern PyTorch defaults. Only
   ever point at `.th` files you trust — pickled checkpoints can execute code.
   A portable bundle poses no such risk: its weights are Safetensors, not
   pickles, and the importer reads weights and configuration without executing
   anything bundled.
5. Dora computes a fresh signature because the overrides changed, so a new XP
   folder is created under `.dora/xps/<new-sig>/`. The source checkpoint folder
   is never touched.

## Verification Policy (Soft)

Warm-start inherently allows dataset changes, so mismatches **do not block**.
They are surfaced as warnings in stdout and persisted into `train_plan.json`:

- Dataset fingerprint mismatch between the source checkpoint (when its
  `resume-metadata/dataset_fingerprint.json` is available) and the current
  workspace. Typical and expected when you intentionally switch datasets.
- Architecture: a warm-started run has to match the recipe you train with;
  `model show <ref>` shows a model's shape first.
- Missing source resume-metadata: a note that compatibility could not be
  inspected at all.

## Workspace Policy

A recipe never names data: pass `--dataset` (your own built or prepared corpus,
or `models/<name>` is only ever the *source*, never the training data). For
warm-starts on a different dataset, the recommended artistic flow is:

1. Build or prepare the dataset you want to explore, `data build my_own …`.
2. Pass that dataset to `train run` alongside `--from`.

The source XP folder (under `.dora/xps/<src-sig>/`) is never modified. Dora
automatically produces a new XP signature for the warm-started run because the
`continue_from` override changes.

## Mutual Exclusion

`--from` cannot be combined with a user-supplied
`--override continue_from=...`. Pick one. If you want full manual control over
the override, drop `--from` entirely and pass
`--override continue_from=<path>` yourself (you lose the metadata trail).

## Merged And Perturbed Checkpoints

`model merge` creates a local checkpoint from two compatible source
checkpoints. The merged file contains only model-weight roots, not optimizer or
scheduler state, so use it as a warm-start source rather than an in-place resume
target. It writes `merged_from.json` next to the output checkpoint. When that
merged checkpoint is passed to `--from`, the lineage sidecar is
included in the job's `train_plan.json` under
`init_from_checkpoint.source_resume_metadata.merged_from`.

`model perturb` rewrites one source into a perturbed copy, adding reproducible
Gaussian noise:

```bash
python -m macchina_ricordi.cli model perturb midwife models/perturbed-midwife --sigma 0.05 --seed 1234
```

It writes `perturbed_from.json` with sigma, layer scope, seed, source
hash, output hash, and perturbed tensor counts. The source may be an XP
reference or a portable bundle. When the destination is a bundle directory, the
output is a **new portable bundle** that keeps the source's configuration and
records its parent (the source's digest) in its own `config.json`; the child is
self-contained, so it never needs the parent path around to be used or
warm-started.

Warm-starting from a derived checkpoint or a perturbed bundle inlines its
lineage under `init_from_checkpoint.source_resume_metadata`. Since newer
versions, the warm-start lineage is also written into the XP folder itself as
`macchina_ricordi_train_metadata.json`. This file is the single
source of truth for a model's identity: its dataset, label, and full init-from
chain (including merge method/alpha or perturb sigma/scope/seed and source SHA256s).

## Reading The Warm-Start Metadata Trail

After a launch, inspect (a dry run prints the same and records nothing):

```bash
# The XP-level metadata (persistent, never overwritten)
cat .dora/xps/<sig>/macchina_ricordi_train_metadata.json

# The plan of the training job that made it
python -m macchina_ricordi.cli train queue list --json        # find the job
cat artifacts/train_jobs/<id>/train_plan.json
```

Useful keys in `macchina_ricordi_train_metadata.json`:

- `command` includes the final Dora invocation, with a single
  `continue_from=<resolved-local-path>` entry.
- `init_from_checkpoint.warnings` shows anything that looked off.
- `init_from_checkpoint.source_resume_metadata` shows what the source run looked
  like (its dataset fingerprint, its training overrides, its environment).
- `init_from_checkpoint.dataset_fingerprint_matches_current_workspace` is
  `false` for every realistic "new dataset" warm-start, `true` if you happen to
  reuse the same prepared data, and `null` if no comparison was possible.

## Relationship To Existing Commands

| Intent | Command |
| --- | --- |
| Continue the same XP with more epochs | `train resume <sig> --epochs N` |
| Wait for an epoch then continue the same XP | `train handoff ...` |
| Blend two local checkpoints | `model merge <a> <b> <out>` |
| Add controlled noise to a local checkpoint or bundle | `model perturb <src> <out> --sigma N` |
| Snapshot a model as a portable bundle to hand over | `model export <ref> --bundle --out-dir models --name <name>` |
| Inspect a bundle placed outside `models/` | `model show <path>` |
| Start a new run from old weights (same or new dataset) | `train run --from … --dataset …` |
| Backfill metadata for older XPs that predate the metadata file | `tidy backfill --commit` |
| Browse all models, merges, perturbations, and generations | [the GUI](gui.md), or `model catalog` in a terminal |
