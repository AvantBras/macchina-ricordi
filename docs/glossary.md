# Glossary

The project uses six nouns. Use them when you talk, write notes, or design a
feature. The code is still catching up, so each entry also says what the thing
is called in the CLI and on disk **today**. When this page and the code
disagree on a *name*, the code is right about what to type and this page is
right about what to call it.

```
source ──build──▶ dataset ──┐
                            ├──train──▶ model ──play──▶ session (takes) ──▶ judgement
                  recipe ───┘     ▲
                                  └── from: another model (warm start, merge, perturb)
```

The one sentence to keep: **a model is a recipe trained on a dataset**,
optionally starting from another model.

---

## Source

Audio you collected. Albums, folders, recordings. The tool reads sources and
never changes them.

- **Where:** your own audio folders, conventionally under `library/`. The GUI can
  also discover `examples/sample_data/` and `training_datasets/` when those legacy
  roots exist. No source audio is included in this checkout.
- **Not:** a dataset. A source has no split, no fixed clip length, and no
  fingerprint.

## Dataset

A named set of clips, split into train and valid, that a model can be trained
on. It owns everything about the *data*: which sources it came from, how they
were cut (slice length), the split, its fingerprint, and the weightings saved
for it.

- **Where:** `datasets/<name>/`, built once by `data build <name>
  <sources…>` (or the GUI's **+ New corpus**): `clips/` + `corpus.json` (where
  every clip came from), the split in `data/train` and `data/valid` (hard links),
  AudioCraft's manifests in `egs/<dataset_name>/`, and `dataset.json` describing
  the build. `data list` / `data show <name>` read it.
- **Never rebuilt:** a different cut is a new dataset under a new name.
- **Legacy workspaces:** older `.workspace_<name>/` and supported legacy roots can
  still be read as datasets. Their name keeps its leading dot
  (`--dataset .workspace_x`). The two-step `data corpus` / `data prepare` workflow
  remains available, but no legacy workspace is supplied here.
- **Training records:** a training records itself in its job's folder (see
  *Model*, below), not in the dataset. Older workspaces may also contain
  `logs/train_plan.json` and `logs/train_run_status.json` from legacy launches.
- **Identity:** `logs/dataset_fingerprint.json` (hashes of every file in the
  split). Its digest goes into a model's automatic label.
- **Weights:** per-clip sampling weights are an *overlay* on a dataset, saved
  under a name (`data rebalance --save-as cov_r1`, or `data mix --save-as a3_b1`
  for shares of draws per source folder) in
  `artifacts/weights/<workspace>/<name>/`. The dataset itself never changes; a
  run uses weights only when given `--weights <name>`, and records which.
- **Frozen means frozen:** `data prepare` refuses an already prepared
  workspace (it would wipe `data/`, `egs/` and `logs/`) unless `--overwrite`.
  Back up any unique audio before allowing an overwrite.
- **Not:** `dataset_name` (`macchina_ricordi_micro_uncond`). It is the folder name under
  a workspace's `egs/` and the name of a single AudioCraft config file,
  `.audiocraft/config/dset/audio/<dataset_name>.yaml`, which every workspace
  shares and which is rewritten before each launch or resume. It's plumbing, not
  a dataset. Avoid calling it one.

## Recipe

How to train: the model's shape (width, depth, heads, codec) and the schedule
(epochs, learning rate, batch size, checkpoint cadence). A recipe says nothing
about data.

- **Where:** `[tool.macchina_ricordi.recipes.<name>]` in `musicgen/pyproject.toml`,
  chosen with `--recipe`. A recipe can `extend` another and name a codec from
  `[tool.macchina_ricordi.codecs]`; `abstract` ones exist only to be extended.
  `--override key=value` adjusts a recipe for one run. The default is
  `reshape-nano`. See [MusicGen](musicgen.md#recipes).
- **Never data:** a recipe that sets `dset=` is refused, and `train run` and
  `data rebalance` require `--dataset` (alias `--workspace-dir`).
- **Segment length** (`dataset.segment_duration`) is part of the recipe (it is
  the horizon the model learns) but has to fit the dataset's slice length. The
  training planner refuses a mismatch (`--allow-segment-padding` overrides it).
- **Code still says profile** in places: `--profile` is an alias of `--recipe`,
  and Python parameters are named `profile_name`. Training metadata records
  both `recipe` and `profile`.

## Model

What training produces: one recipe trained on one dataset, possibly starting
from another model. A model has checkpoints, one per saved epoch.

- **Today:** a Dora XP, `.dora/xps/<sig>/`, where the *sig* is 8 hex characters
  and the model's ID. It has a *label* (`effective_label`; automatic labels look
  like `nr_<workspace>_<fingerprint digest>`), and optionally a short *name*
  from `musicgen/models.toml` (`model name 4ae63477 midwife`).
- **Its files:** `checkpoint.th` (latest/final weights), `checkpoint_<N>.th` (an
  epoch), `history.json` (curves), `macchina_ricordi_train_metadata.json` (recipe,
  workspace, fingerprint, parent), `.hydra/config.yaml` (the resolved config,
  including the data paths it trained on).
- **Relations between models:**
  - *resume*: the same model trains further (`train resume`). Same sig.
  - *warm start*: a new model starts from another's checkpoint
    (`train run --from <model>`). New sig; the parent is recorded. The source
    may also be a [portable bundle](models.md#the-portable-bundle); a run from one
    is a new model with a fresh optimizer and schedule, and needs no source corpus.
  - *merge* and *perturb* make a **checkpoint file**
    (`artifacts/checkpoints/merged|perturbed/<name>/`), not a model. It becomes
    the parent of a model when a run warm-starts from it. Perturbing *into* a
    bundle directory makes a new portable bundle that keeps the source's
    configuration and records its parent digest.
- **One record:** `model show <model>` gathers what training wrote and what
  `musicgen/models.toml` adds (name, title, notes, measured behaviour such as
  `clean_share`). That file is the only place a person writes facts about a
  model; `MODEL_SPECS` in `macchina_ricordi/models/specs.py` is read from it.
  Listening notes and experiment entries are separate annotations.
  `appliance/spec.py` still has its own class called `ModelSpec`: the
  architecture a box needs.
- **How one gets made:** a *training job*, `artifacts/train_jobs/<id>/`. Every
  real `train run` is one, and `train queue add` lines more up for a single
  worker to run in order (the GUI's Atelier uses the same queue). The job holds
  the request, its state, the resolved plan (`train_plan.json`), the process
  record (`train_run_status.json`) and the log. A job is plumbing, not a
  seventh noun: it is how a dataset and a recipe become a model. See
  [MusicGen](musicgen.md#queue-trainings).
- **Pointing at one:** a name, sig (or prefix), label or path, plus `@best`,
  `@last`, `@final` or `@<epoch>` for a checkpoint. For example, `deep@60` means
  epoch 60 of a model you have named `deep`. See
  [Models](models.md).
- **Exported:** a `.nrbox` (built from an `.nrsrc` export) is a model packed for
  the appliance, not a separate kind of model. A **portable bundle** (`model.safetensors`
  plus a `config.json`, schema `macchina_ricordi.model_bundle.v1`) is the directory form
  for hosting, moving and warm-starting from a model — see [Models](models.md#the-portable-bundle).

## Session

One occasion of making audio from one or more models. A session is a set of
**takes** (single audio files) plus the controls used for each.

- **Kinds:** *generate* and *continue* (one batch), *sweep* (a grid of settings),
  *stream* (one take per round), *duet* (two models: a mix and two voices),
  *variations* (re-performances of a parent take).
- **On disk:** `artifacts/sessions/<date>/<kind>_<run id>/` with `session.json`
  and its takes. Sessions from before October 2026 stay in `artifacts/runs/`,
  `sweeps/`, `streams/`, `duets/` and `variations/` and are read the same way.
- **Pointing at one:** `session list` / `session show <id>`; a take is
  `<session>#<n>`, counting from 0. See [Sessions](sessions.md).
- **About a take** you can have a *score* (a number computed from the audio:
  arrhythmic, arc, structure, novelty) and a *description* (text from a
  listening dossier or a witness). Both are readings of a take. They are not
  judgements.

## Judgement

A verdict about takes, given by you or by an AI witness. Human judgements can
serve as reference observations when checking scores. AI-witness ratings are
fallible automated proxies, not validated ground truth.

- **Human:** blind A/B pairs (`judge ab`, the GUI's **Écoute**) go to
  `artifacts/preferences/preferences.jsonl`. Single-clip labels (`judge label`)
  go to `artifacts/preferences/labels.jsonl`.
- **Witness:** a captioner rating (`judge witness`) goes to `artifacts/preferences/witness.jsonl`. Its free-form
  answers (`listen ask`) are text, a description, not a judgement.
- **Impression:** how playing a model felt, sighted and on purpose: a 1–5 gut rating, a note, or both, optionally
  tied to a session (the GUI's model page). Goes to `artifacts/preferences/impressions.jsonl`; never evidence
  for a score, only a diary.
- `judge log` lists every judgement of every kind; `judge prefs` reports which scores agree with yours.
- Code: `macchina_ricordi/judging/`. Scores are in `macchina_ricordi/scoring/`, descriptions in `macchina_ricordi/listening/`.

---

## Annotations

- **Experiment:** a human-written entry in the GUI's experiment ledger (question,
  status, verdict, member models). The ledger is created when you save your own
  experiments; it groups models and is not a step in the pipeline. See [the GUI](gui.md).
- **Portrait:** a dated listening note on what a model sounds like. This is a kind
  of annotation, not a required folder or a supplied set of studies.

## Words to use carefully

| Word | Say instead / meaning |
|---|---|
| profile | **recipe**; profiles became recipes on 2026-10-05 (`--profile` still works) |
| workspace, corpus | **dataset**; *workspace* is a legacy one, *corpus* the clip-gathering half of a build |
| `dataset_name` | the shared AudioCraft config slot; not a dataset |
| XP, sig | **model**, and its ID |
| checkpoint | one saved epoch of a model, or a merge/perturb output file |
| run | ambiguous: say **training** for a model being trained, **session** for audio being made |
| micro | historic. `model_scale=micro` is the base every reshape profile overrides; it isn't a size |
| label | a model's name in Dora (often auto-generated); a **name** is the short one you give it with `model name` |

