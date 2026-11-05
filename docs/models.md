# Models

A **model** is what training produces: one recipe trained on one dataset (see the
[glossary](glossary.md)). On disk it can be a Dora XP (`.dora/xps/<sig>/`) or a portable
bundle (`models/<name>/`). This page is about finding one, naming it, and pointing commands at it.

**Start here:** the published [a9220dd9-ep200 model](https://huggingface.co/AvantBras/a9220dd9-ep200)
comes with a [download/setup/first-generation quickstart](../README.md#a-model-to-start-with).
The helper `uv run python scripts/download_example_model.py` installs its verified bundle into
`models/a9220dd9-ep200/`; no original training XP or dataset is needed.

## Look

```bash
python -m macchina_ricordi.cli model list                         # every model, one line each
python -m macchina_ricordi.cli model list --dataset black_zone    # trained on a dataset whose name contains this
python -m macchina_ricordi.cli model list --recipe reshape-nano --json
python -m macchina_ricordi.cli model show midwife                 # everything known about one model
python -m macchina_ricordi.cli model show midwife@best --json     # ...and which checkpoint @best picks
```

`model show` reads only what training already wrote: the dataset it trained on (from its
Hydra config, which is what training actually read), the recipe, the shape, the segment
length, epochs done and planned (a resume's raised target included), the best valid ce,
the checkpoints on disk, and the model it started from. When two records disagree, for
instance the metadata and the Hydra config naming different workspaces, it says so on a
`warning:` line.

`model show` also prints the model's family and what points at it: its children (and how many
descendants in all), the sessions that played it, counted by kind, and the ledger experiments
that name it. `--json` gives one object per model with the same fields, plus `children` on
`model list --json` and `parents`, `children`, `ancestors`, `descendants`, `sessions` and
`experiments` on `model show --json`. That is the form for scripts and agents.

## The zoo: one graph

Every view of the models reads one graph, built by `macchina_ricordi/models/zoo.py`: the model
records, an edge from each model to each of its parents, every session `play` made, and the
experiment ledger. `model list`, `model show`, `model catalog`, `model lineage`, and the GUI's
covers, index, model page and derive page all render it, so they agree on who came from whom.

Parents are read in one place, `macchina_ricordi/models/provenance.py`, from the
`init_from_checkpoint` a training records. For the file it started from, in order:

1. the sidecars training saw next to that file, embedded in the record. When it holds both a
   merge and a perturbation (a perturbation of a merge-derived checkpoint inherits the merge
   record), the newest operation is the direct one;
2. a `merged_from.json` or `perturbed_from.json` next to the file on disk, for runs recorded
   before training kept that copy;
3. a checkpoint inside `.dora/xps/<sig>/`: that model is the parent;
4. a portable bundle directory the run warm-started from (recorded by its path; the bundle is
   its own record, so its own ancestry is read back through the bundle, not the parent's files);
5. anything else (a file whose sidecar is gone) has no parent model.

A sidecar on disk describes only the file it names as its output: a folder keeps one sidecar,
for the last output written there, so its other files are not explained by it.

A merge's parents are the models behind both of its sources, looking through any merge or
perturbation in between. `model lineage` draws the whole tree, operations included.

The zoo also lists every merge and perturbation output once (`scan_derived`: the sidecars under
`artifacts/checkpoints`, `artifacts/sessions`, `merged_checkpoints`, `perturbed_checkpoints` and each legacy
workspace's `merged_checkpoints`). `model catalog` and the GUI's derive page both read that list.

## Name

```bash
python -m macchina_ricordi.cli model name 4ae63477 midwife
python -m macchina_ricordi.cli model name midwife --clear
```

A name is lowercase, starts with a letter, and may use digits, `.`, `_` and `-`. It cannot
look like a sig or repeat another model's name or Dora label. Names live in
[`musicgen/models.toml`](../musicgen/models.toml), which is Git-tracked: commit it like the
experiment ledger. The same file holds a model's `title`, free-text `notes`, and the
behaviour measured about it (`clean_share`, `top_k_cliff`), which the stream's advisories
and the label tool read. Edit it by hand for anything but names. Nothing is ever written
into `.dora/xps/`.

## Refer to a model

Anywhere a command takes a model, one reference works:

| Reference | Means |
|---|---|
| `midwife` | the model with that name in `models.toml` |
| `4ae63477`, `4ae6` | a sig, or an unambiguous prefix of at least four characters |
| `slerp_a07_warm_midwife_50ep` | a Dora label |
| `models/comelady` | a portable bundle directory, in place or anywhere by path |
| `.dora/xps/4ae63477`, `…/checkpoint_40.th` | a path, as before |

Add a checkpoint with `@`:

| Selector | Picks |
|---|---|
| `@best` | the checkpoint on disk whose epoch had the lowest valid ce |
| `@last`, `@latest` | the newest checkpoint (an epoch file over `checkpoint.th` for the same epoch) |
| `@final` | `checkpoint.th` |
| `@40` | epoch 40 (`checkpoint_40.th`, or `checkpoint.th` if that is epoch 40) |

A typo gets suggestions (`no model called 'midwfe'. Did you mean midwife?`).

## Where references work

| Command | Takes |
|---|---|
| `play generate MODEL`, `play continue MODEL` | a reference as the first argument, or with `--xp-path`; a portable bundle directory is a reference too |
| `play sweep`, `play stream`, `play variations` | `--xp-path midwife@best` (the stream's `--playlist` too); a portable bundle directory works in all of them |
| `play duet` | `--xp-a`, `--xp-b`; either may be a portable bundle directory |
| `train run --from` (alias `--init-from-checkpoint`), `train queue add --from` | `midwife@best` (with no selector, the newest checkpoint), a `checkpoint.th` path, or a portable bundle directory. A queued job may also name the run label of a job queued before it |
| `train queue resume` | the model, or the run label of a queued job |
| `model merge`, `model perturb` | each source checkpoint, e.g. `model merge midwife@best 17087d6a@50 out/checkpoint.th`; `model perturb` also takes a bundle directory |
| `train resume`, `train handoff`, `train watch`, `model curve`, `model lineage` | the model (no `@`: these act on the whole model) |
| `train sidecar` | `--xp-path` |
| `listen portrait` | a name, in addition to what it already matched |

An existing folder or file passed the old way is used as is, so older commands and scripts
keep working.

```bash
python -m macchina_ricordi.cli play generate midwife@best --num-samples 4
python -m macchina_ricordi.cli play stream --xp-path midwife@best --tui --playlist midwife,17087d6a,deep-staircase
python -m macchina_ricordi.cli train run --recipe reshape-nano --dataset .workspace_x --from midwife@best --dry-run
```

## Export

```bash
python -m macchina_ricordi.cli model export midwife@best --box --dry-run
python -m macchina_ricordi.cli model export midwife@best --box
```

A box (`.nrbox`) is the model as a program someone else can play on a CPU ([the appliance](appliance.md)). The
export reads the model record and puts it in the box, so the box knows which model, dataset and epoch it is.

There is a second, lighter way to hand a model over: a **portable bundle** — a plain directory, not a program.
It is what the stream, generation, warm-start and box export can read, so it is the form you host, move and
share.

### The portable bundle

A bundle is one model snapshot frozen into a folder:

| File | What |
|---|---|
| `model.safetensors` | the weights, in [Safetensors](https://github.com/huggingface/safetensors) — data, not code |
| `config.json` | the resolved model architecture, codec contract and snapshot provenance — enough to run it and to warm-start from it |
| `README.md` | who it is, its lineage and its licence |
| `LICENSE` | the weights' licence, which stays separate from the code's MIT |

`config.json` carries the schema `macchina_ricordi.model_bundle.v1`. A bundle is **one snapshot**: no
optimizer or scheduler state, no history, no dataset audio. The parent's files are not needed to use it — a
bundle is self-contained. Bundles are written in Safetensors, which is not a pickled format; nothing here
executes code inside a bundle, only reads weights and configuration. Trust a `.th` checkpoint as before (legacy
pickled AudioCraft files are only ever touched as trusted local inputs), and never treat a downloaded one as
safe to load merely because it is called a checkpoint: never load an untrusted pickled `.th` file.

The standard EnCodec codec used by the default recipe is not inside the bundle — it is downloaded separately from
its public hosted weights, exactly as `.nrbox` export already fetches it. The v1 schema refuses a bundle for a
custom codec or a text-conditioned model. These need their existing compatible legacy workflow; neither
portable-bundle support nor a working `.nrbox` export is promised for them.

```bash
# Snapshot the raw (current) weights of a model into a bundle
python -m macchina_ricordi.cli model export midwife@best --bundle --out-dir models --name midwife --dry-run
python -m macchina_ricordi.cli model export midwife@best --bundle --out-dir models --name midwife
```

- `--bundle` exports the directory form instead of a `.nrbox`. `--out-dir` names the parent directory
  (default `models`); `--name` chooses the bundle folder and public name. The example writes `models/midwife/`.
- `--weights-state model|ema|best` picks which snapshot to freeze. The default is `model` — the raw weights as
  saved. `ema` takes the exponential-moving-average copy when the checkpoint kept one; `best` takes its
  stored best-state weights. The reference (`@7`, `@best`, etc.) chooses the checkpoint file independently.
  Without a selector, the latest checkpoint is used. Requested missing states fail rather than falling back.
  A stored best state may come from an earlier epoch than the containing checkpoint.
- `--force` overwrites an existing bundle; `--dry-run` prints the plan and writes nothing.
- `--box` export takes the same `--weights-state` flag, and its model reference may be a portable bundle
  directory instead of an XP — so you can go straight from a bundle you received to a `.nrbox` for the
  appliance.

### Use a bundle someone else made

Bring the bundle in by hand — download it and extract the folder (any public place works; Hugging Face is
fine as plain file hosting, using a browser or your preferred download tool. Public ungated assets do not
need the retired project-specific upload/download commands):

1. **Drop the folder at `models/<name>/`** — `model list`, `model show`, and the GUI discover bundles there automatically.
2. **Or leave it anywhere** and give its explicit path, for example `model show /path/to/midwife`.
   To have it appear automatically in the GUI and model listings, place the folder under `models/`.

Then it is usable exactly like a trained model: generate from it (`play generate models/midwife`, as the
positional model or with `--xp-path`), stream it (`--xp-path models/midwife`), pick it in the GUI, run it
through `model perturb`, or warm-start a fresh run from it (`train run --from models/midwife --dataset OWN`).

Warm-starting from a bundle is a **new model learning**: the snapshot becomes initial weights with a fresh
optimizer and schedule, on the dataset you name with `--dataset`. No source corpus or parent XP is required —
the bundle itself is the whole starting point.

## Retired Hugging Face transfer

The old checkpoint-transfer surface is gone: `model push` and `model pull`, the earlier
`musicgen hub upload`/`download` spellings, the `hf://repo[@rev][#path]` shortcut for `--from`, and
`--init-checkpoint-token` were retired together. Moving a model to or from another machine is now the portable
bundle above — export it, move the folder, use it on the other side. Public, account-free downloads (the
pretrained codec and listening/captioner models, and the `facebook/musicgen-small` stream fallback) still work
as before.

