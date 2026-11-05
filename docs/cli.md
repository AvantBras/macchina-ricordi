# The command line

`python -m macchina_ricordi.cli <group> <command>`. The groups follow the project's nouns
([glossary](glossary.md)): **data → train → model → play → session**, then **listen / score / judge**,
plus the tools around them. Each group's code lives in the package of the same noun.

| Group | Commands | Package |
|---|---|---|
| `data` | `build` · `list` · `show` · `audit` · `rebalance` · `mix` · `weights list/delete` · `prepare` · `corpus` | `macchina_ricordi/data/` |
| `train` | `run` · `queue add/resume/list/show/cancel/run` · `resume` · `handoff` · `watch` · `sidecar` | `macchina_ricordi/training/` |
| `model` | `list` · `show` · `name` · `lineage` · `catalog` · `curve` · `merge` · `perturb` · `export --bundle/--box` | `macchina_ricordi/models/` |
| `play` | `generate` · `continue` · `sweep` · `stream` · `monitor` · `duet` · `variations` · `replay` | `macchina_ricordi/instrument/` |
| `session` | `list` · `show`: everything `play` made, as takes ([sessions](sessions.md)) | `macchina_ricordi/sessions/` |
| `listen` | text out: `file` (dossier) · `describe` · `describe-dir` · `diff` · `compare` · `ask` · `index` · `portrait` · `describe-calibrate` | `macchina_ricordi/listening/` |
| `score` | numbers out: `file` · `dir` · `structure` · `rescore` · `sim` · `similar` · `novelty` · `calibrate` · `sim-calibrate` | `macchina_ricordi/scoring/` |
| `judge` | verdicts out: `ab` · `label` · `labels` · `prefs` · `witness` · `log` | `macchina_ricordi/judging/` |
| `segment` · `augment` · `loop` | splitting recordings, speed copies, voice-phrase loops | their own packages |
| `tidy` | `prune` · `clean` · `archive` · `backfill` (all dry runs until `--commit`) | `macchina_ricordi/tidy/` |
| `runtime` | `setup` · `check` | `macchina_ricordi/musicgen/` (the AudioCraft/Dora engine) |
| `gui` | the local web front end | `macchina_ricordi/gui/` |

Any command that takes a model takes a [model reference](models.md): a name, sig (or prefix), label or
path, plus `@best`, `@last`, `@final` or `@<epoch>`. Any command that takes a session (`judge ab`,
`score rescore`, `play replay`, `play monitor --session`) takes its folder or its id, and a take is
`<session>#<n>` ([sessions](sessions.md)).

**Listen, score, judge** split by what comes out. `listen` gives text (a dossier, a perceptual description, what
changed between two files, an audio-language model's answers). `score` gives numbers (reward, arc and structure
axes, similarity between two files or across an archive, novelty), all from one similarity engine. `judge` gives
verdicts: your blind A/B choices and labels, and the captioner witness's ratings. Every verdict lands in
`artifacts/preferences/` (`preferences.jsonl`, `labels.jsonl`, `witness.jsonl`, and the GUI's model impressions in
`impressions.jsonl`), and `judge log` reads all four as one list.

## Old spellings

Until October 2026 most of this lived under `musicgen` and `musicgen admin`. The old spellings still work:
they are translated before parsing and print a one-line note (`note: \`musicgen admin resume-xp\` is now
\`train resume\``). Use the current spellings for new commands.

| Was | Now |
|---|---|
| `musicgen data …`, `musicgen train …`, `musicgen runtime …` | `data …`, `train …`, `runtime …` |
| `musicgen train sidecar-cpu` | `train sidecar` |
| `musicgen generate unconditional` / `continue` / `sweep` | `play generate` / `play continue` / `play sweep` |
| `musicgen stream`, `monitor`, `duet`, `variations` | `play stream`, `play monitor`, `play duet`, `play variations` |
| `musicgen admin resume-xp` / `handoff-resume` / `plateau-watch` | `train resume` / `train handoff` / `train watch` |
| `musicgen admin merge` / `perturb` / `lineage` / `catalog` / `score-curve` | `model merge` / `perturb` / `lineage` / `catalog` / `curve` |
| `musicgen admin prune` / `clean-checkpoints` / `archive-checkpoints` / `backfill-metadata` | `tidy prune` / `clean` / `archive` / `backfill` |
| `musicgen admin replay` | `play replay` |
| `score ab` / `label` / `labels` / `prefs` / `judge` | `judge ab` / `label` / `labels` / `prefs` / `witness` |
| `score describe` / `describe-dir` / `describe-calibrate` / `diff` / `compare` | `listen describe` / `describe-dir` / `describe-calibrate` / `diff` / `compare` |
| `listen similar` / `novelty` | `score similar` / `novelty` |

Two of the old spellings are gone outright, not renamed: `musicgen hub upload`/`download` (with the
`model push`/`pull` transfer they stood for) and the `hf://…` checkpoint URI are retired. Hand a model over
as a [portable bundle](models.md#export) instead.

## Cookbook

Everything the GUI does has a command here, and every listing has `--json`.
The examples below assume a prepared dataset called `noisy_10s` and, for the model-inspection
example, an existing model named `midwife`. Replace them with your own dataset and model references;
no audio or trained checkpoints are included in this checkout. See [MusicGen](musicgen.md) for dataset setup.

**Look around** (read-only):

```bash
python -m macchina_ricordi.cli data list --json                 # datasets, built and legacy
python -m macchina_ricordi.cli model list --json                # models: recipe, dataset, epochs, best valid ce
python -m macchina_ricordi.cli model show midwife@best --json   # one model, with its checkpoints and notes
python -m macchina_ricordi.cli train queue list --json          # trainings: queued, running, done
python -m macchina_ricordi.cli session list --json              # sessions: what play made, newest first
```

**Train a recipe on a dataset**, several in a row. Check first with a dry run: it validates the plan without
launching training or recording a training job, but can refresh AudioCraft dataset configuration and fetch
a referenced checkpoint. Running the queue below starts real GPU training; invoke it when ready:

```bash
python -m macchina_ricordi.cli train run --dataset noisy_10s --recipe reshape-deep --epochs 30 --dry-run
python -m macchina_ricordi.cli train queue add --dataset noisy_10s --recipe reshape-deep --epochs 30 --run-label nz-deep
python -m macchina_ricordi.cli train queue add --dataset noisy_10s --recipe reshape-deep --from nz-deep@best --run-label nz-2
python -m macchina_ricordi.cli train queue run                  # returns when the queue is empty
```

**Sample a model and score the takes:** after the queue finishes, confirm completion with `train queue list`.
`play generate` prints a session ID; replace `generate_<run_id>` below with that exact ID.

```bash
python -m macchina_ricordi.cli play generate nz-deep@best --num-samples 4 --duration-seconds 20 --listen
python -m macchina_ricordi.cli session show generate_<run_id> --takes
python -m macchina_ricordi.cli score rescore generate_<run_id>       # writes artifacts/scores/rescore.csv
```

**Share a model, or pick one up** — as a portable bundle, not a hub transfer:

```bash
python -m macchina_ricordi.cli model export nz-deep@best --bundle --out-dir models --name nz-deep   # a folder to host or hand over
python -m macchina_ricordi.cli model show /path/to/nz-deep   # inspect an external bundle by its path
python -m macchina_ricordi.cli train run --recipe reshape-deep --dataset my_own --from models/nz-deep --dry-run   # fresh run from its snapshot
```

Metrics disagree with each other and none is ear-validated: blind A/B (`judge ab <session> <session>`) is the
user's, at the keyboard. See [scoring](scoring.md) and [listening](listening.md).
