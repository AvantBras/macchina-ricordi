# Structure scoring and metric calibration

Three scoring axes now exist, and they do not agree with each other. That is the point of
this document: it explains what each one measures, and how to find out which one matches
your ear instead of guessing.

| axis | command | what it sees | blind to |
|---|---|---|---|
| arrhythmic/percussive | `score file` | busyness, transients, tempo instability | development of any kind |
| arc | rides on sweep `--score` | **harmonic** development (chroma) | timbral, dynamic, rhythmic development |
| structure | `score structure` | development on **four** axes + sections and recurrence | semantics |
| witness judge | `judge witness` | holistic musical quality, semantically | anything a language model can't hear |

## Why the structure axis exists

The arc metric is total variation of chroma. Chroma is a pitch-class distribution, so a
model whose music develops rhythmically or timbrally rather than harmonically reads as
"no development" — it scores near zero on an axis that never applied to it. That is what
happened to the 25 Hz models: `arc_coherence_score` ≈ 0.28 against ≈ 0.78 for the 50 Hz
deep model, which was recorded as "arc-poor" when the accurate statement was
"harmonically flat, but developing normally on every other axis."

`score structure` measures four development axes in parallel, all in `[0, 1]`:

- `dev_harmonic` — chroma movement, computed on the HPSS **harmonic** component so
  percussion stops smearing the pitch estimate.
- `dev_timbre` — MFCC movement.
- `dev_dynamics` — level movement in dB.
- `dev_rhythm` — onset density/strength movement on the HPSS **percussive** component.
- `dev_top2` — the mean of the two strongest axes, so a model that develops on *some*
  axis scores without having to develop on all of them.

and two structural properties from a self-similarity matrix, which plain total variation
cannot see (a random walk maximises total variation without composing anything):

- `ssm_novelty` — Foote checkerboard novelty at its strongest boundary, as an absolute
  contrast. ~0 for uniform texture, →1 for a hard section boundary.
- `ssm_return` — how far similarity climbs back up after falling away, i.e. whether the
  piece comes back to earlier material. A-B-A scores; A-B-C and drones do not.
- `ssm_mean` — average distant-frame similarity: global coherence, "one voice".

The individual `dev_*` and `ssm_*` components above each live in `[0, 1]`. The composite
`structure_score` is reported on a 0–100 scale instead — it is formed with a 100.0 scale
factor around the component blend, so do not compare it directly against an axis value
(an axis at 0.7 is not "70").

```bash
python -m macchina_ricordi.cli score structure clip.wav --json
```

Structure columns also ride along with sweep `--score`, next to the arc columns.

## The silence gate

Near-silent clips post enormous values on every development axis, because the chroma of
near-noise wanders freely. This has produced a wrong conclusion in this project more than
once. So the gate is *inside* the score rather than a filter you have to remember:

- `silence_gate` is 1.0 up to 30 % silence, ramps down, and is 0.0 at 60 %.
- `structure_score` is multiplied by it.
- `arc_coherence_gated` is the arc composite multiplied by it. `arc_coherence_score` is
  left ungated so it stays comparable with numbers recorded before 2026-07-28.
- `summarize_by_group` drops gated clips and reports how many it dropped.

## Re-scoring existing sweeps

```bash
python -m macchina_ricordi.cli score rescore \
  artifacts/sweeps/uncond_deep50_* artifacts/sweeps/uncond_deep25hz_* \
  --output artifacts/scores/rescore.csv --jobs 8
```

Prints per-group means over clips that pass the gate, plus a ranking. Work is librosa/CPU
bound and runs in a process pool.

## The witness judge

`judge witness` asks a captioner witness (Audio Flamingo 3) to rate shape, coherence,
intent and engagement from 1 to 5, and normalises to `[0, 1]`.

```bash
python -m macchina_ricordi.cli judge witness artifacts/sweeps/uncond_deep60_*/combo_000 \
  --paraphrases 2 --output artifacts/scores/judge.csv
```

Two constraints are deliberate. Questions are **holistic**: this project has established
that the witness's claims about timing and order are unreliable, so nothing asks what
happened first. And each axis ships **two phrasings** (`--paraphrases 2`) that are
averaged, because a single phrasing makes the score an artefact of wording.

Every rating is also appended to `artifacts/preferences/witness.jsonl`, beside the A/B verdicts and the labels
(`--no-record` to skip, `--store` for another log). `judge log` lists all three kinds of verdict as one log:

```bash
python -m macchina_ricordi.cli judge log                      # newest 50, with counts per kind and question
python -m macchina_ricordi.cli judge log --kind witness --json
python -m macchina_ricordi.cli judge log --take stream_20261005
```

Known bias, worth holding in mind: the witness was trained on real music, so
"engagement" plausibly rewards sounding like conventional music. For arrhythmic
experimental work that is a confound, not a verdict.

## Blind A/B — the part that actually settles arguments

Every metric here is hand-designed and hand-weighted, and hand-weighted metrics have now
disagreed with the listener twice (the `mega-mid` ranking, then the 25 Hz stack). The
next disagreement is not fixed by a cleverer formula. It is fixed by ground truth.

```bash
# 1. score the candidates
python -m macchina_ricordi.cli score rescore artifacts/sweeps/uncond_* --output /tmp/rescore.csv

# 2. listen blind: each pair is one model at one setting, so only the music differs
python -m macchina_ricordi.cli judge ab artifacts/sweeps/uncond_* --pairs 20 --seconds 10

# 3. find out which metrics match your ear
python -m macchina_ricordi.cli judge prefs --metrics-csv /tmp/rescore.csv

# 4. once there are enough pairs, fit the composite instead of guessing it
python -m macchina_ricordi.cli judge prefs --metrics-csv /tmp/rescore.csv --fit \
  --output artifacts/scores/preference_model.json
```

The ordinary session plays A then B and asks the question (`a` / `b` / `tie`, `r` to
replay, `s` to skip, `q` to quit). The `newer` line flow adds `c` for context-only replay;
`r` repeats context → A → B. `--testing` says what a session tests:

| value | both clips of a pair |
|---|---|
| `metric` (default) | come from one model at the same knobs (a *condition*: the model plus every audio-shaping control, seeds aside), so only chance differs and a verdict says whether a metric follows your ear |
| `parameter:<knob>` | come from one model with every other knob equal, and differ on that one (`parameter:temperature`, `parameter:top_k`): the reveal tallies each pair of values, lower first — "temperature 1 vs 1.3: 2–1–7" is lower wins, ties, higher wins |
| `model` | share every knob and come from different models — a checkpoint counts as its own model, `c5140082@20` vs `c5140082@50` — so a sweep run on two models (or two epochs) pairs combo for combo; the reveal tallies each pair of models the same way |
| `groups` | come from different `--group-by` groups (models, evenings, metric buckets), as before |

A cross-model pair credits the metric with whatever else differed between the clips — the
model's sound, the temperature — so metric agreement is only read from `metric` sessions:
the reveal's running total counts `metric` verdicts only. A sweep is the natural source for
`parameter:` (its combos are a grid); a stream offers the rounds where you moved only that
knob. The tally pools every model among the sources, so pick one model's sessions to learn
about that model. For `model`, sweeps are the source that works: two streams rarely sit on
exactly the same knobs. Sides are shuffled, and pairs
already judged *for that question* are never shown again, so sessions accumulate rather
than repeat.

### The terminal interface

When `judge ab` is attached to a TTY it opens the blind single-keystroke TUI. Use `--tui`
to request it explicitly, or `--line` to force the original prompt-based flow. Without a
TTY, the command always falls back to the line flow.

During judging the TUI shows only the question, pair progress, elapsed time, playback
status, and controls. Paths, groups, metric names and metric values remain hidden until
the session ends. Playback is non-blocking, so switching between A and B does not wait for
the current clip to finish:

| key | action |
|---|---|
| `1` / `2` | play A / B immediately, stopping the current clip |
| `space` | replay the current sequence |
| `c` | replay the context reel in a `newer` session |
| `a` / `b` / `t` | record A, B, or tie and advance |
| `z` | undo the most recent verdict and re-queue that pair |
| `s` | skip the pair |
| `m` | enter a short note for the next verdict |
| `,` / `.` | shorten / lengthen the playback limit |
| `q` | quit; everything already recorded remains on disk |
| `?` | show or hide the key help |

After quitting or completing the last pair, the terminal is restored and the session
prints a de-blinded pair/verdict table (in a `metric` session each side shows its round and
the targeted metric's value; otherwise its group). When the candidate records carry metric
values, it also prints agreement for tonight's judgements and the running-store total over
every clip of the chosen sources, with its verdict (`agrees` / `inverted` / `undecided`,
`insufficient` under 8 pairs); a targeted session prints only the targeted metrics. The session sidecar records skipped and unreached pairs as well as verdicts.

### In the browser

`python -m macchina_ricordi.cli gui`, then **Écoute**, runs the same session from a web page:
same store, same pair selection, same sidecar, and a pair judged in one front end is
never offered again in the other. The server keeps the pairs and serves the audio at
opaque A/B URLs, so the page stays blind until the reveal. See [the GUI](gui.md#blind-listening).

### Where the clips come from

A source is any [session](sessions.md), by folder or id (`stream_20261005T141908Z`, `20261005T1419`):
a sweep (`index.csv`), a **stream session** (`performance.jsonl`), or a generate, continue, duet or
variations session, mixed freely in one command. Stream sessions are read
and never written; the round's logged `score` payload is used as-is, which is the only way
`novelty_raw` / `novelty_whitened` can be tested at all — they depend on the eight rounds
before the clip and cannot be recomputed from the wav. For the offline suite (`dev_*`,
`ssm_*`) run `score rescore` over the session's `rounds/` and pass that CSV to
`judge prefs`.

With `--testing groups`, `--group-by` decides what a cross-group pair *means* for stream rounds:

| value | a pair compares |
|---|---|
| `model_xp` (default) | two models |
| `session` | two evenings |
| `bucket:METRIC` | the top vs the bottom tercile of METRIC, terciles taken within each session, middle dropped |

### Asking something other than "which is better?"

`--question` picks from a fixed catalog (`macchina_ricordi/judging/questions.py`): `better`
(default), `more_developed` ("which one went somewhere?"), `harder` ("which hits
harder?"), `returns` ("which came back to earlier material?"), and the shared-context
`newer` protocol below. An attribute question
gives cleaner signal for its own family and is *not* evidence about anything else — a
`harder` verdict never counts toward `dev_dynamics`, and only `better` feeds the
composite fit.

### Shared-context novelty (`newer`)

Novelty is relative to the previous eight stream rounds, so it needs a stricter protocol
than cold A/B playback:

```bash
python -m macchina_ricordi.cli judge ab artifacts/streams/2026* \
  --question newer --pairs 40 --context-window 3 \
  --disagree novelty_raw,novelty_whitened
```

`newer` accepts stream sessions only; sweeps and mixed sweep/stream inputs are rejected.
Candidates must come from one session and be 1–3 playable `performance.jsonl` positions
apart. Each candidate's memory is the previous eight playable rows in log order. The
exact chronological intersection must contain at least five rounds. In a `metric` session
both rounds also share a condition. With `--testing groups --group-by bucket:novelty_whitened`,
memory is built before high/middle/low tercile filtering, so a middle-tercile round can still
provide honest context for a high-vs-low comparison, and cross-group pairing stays strict
unless `--same-group-ok` is explicit.

Every eligible pair uses all 5–7 shared rounds. The temporary reel takes a fresh-audio
tail from each in order, applies 20 ms fades at each splice, and uses
`min(4.0, 21.0 / shared_round_count)` seconds per round: 20–21 seconds in total. A and B
are also temporary fresh-only WAVs with their repeated prompts removed. Source stream
directories are read-only, preparation is validated before judging, and all temporary
audio is removed on completion, quit, interrupt, or error.

In the TUI each new pair automatically plays context → A → B. `space` repeats that full
sequence, `c` replays only the full context, and `1` / `2` switch immediately to A/B.
The line fallback starts with the same sequence and accepts `c` for context and `r` for
the sequence. `--seconds` and the live `,` / `.` setting apply only to A/B; the context
reel always plays in full.

### Alarm labels and fitted thresholds

Threshold alarms need single-clip yes/no evidence, not A/B preferences. `judge label`
draws 30 clips by default, balanced across three ranges of one alarm, and asks one fixed
question while keeping the path, range and predictor value blind:

| alarm | question | ranges | current reference |
|---|---|---|---|
| `timbral_coherence` | did this collapse / go static? | `<0.95`, `0.95–0.985`, `≥0.985` | alert below `0.985` |
| `silence_gate` | is this effectively silence? | `<0.30`, `0.30–0.60`, `≥0.60` | gate at `0.30`/`0.60` |
| `temperature_floor` | did this go silent or leave disruptive gaps? | `<1.0`, `1.0–1.4`, `≥1.4` | alert below `1.4` |
| `horizon` | did this go silent or leave disruptive gaps? | `<0.90`, `0.90–1.0`, `≥1.0` of trained horizon | warn at `0.90`/`1.0` |
| `top_k_cliff` | did this go silent or leave disruptive gaps? | `≤1`, `1–2`, `≥2` times the model cliff | alert at `≤1` |

```bash
python -m macchina_ricordi.cli judge label artifacts/streams/2026* \
  --alarm timbral_coherence --clips 30

python -m macchina_ricordi.cli judge labels --alarm timbral_coherence
```

The metadata alarms (`temperature_floor`, `horizon`, `top_k_cliff`) accept stream
sessions only. Coherence and silence can use any session whose takes already carry the
predictor: a sweep's `index.csv` or a stream's live score.
Clips come from `sessions/takes.py` (`playable_takes`), the same listing `judge ab` and
`score rescore` read. Stream clips are played fresh-only: the repeated continuation
prompt is removed into a validated temporary WAV, while the label store retains only the
original source path. Temporary audio is removed on every exit path and stream sessions
stay read-only.

The labeling TUI autoplays each clip. `space` replays; `y` / `n` / `u` answer yes, no or
unsure; `z` undoes; `s` skips; `m` adds a note; `,` / `.` adjust the playback limit; `q`
quits. The line fallback uses `yes` / `no` / `unsure`, `r`, `s`, and `q`.

Labels append to `artifacts/preferences/labels.jsonl` under the separate
`macchina_ricordi.scoring.label.v1` schema. Each answer snapshots every available audit value,
its stratum, sampling-frame fingerprint and inclusion probability. Equal-stratum
sampling changes prevalence, so `judge labels` shows both raw confusion counts and
inverse-probability-weighted precision, recall and F1. Weighted estimates remain scoped
to one collection session; multiple sessions also get a clearly marked raw aggregate.
`unsure` is reported but excluded from the confusion matrix. A recommendation is the
weighted max-F1 threshold (precision, then proximity to the current threshold break
ties), and is suppressed below eight decided labels or when only one answer class exists.
Nothing automatically changes a live guardrail.

The complete human handoff is one 30-clip session for each alarm, followed by
`judge labels` for each report. The hard prompt-length clamp is not a listening audit:
it prevents a crash and deliberately remains outside this bench.

### Spending judgements where they inform

Judgements-per-evening is the scarce resource, so pairs can be targeted:

```bash
# pairs this metric strongly separates (top half of |Δ| by default)
python -m macchina_ricordi.cli judge ab artifacts/streams/2026* --target-metric arc_coherence_score

# pairs where two metrics rank the sides oppositely — which one does the ear follow?
python -m macchina_ricordi.cli judge ab artifacts/streams/2026* --disagree arc_coherence_score,score
```

Targeting buys informativeness, not isolation: a pair with high contrast on one axis
usually has contrast on others too, so agreement over targeted pairs is evidence about
the metric, never proof that nothing else moved. The session prints that caveat itself.
When targeting leaves no pairs, the command says so rather than quietly falling back to
untargeted ones.

### What a session leaves behind

Verdicts append to `artifacts/preferences/preferences.jsonl` (`preference.v1`: the v0
records read back as `better`), carrying the question, the session id and the `--seconds`
truncation that was in force. `newer` records retain only original source paths plus the
protocol version, session positions, gap, memory/overlap sizes, excerpt recipe, and
fresh-only playback metadata; no temporary path is persisted. Each run also writes
`artifacts/preferences/sessions/<UTC>_session.json` — sources, question, seed, targeting,
and every pair offered, including the skipped and unreached ones. The store says what was
judged; the sidecar says what was offered.

Single-clip sessions write `<UTC>_label_session.json` in the same sessions directory,
including all offered, skipped and unreached clips. Their verdicts live only in
`labels.jsonl`; they never enter the pairwise preference store.

`judge prefs` reports per-metric **agreement**, one table per question: the fraction of
pairs where the metric's higher value picked your winner. 0.5 is a coin flip, below 0.5
means the metric is *inverted*. With `--fit` it runs a regularised Bradley-Terry fit over
metric differences — `better` judgements only — and projects the result onto
`structure_score`'s weights, which `load_structure_weights()` can then activate.

Around 50 pairs is where the agreement numbers start to mean something; below ~8 per
metric they are reported as `insufficient`. An evening of listening is the intended dose.
Paths in the store are matched against the scored CSV in both relative and absolute form,
so a session sourced from `artifacts/streams/…` still lines up with a CSV holding full
paths.

## Interpreting a disagreement

As of 2026-07-28, on the unconditional sweeps, the three axes rank the models
differently. When they disagree, the ranking to trust is the one that agrees with your
blind A/B store. If no metric agrees, that is a finding: the axis you care about has not
been built yet.
