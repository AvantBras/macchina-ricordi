# MusicGen Stream — Playing the Model

`play stream` turns generation into an iterative loop you can *play*. It
keeps generating continuation rounds against the previous tail, and re-reads a
small JSON file at the start of every round so you can change knobs while it
runs. Each completed round is enqueued to a sounddevice playback worker so
you hear round N while round N+2 renders. Generation runs through one persistent
`.venv-musicgen` worker process, so the model is loaded once per session and
reloaded only when `model_xp` changes.

## Quickstart

Run from the source checkout after [setting up the MusicGen runtime](musicgen.md#1-set-up-the-runtime).

```bash
# Install the system PortAudio library (Debian/Ubuntu/Pop!_OS).
sudo apt install libportaudio2
# Install the optional Python audio dependency once.
uv sync --extra dev --extra instrument

# Run a session against a trained XP. Ctrl-C ends it cleanly.
python -m macchina_ricordi.cli play stream \
  --xp-path .dora/xps/0444687c \
  --device cuda \
  --max-rounds 0 \
  --completion-duration 10 \
  --tau 5
```

The `--xp-path` above is an example checkpoint. Stream mode plays through a
trained model, so supply one you have — any [model reference](models.md), a
local Dora XP folder, or a [portable bundle](models.md#the-portable-bundle)
directory such as `models/comelady` — and a GPU (`--device cuda`). The private XPs named
throughout this document are illustrative and are not downloadable from this checkout.

Without a local checkpoint, omit `--xp-path` to use the public
`facebook/musicgen-small` model. It downloads pretrained weights on first use
and still requires the MusicGen runtime and a suitable NVIDIA GPU:

```bash
uv run ricordi play stream --tui --device cuda
```

The first line of output names the session directory. Open
`<session_dir>/control.json` in your editor while the loop runs and the next
round will pick up your changes.

## The control file

`control.json` lives in the session directory and is re-read at the start of
every round. Atomic writes only — the loop tolerates partial reads.

Every program that rewrites it holds `control.json.lock` for the whole
read-modify-write (`control_file_lock` in
`macchina_ricordi/instrument/stream/session.py`): the TUI's keys, and the loop when it
resets a one-shot command, so two writers never drop each other's edit. The
loop resets only the command in the file as it then is, so a knob moved right
after a `pin` or a `reroll` stays moved; it used to write back the copy it read
when the round started, which undid the move.

```json
{
  "command": "continue",
  "pin_target": null,
  "tau_seconds": 5.0,
  "completion_duration_seconds": 10.0,
  "temperature": 1.0,
  "top_k": 250,
  "top_p": 0.0,
  "min_p": 0.0,
  "cfg_coef": 3.0,
  "extend_stride": 8.0,
  "model_xp": ".dora/xps/0444687c",
  "candidates": 1,
  "novelty_weight": 0.0,
  "stuck_weight": 0.0,
  "stuck_kick_rounds": 0,
  "stuck_kick_temperature": 0.4,
  "stuck_kick_top_k_factor": 2.0,
  "motif_dir": null,
  "motif_period": 0,
  "motif_mix": 0.5,
  "prompt_tilt_low_db": 0.0,
  "prompt_tilt_mid_db": 0.0,
  "prompt_tilt_high_db": 0.0,
  "prompt_rate": 1.0,
  "prompt_rate_mode": "tape",
  "max_buffer_ahead_seconds": null,
  "resume_buffer_ahead_seconds": null,
  "throttle_enabled": true,
  "round_delay_seconds": 0.0
}
```

| Field | Meaning |
| --- | --- |
| `command` | One of `continue` (default), `reroll`, `pin`, `inject`, `inject_light`, `inject_heavy`, or `stop`. `reroll`, `pin`, all injection commands, and `stop` are one-shot: the loop resets them to `continue` after acting on them. |
| `pin_target` | Internal one-shot target written by the TUI when `p` is pressed: the round currently playing (or next queued round at a playback boundary). It is cleared when the command is consumed. Manual `pin` edits can omit it. |
| `tau_seconds` | Length of the previous round's tail used as the next continuation prompt. Typical: 3–8s. Clamped to the round's actual length, and to 1s below the model's trained `segment_duration` — see [the model horizon](#the-model-horizon). |
| `completion_duration_seconds` | How many seconds of fresh audio each round should append (1–60s). The seed uses this as its full duration; continuation requests add it after the actual prompt window. |
| `temperature`, `top_k`, `top_p`, `cfg_coef`, `extend_stride` | Sampling knobs (same meaning as `play generate`). |
| `min_p` | min-p sampling threshold (0–1; replaces top-k/top-p when > 0): keep tokens with `p >= min_p * p_max` after temperature. Confidence-adaptive — meant for *high* temperatures (see [creative tools](musicgen-creative.md#min-p)). At low temperature in confident passages it can collapse all `candidates` takes to the same audio; pair it with temp ≥ 1.3 or leave it 0 when relying on best-of-N. |
| `candidates` | Best-of-N: generate N takes per round in one batched call (near-free on GPU — the AR loop is launch-bound, so bs=4 costs the same wall-clock as bs=1) and keep the lowest-silence/highest-arc take. Takes whose *ending* is silent are ranked below the ones that keep playing (last 10% of fresh audio above 35% silence): a round's tail is the next round's prompt, so a fade-out compounds into a silent stream. Both silence gates are skipped rather than enforced when no take clears them. Losers are deleted; per-candidate scores land in `performance.jsonl` under `score.candidates` (`silence_fraction`, `tail_silence_fraction`, `arc_coherence_score`, `novelty`). Clamped to 1–8. Selection scoring costs ~0.5s CPU per candidate, so keep `N x 0.5s` inside the round budget. |
| `novelty_weight` | Anti-déjà-vu (0–1; needs `candidates` > 1): candidate ranking becomes `(1-w)*arc + w*novelty`, both minmax-scaled within the batch, where novelty is the distance from the last 8 selected rounds' MFCC fingerprints. `0.3–0.5` audibly steers the stream away from re-entering the same well; the silence gates still apply first. Per-candidate `novelty` lands in `score.candidates`. |
| `stuck_weight` | Drone guard for best-of-N (0–1; needs `candidates` > 1): ranking becomes `(1-w)*rank + w*(1-stationarity)`, minmax-scaled in the batch like novelty, so candidates whose last 5 s barely move lose. Stationarity is the mean cosine of each frame's mel spectrum to the window's mean one; it also catches near-drones that would set in a few rounds later. Per-candidate `stationarity`, `chroma_top` and `drone_score` land in `score.candidates`. No threshold. |
| `stuck_kick_rounds` | Drone escape (0 = off; try 2). When this many rounds in a row end on a drone, the next round alone runs at `temperature + stuck_kick_temperature` and `top_k * stuck_kick_top_k_factor` (default +0.4 / ×2, capped at 3.0 / 2048), then your settings come back. A drone is `drone_score = stationarity × chroma_top >= 0.4` on the last 5 s of fresh audio. That is an absolute level, but it sits mid-gap: blind-labelled drones scored 0.60–0.70 across 6 models, and nothing else went above 0.26. Fires on ~2.5% of past played rounds, mostly on the models that drone. A drone that survives gets another kick k rounds later. Fires a `stuck_kick` event; kicked rounds carry `stuck_kick` in `performance.jsonl`. Works at `candidates = 1`. Validation: drone guard. |
| `motif_dir` | Motif bank directory for `pin`, one-shot injection commands, and recapitulation. `null` = `<session>/motifs`. Point separate sessions at a shared bank to give performances a common identity. |
| `motif_period` | Every Nth round, the continuation prompt becomes a blend of the current tail and a pinned motif's tail — themes *return* instead of drifting away forever. `0` = off. The motif is chosen deterministically from the bank per round (seeded by `--seed`). |
| `motif_mix` | Motif share in the recap blend (0–1, default 0.5). Higher = harder recall of the theme, lower = a faint echo. |
| `tail_rescue` | On by default. When the tail that would become the next prompt is more than 60% silence, re-prompt from the latest earlier window *in the same round* that still has audio, stepping back in 0.5s hops. Silence at the end of a round is otherwise self-feeding: it becomes the next prompt, the model continues the fade, and the stream dies from the tail inward. The cost is an audible jump cut, so the ceiling is set high enough that only a genuinely dying tail trips it. The search never reaches back into that round's own prompt echo (re-prompting with audio that was already a prompt is exactly the déjà-vu `novelty_weight` fights), and it gives up rather than reach for a worse window — a round that is silent throughout keeps its plain tail. When it fires it prints `tail_rescue:` and records `tail_rescued_from_seconds` in `performance.jsonl`. Costs one extra audio decode of the previous round per round. Unlike the candidate tail gate, this works at `candidates = 1`. |
| `prompt_tilt_low_db` / `prompt_tilt_mid_db` / `prompt_tilt_high_db` | Three-band EQ (±12 dB) applied to the continuation prompt *before* the model reads it, so it hears a bass- or treble-heavy recent past and continues from it. Crossovers at 250 Hz and 4 kHz, smoothly, and the band weights are normalised so setting all three to the same value is a flat gain of exactly that. Works at `candidates: 1` and **compounds** while held. Bound to `j`/`J`, `k`/`K`, `l`/`L`; see [Prompt steering](#prompt-steering-tilt-and-rate). |
| `prompt_rate` | Varispeed of the prompt window only (0.5–2.0). `rate × tau` seconds of tail are resampled down to *exactly* tau, so prompt token count, the horizon guard and the buffer arithmetic never move. Only what is fed back is shifted, never what is played. Bound to `<`/`>`. |
| `prompt_rate_mode` | `tape` (default) resamples — pitch and tempo move together, transients intact. `tempo` phase-vocodes — pitch preserved, transients smeared, which on arrhythmic percussive material is very audible. Control-file only; no key. |
| `model_xp` | Optional Dora XP folder. When changed mid-session, the persistent worker reloads that XP for the next continuation round. Invalid paths are skipped with a `model_swap_skipped:` warning; the loop keeps the previous XP. The first round after a swap takes longer (model load), and the seam is audible as a deliberate scene change. Setting it back to `null` keeps the current XP — to "unset", restart the session. In the TUI, `--playlist` binds the digit keys to a set of XPs so a swap is one keystroke. |
| `max_buffer_ahead_seconds` | Throttle cap: pause generation once the playback buffer is this far ahead, resume at `resume_buffer_ahead_seconds`. `null` = no throttle. `--max-buffer-ahead-seconds` only sets the *initial* value; the control file is where it then lives, and it is re-read **inside** the throttle wait, so a change lands during the wait rather than a round later. See [Throttling](#throttling--keeping-the-buffer-from-running-away). |
| `resume_buffer_ahead_seconds` | Hysteresis resume threshold (the buffer floor), clamped to `[0, max]`. `null` = `max - 10`. `--buffer-floor-ratio` sets it as a fraction of the cap at launch; the TUI's `{`/`}` move that fraction and `-`/`+` keep it. Raising it above the current buffer ends a wait in progress; lowering it (with the cap) holds the wait until the buffer has actually shrunk. |
| `throttle_enabled` | `false` holds the throttle open: generation runs ahead unbounded while the cap above stays parked, so switching back restores this session's own thresholds. This is how you bank a cushion before a `model_xp` swap — see [Banking a cushion](#banking-a-cushion-for-a-model-swap). Bound to `t` in the TUI. Distinct from `max_buffer_ahead_seconds: null`, which means the session has no throttle configured at all. |
| `round_delay_seconds` | Rest this long before each round (0–3s, after any throttle wait), then re-read the knobs, so a change made during the rest lands in that round. Cut short (`rest_cut` event) once the buffer holds no more than the last round took to make, so a rest never runs playback dry. `0` = off. `--round-delay-seconds` sets the initial value; `w`/`W` in the TUI. |

`--round-duration` and `round_duration_seconds` are intentionally removed from
stream mode rather than aliased: migrate commands and live control files to
`--completion-duration` and `completion_duration_seconds`. The read-only
monitor can still display older performance logs by deriving their historical
fresh duration as old total duration minus `tau`.

### `command: pin`

"I like what I just heard — remember it." In the TUI, copies the round that is
playing when you press `p` into the motif bank as
`motif_<run_id>_<round-file-stem>.wav`, then continues generating as usual.
This target is captured at keypress time, so a generation buffer does not make
pinning jump ahead to the newest generated round. A manually edited `pin`
without `pin_target` uses the playback worker's current chunk when the command
is observed, or the continuation source when playback is disabled. Combine
with `motif_period` for live thematic form: pin two or three rounds you love
early in a session, set `"motif_period": 5`, and the piece keeps coming back
to them. Pinned motifs persist on disk after the session ends.

### `command: inject_light` / `command: inject_heavy`

"Bring the last pinned idea into the next continuation." Press `i` in the TUI
for `inject_light` (`0.2` motif share), or `I` for `inject_heavy` (`0.6` motif
share). You can also write either command to `control.json`. The newest motif
in `motif_dir` is mixed into the current tail, and that mixed prompt is used
for the next sample only. The command is consumed before generation, so the
following sample returns to the ordinary tail. If the motif bank is empty or
the blend fails, generation continues with the ordinary tail and records an
`inject_skipped` warning. The older `inject` command remains a `0.5`-weight
compatibility alias.

### `command: reroll`

The current round is re-rendered with a fresh seed (same control values, new
take number). The file is saved as `round_NNN_take_K.wav` alongside the
original `round_NNN.wav`. The *last* take of each round is what feeds the
next round's prompt.

### `command: stop`

Loop exits cleanly before the next continuation: drains playback, writes the
manifest, returns 0. Ctrl-C is the second stop path.

## Session layout

A stream is a [session](sessions.md) whose takes are its rounds. New ones go to
`artifacts/sessions/<date>/stream_<run_id>/` (with `--output-dir D`, `D/<run_id>/`); the ones from before
October 2026 stay in `artifacts/streams/<run_id>/`.

```
stream_<run_id>/
├── session.json              # the session record, written when the stream stops
├── control.json              # live-edited; loop re-reads each round
├── control.json.lock         # empty; held while a program rewrites control.json
├── live_status.json          # what the loop and the speakers are doing right now (see below)
├── playlist.json             # the --playlist models in key order (only with --playlist)
├── launch.json               # the GUI job that started it and its log (only when started from the GUI)
├── manifest.json             # final summary (also written on Ctrl-C)
├── performance.jsonl         # one row per round, replayable later
├── events.jsonl              # one row per event (see below)
├── worker.log                # generation subprocess diagnostics (--tui only)
│                             # manifest.json records which novelty calibration was used
├── motifs/                   # appears on first `pin` (unless motif_dir is set)
│   └── motif_<run_id>_round_004.wav
├── prompts/                  # blended prompts written on recap rounds
│   └── prompt_006.wav
└── rounds/
    ├── round_000.wav         # round 0 (seed)
    ├── round_001.wav
    ├── round_001_take_1.wav  # appears if you set command:reroll on round 1
    ├── round_002.wav
    └── ...
```

Recap rounds record `motif_source` (the motif file used) in
`performance.jsonl`, and their `prompt_audio` points at the blended
`prompts/prompt_NNN.wav` instead of the previous round.

`performance.jsonl` captures, per round: `round_idx`, `take_idx`,
`timestamp`, `command_consumed`, `active_control` (full snapshot),
`seed_used`, `round_wav`, `duration_seconds`, `elapsed_seconds`,
optional `score`, the prompt-window pointers for continuation rounds,
the active `model_xp`, plus the per-round **buffer telemetry**:
`playback_seconds` (how much of this round's audio actually reaches the
speakers — `duration_seconds - prompt_duration_seconds`) and
`buffer_seconds_ahead` (how far ahead of the playback head the queue was
at the moment the round was enqueued). This is the substrate a future
"replay a performance" command will read.

### Reading the per-round status line

The loop now prints one line per round to stdout:

```
seed: gen=11.4s budget=10.0s ahead=+10.0s
round 1: gen=12.1s budget=10.0s ahead=+14.9s
round 2: gen=12.4s budget=10.0s ahead=+14.5s
round 3: gen=13.0s budget=10.0s ahead=+13.5s
model_swap: .dora/xps/A -> .dora/xps/B
round 4: gen=22.7s budget=10.0s ahead=+3.8s
round 5: gen=12.3s budget=10.0s ahead=+3.5s
```

- `gen`: wall-clock seconds the generation step took.
- `budget`: net new audio this round contributes to playback
  (`completion_duration`).
- `ahead`: estimated seconds of audio queued or in-flight that the
  player hasn't consumed yet. **Positive and growing** = the buffer is
  filling and you have slack for disruptions; **trending toward zero** =
  you'll underrun soon; **`+0.0s`** = playback is starving (this is
  where ALSA xruns appear).

The `ahead` figure is a wall-clock approximation: it ignores PortAudio's
small internal buffer (~100ms with `latency='high'`) and undercounts
after any silence gap (because wall-clock keeps ticking through silence
the buffer didn't actually queue). For steady-state tuning of
`completion_duration` / `tau` / model choice, that's plenty accurate.

`buffer_seconds_ahead: null` in `performance.jsonl` means playback was
disabled (`--no-play` or sounddevice unavailable). The
`playback_seconds` column is still populated.

### Novelty is always measured now

Every round records the selected take's distance from the last 8 rounds'
fingerprints, in `performance.jsonl` under `score`:

| key | meaning |
|---|---|
| `novelty_raw` | `1 − max(cosine similarity)` over unwhitened MFCC fingerprints |
| `novelty_whitened` | `(1 − max(cosine similarity)) / 2` after per-dimension z-scoring; `null` without a calibration |

Read `novelty_whitened`, falling back to `novelty_raw` — that is what the
monitor does. (A third field, `novelty_vs_memory`, held whichever of the two
was available and was dropped on 2026-08-06: it was a literal alias of the
whitened value in every calibrated session and silently meant something else
otherwise. Sessions recorded before then still carry it.)

This happens at `candidates: 1`, at `novelty_weight: 0`, and with
`--score-rounds` off — the configuration you normally play in, and the one
where novelty used to be computed nowhere at all. It costs one MFCC per
round. The seed round has `null` (nothing to compare against yet).

**Novelty is a reading, not a control.** `novelty_weight` still feeds
candidate selection, but that blend is known-broken and awaits validation.
Raw and whitened are logged side by side precisely so they can be compared
on identical rounds.

`--novelty-calibration PATH` supplies the whitening statistics. The public
checkout ships no calibration, so unless you supply a usable file novelty uses
unwhitened cosine distance. This was near 0.004 with little dynamic range in the
author's measurements; the session reports missing calibration
with a `novelty_calibration` warning event. Build the file with:

```bash
python -m macchina_ricordi.cli score sim-calibrate analysis/calibration_corpus \
  --output analysis/novelty_calibration.json
```

and pass it to the stream command explicitly:

```bash
python -m macchina_ricordi.cli play stream \
  --xp-path .dora/xps/0444687c --device cuda \
  --novelty-calibration analysis/novelty_calibration.json
```

The corpus behind that file decides what "novel" is measured *against*, so
the choice is a real one. Keep a given calibration file fixed when comparing
sessions: rebuilding it changes the scale of future whitened readings, not values
already written to old session logs. Only the
`mfcc` block is used here; the `rhythm` block of any
calibration written before 2026-08-06 is one dimension too wide since
`dominant_tempo_consistency` left the vector, and `load_calibration` now skips
a stale block rather than installing it. Whichever is used is
recorded in `manifest.json` under `novelty_calibration`. Retain the file and its
provenance when comparing whitened readings; do not treat raw and whitened values
or values from different reference corpora as interchangeable.

### The event sidecar

`performance.jsonl` has one row per round and no notion of an *event*, so
things like throttle waits and playback errors used to exist only as
stdout lines — invisible to anything reading the session afterwards.
Every such line is now also appended to `events.jsonl`:

```json
{"timestamp": "2026-07-29T11:13:41Z", "round_idx": 2, "kind": "tail_rescue",
 "level": "warning", "message": "round_001.wav tail is silent; re-prompting from 1.5s (was 3.0s)",
 "data": {"rescued_to": 1.5, "rescued_from": 3.0}}
```

`kind` is the same token the terminal prints before the colon, `level` is
`info` / `warning` / `error`, and `round_idx` is the round the generator
was working on when it fired. The kinds: `session_start`, `session_end`,
`tail_rescue`, `model_swap`, `model_swap_skipped`, `pinned`,
`pinned_injection`, `inject_skipped`, `motif_recap`, `motif_skipped`, `throttle`,
`generation_slow_warning`, `horizon_guard`, `playback_started`, `playback_warning`,
`playback_error`, `live_status_failed`.

Writes are single appends taken under a lock, because the playback worker
emits from its own thread. If the sidecar cannot be written the session
prints `event_log_disabled:` once and carries on printing normally — a
performance is never interrupted for a log.

### The live status

Two things a session knows only in memory: what the round loop is doing right
now, and which round the speakers are on. The TUI can draw them because it runs
inside the session. For anything in another process (the GUI), the session
rewrites `live_status.json` every half second, from its first round until it
stops:

```json
{
  "pid": 2414957,
  "updated_at": "2026-10-22T09:42:55.089230+00:00",
  "phase": "generating",
  "round_idx": 214,
  "take_idx": 0,
  "phase_seconds": 4.2,
  "playing": "/…/stream_20261022T093000Z/rounds/round_210.wav",
  "buffer_seconds_ahead": 38.5
}
```

| key | meaning |
|---|---|
| `phase` | `starting`, `waiting` (between rounds), `throttled` (waiting for the buffer to drain), `resting` (`round_delay_seconds`), `generating`, `scoring` or `stopped`; for `round_idx` / `take_idx`, entered `phase_seconds` before `updated_at` |
| `playing` | the round coming out of the speakers, or the next queued one at a boundary; `null` with playback off or drained. It is what `p` pins in the TUI, so another front end hands it back as `pin_target` |
| `buffer_seconds_ahead` | the buffer now, where `performance.jsonl` only has the reading taken when each round was queued |
| `pid`, `updated_at` | the heartbeat. A session is live while its process exists and the file is fresh; one killed before it could write its manifest leaves a stale file behind |

Like the event sidecar it is best-effort: a write that fails logs one
`live_status_failed` event, and the session plays on.

## Playing from one terminal: `--tui`

`play stream --tui` draws the dashboard in the terminal it runs in and
takes keyboard input, so one window both shows the session and steers
it — no second terminal, no editing JSON mid-performance.

The GUI's [Scène page](gui.md#playing-a-stream) is the same instrument in a
browser, with the same keys; the two can steer one stream at once.

```bash
python -m macchina_ricordi.cli play stream --tui \
  --xp-path .dora/xps/8aa68bbe --device cuda --completion-duration 10 --tau 5
```

| Key | Action |
| --- | --- |
| `↑` / `↓` | `temperature` ±0.05 (0.05–3.0) |
| `←` / `→` | `tau_seconds` ∓/± 0.5 (0.5–60; independent of completion) |
| `[` / `]` | `completion_duration_seconds` ∓/± 1.0 (1–60) |
| `,` / `.` | `top_k` ∓/± 10 (0–2048; 0 = off) |
| `n` / `N` | `novelty_weight` ∓/± 0.05 (0–1) |
| `d` / `D` | `stuck_weight` ∓/± 0.05 (0–1) |
| `g` / `G` | `stuck_kick_rounds` ∓/± 1 (0–8; 0 = off) |
| `c` / `C` | `candidates` ∓/± 1 (1–8) — best-of-N. Every ranking nudge, `novelty_weight` included, is inert at `1` |
| `-` / `+` | `max_buffer_ahead_seconds` ∓/± 5 (5–300); the floor follows at the same fraction of the cap. On an unthrottled session the first press switches the throttle on at 30s |
| `{` / `}` | buffer floor (`resume_buffer_ahead_seconds`) ∓/± 0.1 × the cap (0–1). 1.0 tops the buffer up round by round; 0 drains it dry first |
| `w` / `W` | `round_delay_seconds` ∓/± 0.1 (0–3) |
| `t` | throttle on/off, keeping the cap. Hold it open to bank a buffer cushion before a model swap — see below |
| `j`/`J` `k`/`K` `l`/`L` | prompt tilt low / mid / high band ∓/± 1.5 dB (±12). Three adjacent keys, low to high left to right; shift boosts |
| `<` / `>` | `prompt_rate` ∓/± 0.02 (0.5–2.0) — varispeed of the prompt only |
| `0` | prompt steering back to neutral: tilt flat, rate 1.0. Leaves every other control alone |
| `1`–`9` | swap `model_xp` to that slot of `--playlist` |
| `tab` | cycle to the next `--playlist` model, wrapping |
| `r` | queue a `reroll` |
| `p` | pin the round currently playing (target captured immediately) |
| `i` | inject the newest pinned motif into the next sample at weight `0.2` |
| `I` | inject the newest pinned motif into the next sample at weight `0.6` |
| `Q` | queue a `stop` — capital only, so a stray `q` cannot end a performance |
| `f` | toggle metrics between the last-generated and now-playing round (display only) |
| `b` | blind: hide every per-round reading (display only — see below) |
| `?` | toggle the key list |
| `Ctrl-C` | interrupt, exactly as without the TUI |

Holding a key sweeps: every repeat in a burst is applied, not just the
first. Each edit flashes a confirmation (`temperature 1.20 → 1.25`) under
the header, and the header says which round it lands at. Most edits affect the
generator's future rounds; `p` is the exception and captures the playback
round at the moment of the keypress.

**Keys write `control.json`.** There is no second source of truth: the
same read-modify-write an editor would do, holding `control.json.lock` like
the loop's own command reset (see [the control file](#the-control-file)). So
you can still edit the file by hand in
another terminal while the TUI runs, and fields with no key —
`min_p`, `motif_*`, `cfg_coef`, `prompt_rate_mode` — are edited
that way and show up in the TUI's control row.

### Model playlist keys

`--playlist` takes a comma-separated list of model folders — XPs or
[portable bundles](models.md#the-portable-bundle) — and binds them to
the digit keys, with `tab` to cycle:

```bash
python -m macchina_ricordi.cli play stream --tui \
  --playlist .dora/xps/8aa68bbe,.dora/xps/f05eba09,.dora/xps/0444687c \
  --device cuda --completion-duration 10 --tau 5
```

`1` through `9` set `model_xp` to that slot; the round loop then does its
ordinary [live model swap](#live-model-swap) on the next round, seam and
reload cost included. A scene change becomes a keystroke instead of a
mid-performance edit. Press `?` and the playlist is listed under the key
help, numbered, so you don't have to remember the order.

The first entry is also the starting model when `--xp-path` is omitted —
otherwise slot 1 would only become true once you pressed it. Entries are
resolved to absolute paths at startup, the same normalisation the loop
uses to decide whether a swap is really a swap, so pressing the key for
the model already loaded flashes `already loaded` instead of paying for a
reload. Folders that do not exist are reported at startup
(`playlist_missing:`) and, if pressed, take the loop's usual
`model_swap_skipped` path rather than killing the session. Only the first
nine entries get a key. The list is also written to the session as
`playlist.json`, so the GUI's Scène page offers the same numbered slots.

### Prompt steering: tilt and rate

The tilt's range and crossovers, the tape-speed limits and the weights of `i`/`I` are shared with [the
appliance](appliance.md#controls) through one vocabulary, `macchina_ricordi/instrument/controls.py`, so a gesture means
the same thing on both instruments.

Every other control changes *how the model samples*. These two change **what
the model believes its recent past was**, by transforming the tail before it
is re-prompted — the same transformed-prompt path `motif_mix` and
`tail_rescue` already ride. Two consequences follow, and they are the whole
character of these knobs:

- **They work at `candidates: 1`.** Nothing is being selected; the
  generation itself is being steered.
- **They compound.** Each round's output becomes the next round's prompt, so
  a tilt held for a few rounds drags the stream's whole spectrum rather than
  colouring one round. Return to neutral and the drift *stops where it got
  to* — the stream stays there, because the model is now genuinely making
  that music. This is a different gesture from every other control, which
  snap back when you let go.

The interesting failure mode is the model fighting back toward its training
distribution, and that fight is audible: push the low band hard and a micro
trained on bright percussion will keep trying to climb back out.

```bash
python -m macchina_ricordi.cli play stream --tui \
  --xp-path .dora/xps/8aa68bbe --prompt-tilt-low 4.5 --prompt-rate 1.04
```

**Tilt** is a three-band EQ on the prompt window: `j`/`J` low (below
250 Hz), `k`/`K` mid, `l`/`L` high (above 4 kHz), lowercase cutting and
shift boosting, 1.5 dB per press to ±12. Crossovers are smooth rather than
brick-wall — a hard band edge rings, and ringing would be indistinguishable
from the model responding. The band weights are normalised, so setting all
three to the same value is a flat gain of exactly that many dB and nothing
else.

**Rate** (`<` / `>`) is tape-style varispeed of the prompt *only*. The
mechanically important part: `rate × tau` seconds of tail are resampled down
to **exactly tau**, rather than tau seconds being stretched to tau/rate. Token
count, the [horizon guard](#the-model-horizon) and the buffer arithmetic are
therefore untouched by a rate change — otherwise every nudge would silently
perturb the round budget. Because only what is fed back moves, and never what
is played, the drift arrives a round later as the music genuinely migrating in
pitch and tempo rather than as an effect pedal on the output. `±10%` is well
inside what EnCodec round-trips cleanly; past roughly `±15%` it contributes
artefacts of its own.

`prompt_rate_mode` picks how: `tape` (default) resamples, so pitch and tempo
move together and transients survive; `tempo` phase-vocodes, preserving pitch
and smearing transients — very audible on arrhythmic percussive material,
which may be the point or may be exactly wrong. Control-file only.

`0` puts both back to neutral in one press, and touches nothing else.
Counting keypresses back to flat mid-performance is not playing.

Two honest limits:

- A rate above 1.0 needs `rate × tau` seconds of tail ending where the plain
  window ends. When the round cannot reach that far back — a short round, or
  a recap round whose prompt is a freshly rendered blend exactly tau long —
  the window is clamped to what exists and the round runs at a **lower**
  rate than requested. That is logged once as `prompt_rate_clamped` and
  `performance.jsonl` records the rate actually applied, not the one asked
  for.
- The transform is applied *last*, after any motif blend, so what the model
  hears is what the keys say. The cost is one extra decode and one small WAV
  per round in `<session>/prompts/`, paid only while a knob is off neutral.

Both land in `performance.jsonl` per round (`prompt_rate`,
`prompt_rate_mode`, `prompt_tilt_db`) and show on the monitor's flags line
and control row while engaged.

### Banking a cushion for a model swap

A swap stalls generation for the model load — 10–20s for a
micro-unconditional on a 3060, longer for the bigger reshape models. Nothing is
being generated during that time, so the only thing keeping the speakers
fed is buffer you already have. Under a throttle you often do not: the
whole job of the cap is to *stop* the buffer from growing, so a 20s cap
and a 40s reload means the player runs dry mid-swap.

`t` holds the throttle open. Generation runs ahead unbounded until you
press it again; the cap stays parked in `control.json`, so switching back
restores this session's own thresholds rather than a default.

The working gesture is:

1. `t` — throttle off. Watch `buffer ahead` climb in the frame.
2. Wait until it comfortably exceeds the reload cost of the model you are
   going to.
3. `1`–`9` or `tab` — swap. The reload stall is spent out of the cushion.
4. `t` — throttle back on. The buffer drains to the cap and the
   instrument is responsive again.

While it is open the frame says so on the buffer line, for as long as it
lasts:

```
  buffer ahead   142.0s  ⚠ throttle off — running ahead (cap 60s parked)
```

That warning is deliberate. A throttle left open is not harmful in
itself, but the buffer grows without bound, and buffer *is* the latency
between a keypress and hearing it — leave it open and every other knob
goes numb, gradually and without any other symptom. `-`/`+` nudge the cap
while it is open (the flash says `— throttle still off`) but never re-arm
it on their own: silently re-throttling in the middle of a cushion would
be the worst possible moment for it. Pressing `t` on a session with no
cap configured flashes rather than doing nothing.

`manifest.json.throttle.enabled` records the state the session ended on,
which is what a runaway buffer trace in the sparkline means after the
fact.

Above the frame, a live status line reports what the loop is doing right
now (`generating round 214 · 4.2s`, `throttled (buffer full)`,
`scoring`), which `performance.jsonl` cannot show because the round is
not finished yet.

### What changes when the TUI is on

- **Nothing prints.** Per-round status lines and event lines would
  scribble over the frame, so they are suppressed; every one of them is
  still written to `events.jsonl` and shown in the events box.
- **Worker diagnostics go to `<session>/worker.log`.** The generation
  subprocess deliberately routes its stdout into stderr (AudioCraft
  chatter, `Loading weights` progress bars, torch warnings), which it
  inherits from this process — straight onto the frame. With `--tui` that
  is redirected to a file instead. Look there when a model fails to load.
- **The alt-screen is used**, so your scrollback comes back on exit and
  the `rounds_written` / `stopped_by` summary prints to the normal
  screen. The terminal is restored from the exit path, from `atexit`, and
  from SIGTERM/SIGHUP. Only a `SIGKILL` can leave it in cbreak mode; if
  that happens, `reset` fixes it.
- `Ctrl-C` still works: the terminal is put in *cbreak*, not raw, so
  `ISIG` stays enabled and the existing interrupt path is untouched.

`--tui` needs an interactive terminal. Under `nohup`, a pipe or a
non-TTY, it prints `tui_unavailable:` and falls back to the normal
scrolling output rather than failing.

### What the stream scores

The round loop used to run the full arrhythmic scorer (29 fields) plus the
full arc scorer for every round **and every best-of-N candidate**: ~850 ms on
a 10 s round, against ~700 ms to generate it. Scoring, not generation, was the
budget. `macchina_ricordi/scoring/live.py` now computes exactly nine numbers — the six
the panel draws, plus `arc_tv4` (free, arc is derived from it),
`max_silence_gap_seconds` and the durations. Measured on a 10 s round:

| | was | now |
|---|---:|---:|
| `timbral_coherence` | 255 ms | 10 ms |
| `chroma_drift` | 97 ms | — |
| `arc_tv4` (four chroma-CQTs) | 190 ms | 190 ms |
| HPSS → `percussive_energy_ratio` | 186 ms | 186 ms |
| the other 26 arrhythmic fields | 94 ms | — |
| onsets → `onset_density` | 16 ms | 16 ms |
| silence → `silence_fraction` | 7 ms | 7 ms |
| **total** | **853 ms** | **450 ms** |

End to end on the 3060, a real `f05eba09` session at `candidates: 4`,
`completion-duration 6`: **2.80 s → 1.80 s per round**, of which ~0.55 s is the
generation itself. Scoring is still the larger half, but no longer by four to one.

Every retained field is bit-for-bit the number it was before — the geometry is
shared with the full scorers rather than re-derived, and a test asserts exact
equality against them. The biggest single item was not a metric at all but
waste: `timbral_coherence` only ever used `mfcc_cosine`, while
`compute_similarity_waveform` also built the spectral and rhythm vectors and
discarded them — and the rhythm vector runs the whole arrhythmic scorer on each
half. Fixing that made offline scoring faster too.

Two things worth knowing:

- **Sessions now log nine numbers where they logged thirty-eight.** A future
  stream-statistics pass cannot ask the 2026-08 questions of a 2026-08-06-or-later
  session. That is the deliberate trade — those questions were asked, and the
  answer was that most of the fields restate each other. Offline scoring is
  untouched: `score file`, `score dir` and the sweeps still compute everything,
  and that is where a corpus for a *new* question should come from.
- **`percussive_energy_ratio` is now 45% of what remains**, all of it in the
  HPSS that isolates the percussive component. `onset_strength_mean` reads the
  same force axis for free (ρ = +0.55, near-identical between-model variance
  share) if speed ever matters more than the exact field.

`novelty_raw` is the one dropped row still computed: it costs 0.03 ms, and
logging it beside `novelty_whitened` on identical rounds is the only way the
two stay comparable.

### Playing blind

`b` hides every per-round reading — the metrics block, the candidate table
and the arc/novelty/silence sparklines. What stays is everything that is not
a judgement about the music: buffer health, the control row, the guidance
block and the events. `--blind` starts a session that way. Nothing is turned
off: the readings are still computed and still written to
`performance.jsonl`. You just stop seeing them while you play.

This exists because of a specific hole in the stream-statistics corpus. All
3,523 rounds of it were played with the numbers on screen, so the strongest
relationship in the whole investigation — the player reaching for the novelty
knob exactly when whitened novelty was low — is circular. The number may have
prompted the reach. There is no way to tell from that corpus, and no amount
of re-analysis fixes it.

A blind session is the cheapest possible test: play by ear, and if you still
reach for `n`/`N` at the same moments, the reflex is yours rather than the
readout's. It is also the precondition for that blind A/B — the top research
follow-up. Until that runs, novelty stays a control and a reading — never an
automatic trigger.

## Metrics: generated vs playing

By default the metrics block describes **the round the generator just
finished**. That is right for steering — it is what your next control edit
builds on — and wrong for listening, because with a buffer you are hearing
something from a minute ago. Correlating a number with what is actually in
your ears needs the other one:

```bash
play monitor --follow playing
play stream --tui --follow playing      # or press `f` to toggle live
```

With `--follow playing` the metrics and candidate table describe **the
round currently coming out of the speakers**, several rounds behind the
generator. This is what you want while validating a metric by ear; it is
*not* what you want while steering, because the numbers lag your edits.

The frame always says which one it is showing, so a lagging reading can
never be mistaken for a live one:

```
  metrics · round 224 · last generated
  metrics · round 210 · ▶ NOW PLAYING, 4 behind — lags the generator
```

When the mode is on but playback position is unknown, it falls back to the
newest round and says so — `(nothing playing)` when the buffer has drained,
`or earlier · ▶ playing — buffer predates --history` when the buffer reaches
further back than `--history` rounds, so the label never claims to know a
position it only clamped to.

`f` in the TUI toggles the mode. It is display-only and never writes
`control.json`, so looking at a performance differently cannot change it.

## Monitoring a live session

`play monitor` is a read-only dashboard over a session's
`performance.jsonl` and `control.json`. It never writes to the session,
so it is safe to attach mid-performance, several can watch the same
session, and it works on finished ones too. Run it in a second terminal:

```bash
play monitor                     # the newest stream session, old home or new
play monitor --session 20260729T075418Z   # a session id or folder
play monitor --once              # one frame, no redraw loop
```

```
stream monitor — 20260729T075418Z                       round 737   4s ago
──────────────────────────────────────────────────────────────────────────
  hearing round 734 (3 behind) · an edit now lands at round 738
  last sample  round_737.wav  gen 2.7s / fresh 10.0s  0.27x rt
  buffer ahead    32.2s  ██████████··········  max 60s
──────────────────────────────────────────────────────────────────────────
  arc              0.290  ███·············
  timbral_coh     steady  (≥0.985)
  novelty          0.000  ················
  silence          0.000  ················
  onset_density    6.410  ████████········
  percussive       0.412  ██████··········
──────────────────────────────────────────────────────────────────────────
  candidates          arc   tail-sil    novelty
     #0           0.217      0.000     0.0001
   ◀ #2           0.290      0.000     0.0000
──────────────────────────────────────────────────────────────────────────
  arc           ▃▃▃▂▄▄▄▅▆▄▄▅▄▄▃▅▃▃▅▃▅▃▂▄▃▃▃▃▃▄▃▄▄▄▅▅▃▄▃▇▅▅▆▄▅█▂▃▄▃▃▂▁▁▁▃
  novelty       ▂▁▂▄▅▃▂▂▂▂▃▃▄▃▃▁▁▂▁▁▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▂▁▁▁▁▁▂█▅▂▁▁▁▁▁▂▁
  buffer        █▂▂▃▄▄▅▅▆▇▇▁▂▂▃▄▄▅▆▇▇█▂▂▃▄▄▅▆▆▇█▁▂▃▃▄▅▅▆▇▇▁▂▂▃▄▄▅▆▇▇█▁▂▃
                last 60 rounds
──────────────────────────────────────────────────────────────────────────
  control   temp 0.70  tau  9.0  completion 19.0  cands 8  novelty 0.01  top_k 70
  model     .dora/xps/8aa68bbe
  guidance  ⚠ prompt+completion 28.0s overruns 8aa68bbe's 15s horizon — expect gaps
            ⚠ temperature 0.70 is below the 1.4 floor for a model that gaps (95% of takes clean)
──────────────────────────────────────────────────────────────────────────
  events
    10:04:12  r712  model_swap: .dora/xps/1f225abf -> .dora/xps/8aa68bbe
  ! 10:06:28  r725  tail_rescue: round_724.wav tail is silent; re-prompting
    10:07:01  r731  throttle: ahead=60.3s >= max=60.0s, waiting (resume at 4
    10:07:44  r731  throttle: resumed (ahead=44.7s)



```

**It shows the last round the generator produced, not what you are
hearing.** Those are different things, and the gap is the point: with a
60s throttle buffer you are listening to audio finished a minute ago, so
a `control.json` edit stays inaudible for several rounds. The first line
spells that out —

```
hearing round 734 (3 behind) · an edit now lands at round 738
```

— so you know how long to wait before judging a change. The estimate
comes from the last row's `buffer_seconds_ahead` minus the row's age,
walked back through `playback_seconds`, and inherits every approximation
in that figure (see above); it is an instrument, not a clock. When the
buffer has drained since the last logged round the line instead reads
`playback drained`, which means the generator stopped, or is throttled or
between rounds.

The rest of the frame:

- **last sample / buffer ahead** — generation wall-clock against the net
  new audio the round contributes, as a realtime multiple. Above `1.00x
  rt` it flags `⚠ starving`.
- **meters** — six rows, one per axis the statistics corpus actually found:
  `arc_coherence_score`, `timbral_coherence` (as an alarm, below), novelty,
  `silence_fraction`, `onset_density` (*rate*) and `percussive_energy_ratio`
  (*force*). Bar lengths use fixed
  display ranges, so they are comparable across rounds; the number is
  always the truth. Empty rows mean the metric was not recorded —
  `--score-rounds` off, or `candidates: 1` with `novelty_weight: 0`,
  which is the case where novelty is never computed at all. `arc_tv4` and
  `arc_coherence_gated` are still written to the sidecar and deliberately
  not drawn: at ρ = 0.9994 and 98% identity with `arc_coherence_score`,
  a row for either is a second copy of the arc row. Four more rows were
  dropped in 2026-08 along with their computation — see
  [what the stream scores](#what-the-stream-scores).
- **`timbral_coh`** — an alarm, not a score. 63% of rounds sit above 0.985
  where the number carries almost nothing, and the field's usable range is
  entirely in its low tail. Above the line the row reads `steady`; below it
  you get `⚠ lurching` plus the number and a bar scaled to the tail. The
  raw value is untouched in the sidecar. This is a display experiment
  (2026-08-06) to be judged at the instrument — if the alarm reads worse
  than the number while playing, it goes back to being a plain meter.
- **candidates** — the per-candidate table best-of-N actually chose
  from, with `◀` on the winner. Worth watching: it is where you can see
  a selection go against the arc ranking, and why.
- **sparklines** — arc, novelty, silence and buffer over the last
  `--history` rounds (default 60), each self-scaled. The buffer trace
  showing a repeating sawtooth is the throttle's hysteresis working: it
  drains to the resume threshold, then generates back-to-back until full
  again.
- **control** — the live values from `control.json`, so you can confirm
  an edit was picked up, plus any pending one-shot command. Controls
  sitting at their off value (`novelty_weight 0`, `min_p 0`,
  `prompt_rate 1.0`) are hidden so the row carries what is actually
  engaged, and it wraps onto a second line rather than clipping — ten
  knobs no longer fit 80 columns, and the ones that used to fall off the
  end were the newest.
- **events** — the tail of `events.jsonl`, newest at the bottom, with `!`
  on warnings and `✗` on errors. The box is **always** `--events` lines
  tall (default 8, `0` hides it): it scrolls internally while the frame
  around it never changes size, so nothing shifts under your eyes as
  events arrive. Sessions recorded before the sidecar existed have no
  `events.jsonl`; for those the pane reconstructs what round rows do
  record — rescues, recaps, model swaps, one-shot commands, slow rounds —
  and labels itself `(derived from round rows)`. Throttle waits and
  playback errors are not recoverable that way, which is what the sidecar
  is for.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--session` | newest | Session directory to watch. |
| `--streams-dir` | every stream session | Watch the newest session in this folder instead. |
| `--interval` | `2.0` | Seconds between redraws. |
| `--history` | `60` | Rounds behind the sparklines. More rounds than columns are bucket-averaged, and the caption says how many rounds per column. |
| `--events` | `8` | Height of the events box in lines; `0` hides it. Fixed, so the frame never changes size. |
| `--follow` | `generated` | Which round the metrics block describes — see **Metrics: generated vs playing** below. |
| `--width` | terminal | Frame width, clamped to 64–110. Lines are clipped, never wrapped. |
| `--once` | off | Print one frame and exit — useful in a `watch` or a script. |

## CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--xp-path` | — | The model: a folder, any [model reference](models.md) such as `midwife@best`, or a [portable bundle](models.md#the-portable-bundle) directory. |
| `--checkpoint` | `latest` | `latest` / `final` / epoch number. |
| `--checkpoint-path` | — | Direct path to a `.th` file. |
| `--recipe` | — | Recipe to record in metadata; read from the model when omitted (`--profile` is an alias). |
| `--device` | — | `cuda` / `cpu` (auto). |
| `--seed-mode` | `uncond` | `uncond` = round 0 is unconditional; `excerpt` requires `--seed-audio`. |
| `--seed-audio` | — | WAV used as round 0 in excerpt mode. |
| `--max-rounds` | `0` | Cap on continuation rounds beyond the seed. `0` = infinite (stop via control file or Ctrl-C). |
| `--completion-duration` | `10.0` | Fresh audio appended per round; live-editable. Continuation request duration is this plus the actual prompt duration. |
| `--tau` | `5.0` | Initial tail length; live-editable. |
| `--temperature`, `--top-k`, `--top-p`, `--cfg-coef`, `--extend-stride` | env defaults | Initial samplers; live-editable. |
| `--seed` | — | Base seed; each round derives its own deterministic seed from `base + 1009·round + 17·take`. |
| `--output-dir` | `artifacts/sessions/<date>/` | Parent folder; the session is `<dir>/<run_id>/` (default: `stream_<run_id>/`). |
| `--play / --no-play` | `--play` | Toggle sounddevice playback. Falls back to silent with a warning if the `instrument` extra is missing. |
| `--tui` | off | Draw a single-pane dashboard in this terminal and take keyboard input. See **Playing from one terminal** above. |
| `--playlist` | — | Comma-separated model folders — XPs or [portable bundles](models.md#the-portable-bundle) — bound to the digit keys `1`–`9` (and `tab`) in `--tui`. Also the starting model when `--xp-path` is omitted. See **Model playlist keys**. |
| `--follow` | `generated` | With `--tui`: which round the metrics describe. See **Metrics: generated vs playing**. |
| `--novelty-calibration` | none (the checkout ships no calibration) | Whitening statistics for the novelty reading. Optional: build one and pass it explicitly (e.g. `--novelty-calibration analysis/novelty_calibration.json`); without it, novelty stays raw cosine. |
| `--no-novelty-calibration` | off | Measure novelty with raw cosine, ignoring any calibration file. |
| `--crossfade-seconds` | `0.02` | Micro-crossfade window at the splice between rounds. `0` disables (you'll get the raw splice — fine for percussion, may click on sustained tones). |
| `--max-buffer-ahead-seconds` | — | Throttle: pause generation when the playback buffer is at least this far ahead of the player, resume when it drains. Unset = no throttle. Initial value only — live-editable, and bound to `-`/`+` in `--tui`. See **Throttling** below. |
| `--resume-buffer-ahead-seconds` | `max - 10` | Hysteresis resume threshold. Only meaningful with `--max-buffer-ahead-seconds`. Live-editable; the `-`/`+` keys carry its gap along. |
| `--score-rounds / --no-score-rounds` | `--no-score-rounds` | Score each round with the arrhythmic-percussive scorer. |
| `--candidates` | `1` | Initial best-of-N candidate count; live-editable via `control.json` and bound to `c`/`C` in `--tui` (see the field above). |
| `--min-p` | `0` | Initial min-p threshold; live-editable (`min_p`). See the field above for the best-of-N interaction. |
| `--novelty` | `0` | Initial anti-déjà-vu weight; live-editable (`novelty_weight`). Try `0.3–0.5` with `--candidates 4+`. |
| `--stuck-weight` | `0` | Initial drone-guard ranking weight; live-editable (`stuck_weight`). Needs `--candidates` > 1. |
| `--stuck-kick` | `0` | Rounds in a row stuck before one hotter round; live-editable (`stuck_kick_rounds`). Try `2`. |
| `--motif-dir` | `<session>/motifs` | Motif bank for `pin` and recapitulation; share one across sessions for a common identity. |
| `--motif-period` | `0` | Recap every Nth round with a pinned motif blended into the prompt; live-editable. |
| `--motif-mix` | `0.5` | Motif share in the recap blend; live-editable. |
| `--prompt-tilt-low` / `--prompt-tilt-mid` / `--prompt-tilt-high` | `0` | Initial three-band EQ on the continuation prompt (dB); live-editable, bound to `j`/`k`/`l`. See **Prompt steering**. |
| `--prompt-rate` | `1.0` | Initial varispeed of the prompt window; live-editable, bound to `<`/`>`. |
| `--prompt-rate-mode` | `tape` | `tape` (pitch+tempo) or `tempo` (pitch-preserving phase vocoder); live-editable, no key. |
| `--no-tail-rescue` | off (rescue on) | Disable re-prompting around a silent tail; live-editable as `tail_rescue`. Pass it when the jump cut is worse than the fade for what you are making. |

## The model horizon

Every micro is trained on fixed-length segments (`dataset.segment_duration`,
15s for most of the zoo, 10s for the 25/50 Hz deep models). That number is
two different boundaries at once, and the stream now knows it: it is read
off the XP's own `.argv.json`, with
`macchina_ricordi/models/specs.py::MODEL_SPECS` as the fallback for a sig
whose folder has moved.

**Hard — the prompt.** AudioCraft caps a continuation prompt at
`min(duration, max_duration)`, so a prompt that *reaches* the trained
segment length raises `AssertionError: Prompt is longer than audio to
generate` and kills the round. Since `tau` is a live `←`/`→` control and
motif mixes ride the same path, that abort is reachable from the keyboard
on the 10s-segment models — one Phase 4 script run into it. `tau` is
therefore clamped to 1s below the horizon before anything reads it, and
the clamp is logged once as a `horizon_guard` event (once, not per round,
so a `tau` parked above the ceiling does not fill the events box).

**Soft — the completion.** Overrunning the horizon with the *whole*
window never crashes: past `max_duration` AudioCraft switches to its
windowed extension loop. It degrades instead. `73e1a314` is 97.7% gap-free
at 15s rounds and 68% at 17s — two extra seconds for a 28× jump in mean
silence — and `d371db2c` already degrades as it *approaches* 15s, its exact
trained length. Overrunning is sometimes the sound you want, so this stays
a warning in the `guidance` line of the frame and nothing more.

Rule of thumb: keep `tau + completion_duration` under the model's
`segment_duration`. Both figures are on the frame; the session's manifest
records the horizon it was played against.

### Temperature and top_k guidance

The same `guidance` block carries the two other pieces of tuning advice the
stream-statistics corpus actually supports. Both are surfaced, neither is
forced: `StreamControl` keeps its `temperature = 1.0` default and nothing
turns a knob on its own.

- **Temperature ≥ 1.4 on any model that produces silence at all.** This is
  the strongest practical result in the investigation, and with the player
  removed it is causal rather than merely observed — `f05eba09` goes from
  39.4% silence at T=1.0 to 4.1% at T=2.0. Temperature saturates around 1.6
  and moves four indicators within one round; it does *not* move arc.
  Midwife (99.1% of takes clean) is the only model clean enough to be exempt,
  and the warning names the model's clean share so you can see why it fired.
- **Avoid `top_k = 25` on `4ae63477` and `8aa68bbe`.** The silence cliff is
  sharp, sits exactly there rather than sloping, and exists in only two of
  the four models tested — 50 is already safe. Phase 3's observational
  "top_k ↓ → silence" does not generalise beyond those two.

The numbers behind both live in `macchina_ricordi/models/specs.py`
(`clean_share`, `top_k_cliff`), drawn from the stream-statistics
investigation's 2026-08 parameter-impact run.

## Tips

- **Seed selection matters.** Round 0
  sets the trajectory. In-distribution seeds (uncond from the XP itself, or
  an excerpt from its training set) get "weirded-out variations". Out-of-
  distribution seeds get cut off and replaced by the prior.
- **τ is the memory knob.** Short τ (≈1s) feels like cuts; long τ (≈8s) feels
  like continuity. There is no right answer; both are gestures.
- **Reroll is composer-like, not safety-like.** If a round doesn't fit, set
  `command: reroll` rather than starting a new session. The prompt window is
  preserved.

## Smoothness — why playback feels continuous

Each round's first τ seconds are an EnCodec re-encoding of the previous
round's tail (the model already designed the overlap).
The playback worker honours that:

1. **Trim the echo.** It skips the first τ seconds of every continuation
   round before sending audio to the speakers, so you don't hear `tail[N]`
   immediately followed by an EnCodec-degraded copy of itself.
2. **Micro-crossfade the splice.** A 20ms (default) overlap-add at the
   splice prevents sample-level clicks caused by EnCodec's small phase
   shift. Configurable via `--crossfade-seconds`; `0` disables.

For silence-gap-free playback, **pick `completion_duration` such that
generation time per round < `completion_duration`**. That's the net new
audio each round adds to the timeline after the echo is trimmed; the
generator has exactly that much wall-clock to render the next round
before the player runs dry.

Example: with `--completion-duration 10 --tau 5`, playback consumes
~10s per continuation round and the model requests ~15s including the prompt.
If that request takes 12s on the hardware you're using, you'll underrun within
a few rounds — increase `--completion-duration` or shrink `--tau`.

**Caveat — generation cost is step-shaped, not linear.** AudioCraft
chunks generation by `max_duration` (10s for `micro-unconditional`) and
strides by `extend_stride` (default 8s) between chunks. So crossing a
chunk boundary in `tau + completion_duration` adds roughly one whole chunk's
wall-clock, not a proportional slice. On the reference 3060,
`--completion-duration 12 --tau 3` (15s total, 2 chunks) is the empirical
sweet spot; `--completion-duration 15 --tau 5` (20s total, 3 chunks)
consistently underruns. Bumping
duration to "fix" an underrun can backfire — the next safe stops are
smaller (e.g. `--tau 2`) or *much* bigger (e.g. `--completion-duration 21`
which gives 3-chunk gen plenty of playback room).

The loop prints `generation_slow_warning: round K took Xs but completion
duration is Ys` the moment it detects this. If you see it, adjust the
control file before silence catches up. If audio does run out and ALSA
reports xruns, the playback worker disables itself for the rest of the
session — generation keeps going, you can still see rounds being written
to disk, but you won't hear them until you restart with bigger
`completion_duration`. Use `latency='high'` is built in, so the ALSA buffer
already absorbs ~100ms of jitter; for steady-state slowness only the
budget matters.

## Throttling — keeping the buffer from running away

Since the persistent stream worker keeps the model loaded across rounds,
generation often runs *much* faster than playback. Without intervention,
the loop generates rounds back-to-back as fast as it can, and the
`ahead=…` figure climbs without bound — you'll see WAVs pile up on disk
minutes before they're played. The flip side is also a problem: a huge
buffer means edits to `control.json` (a new `temperature`, a `model_xp`
swap, a `reroll`) won't be *heard* until the buffer drains down to that
edit's round.

`--max-buffer-ahead-seconds` caps the buffer. When `buffer_seconds_ahead`
reaches the cap, the loop pauses generation (1-second polls) until the
buffer drains back to `--resume-buffer-ahead-seconds` (default:
`max - 10`). Then it resumes generating. Hysteresis prevents fast
on/off oscillation at the threshold.

```bash
python -m macchina_ricordi.cli play stream \
  --xp-path .dora/xps/0444687c \
  --completion-duration 12 --tau 3 \
  --max-buffer-ahead-seconds 60 \
  --resume-buffer-ahead-seconds 45
```

The output looks like:

```
seed: gen=4.2s budget=12.0s ahead=+12.0s
round 1: gen=4.1s budget=12.0s ahead=+22.8s
round 2: gen=4.0s budget=12.0s ahead=+30.7s
round 3: gen=4.0s budget=12.0s ahead=+38.5s
round 4: gen=4.0s budget=12.0s ahead=+46.4s
round 5: gen=4.0s budget=12.0s ahead=+54.3s
throttle: ahead=62.1s >= max=60.0s, waiting (resume at 45.0s)...
throttle: resumed (ahead=44.7s)
round 6: gen=4.0s budget=12.0s ahead=+52.6s
throttle: ahead=60.3s >= max=60.0s, waiting (resume at 45.0s)...
throttle: resumed (ahead=44.5s)
round 7: gen=4.0s budget=12.0s ahead=+52.4s
...
```

Tradeoffs:

- **Bigger `max`** = more cushion for a `model_xp` swap (the new model
  can take 10–60s to load; a 60s buffer absorbs that without the
  player starving). But your control-file edits take that long to be
  heard.
- **Smaller `max`** = more responsive instrument; edits land in a few
  rounds. But less room for the swap-load to disrupt playback.
- A reasonable default for live play with model swaps:
  `--max-buffer-ahead-seconds 60 --resume-buffer-ahead-seconds 45`
  (15s of hysteresis) — large enough for a micro-XP swap, small enough
  for `temperature` / `tau` tweaks to be audible within a minute.
- You do not have to size the cap for the worst swap you might make.
  Run a small, responsive buffer, and press `t` to open the throttle for
  the minute before a swap — see
  [Banking a cushion](#banking-a-cushion-for-a-model-swap).
- `command: stop` is honoured during the wait — you can stop a session
  mid-throttle without waiting for the buffer to drain.

### Retuning the buffer live

Both thresholds are ordinary control fields, and the flags above only set
their initial value. They are re-read **on every poll of the throttle
wait**, not frozen when the wait began — because the buffer is precisely
what stands between a control edit and hearing it, so a wait of a minute
is the worst possible place for a change to the buffer size to be
deferred. In the TUI, `-` and `+` move the cap in 5s steps and carry the
current hysteresis gap along; by hand, edit
`max_buffer_ahead_seconds` / `resume_buffer_ahead_seconds` in
`control.json`.

What each direction does to a wait already in progress:

- **Raising the resume threshold** above the current buffer ends the wait
  on the next poll. So does raising the cap, setting it to `null`, or
  setting `throttle_enabled: false` — the `t` key, and the case that
  matters most, since you reach for it precisely when a wait is what is
  in your way (`throttle: throttle switched off during wait; resuming`).
- **Lowering the cap** *extends* the current wait, which is the point:
  the buffer drains to the new, smaller size now instead of staying at
  the old size for another full round. Every knob you touch afterwards is
  audible sooner.

A change picked up mid-wait logs a `throttle_retuned` event with both new
values, so a session's responsiveness over time is recoverable from
`events.jsonl`.

`manifest.json.throttle` records the thresholds the session *ended* on —
they are live controls, so these can differ from `initial_control`.

## Live model swap

Set `model_xp` in `control.json` mid-session to a different Dora XP
folder — or, in the TUI, press the digit key for it after starting with
[`--playlist`](#model-playlist-keys), banking a buffer cushion with `t`
first. The persistent worker reloads that
XP for the next continuation round, paying one model-load cost (≈10–20s for micro-unconditional on a
3060, longer for the bigger reshape models). Later rounds reuse the loaded model.
The audio character changes audibly at that round — frame it as a deliberate
scene change.

Each row in `performance.jsonl` records the active `model_xp` per round,
so a session that hopped through several voices is queryable after the
fact. `manifest.json` adds `swap_count` and `final_xp_path` for an
at-a-glance summary.

## Limitations (v0.5)

- Mid-session `checkpoint` selection is fixed at session start; only
  `model_xp` is live-editable. To change checkpoint epochs, restart the
  session.
- Model-swap cost is paid synchronously (the "seam" variant). A future
  "smear" variant that pre-loads the next XP in parallel with rendering the
  current round is deferred.
- Not every control has a key. `min_p`, `motif_*`, `cfg_coef` and
  `prompt_rate_mode` are still editor-only; see
  [the key table](#playing-from-one-terminal---tui).
- The performance log isn't yet rendered as a curve. A future
  `play stream-curve <run_id>` could draw a sparkline from it.
