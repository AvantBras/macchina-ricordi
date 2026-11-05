# Listening Dossiers

Use listening dossiers when you want a text-native description of what happens
inside one audio clip. Scoring answers "how arrhythmic/percussive is this?".
Listening answers "what happens over time, and what evidence supports that
reading?".

The default implementation is deterministic and dependency-light. It uses the
same DSP/scoring stack as the rest of the project. Optional semantic probes can
add CLAP audio-text hints when the `semantic-listening` extra is installed.
Optional captioner witnesses can add a separate audio-language model account
when the `captioner-witness` extra is installed. MERT and neural archive search
are still out of scope.

**Trust boundary:** captioners currently allow model-repository Python execution by default
(`trust_remote_code=True`). Only use trusted model publishers, especially with a custom
`--captioner-model-id`. Pass `--no-captioner-trust-remote-code` to disallow it; models needing
custom code may then fail to load. See [security and trust boundaries](../SECURITY.md).

Install both optional neural listeners when you want the richest reports:

```bash
pip install -e ".[semantic-listening,captioner-witness]"
```

## Single File

```bash
python -m macchina_ricordi.cli listen file /path/to/audio.wav
```

By default this prints a Markdown listening dossier to stdout. The report
includes:

- source and analysis metadata
- a short agent-readable summary
- whole-clip perceptual labels and score evidence
- coarse form regions
- fixed timeline windows
- a compact ASCII band/onset sketch
- warnings, when present

## Sidecars

Write canonical JSON and Markdown sidecars next to the audio file:

```bash
python -m macchina_ricordi.cli listen file /path/to/audio.wav --sidecars
```

For `audio.wav`, this writes:

- `audio_listening_report.json`
- `audio_listening.md`

You can override either path explicitly:

```bash
python -m macchina_ricordi.cli listen file /path/to/audio.wav \
  --output artifacts/listening/audio.json \
  --markdown artifacts/listening/audio.md
```

Explicit paths and `--sidecars` can be combined. Explicit `--output` and
`--markdown` win for their respective formats.

## JSON Output

Print the canonical JSON dossier to stdout:

```bash
python -m macchina_ricordi.cli listen file /path/to/audio.wav --json
```

The top-level JSON shape is:

```yaml
schema: macchina_ricordi.listening_dossier.v0
analysis:
audio:
provenance:
whole_clip:
timeline:
sketches:
notable_events:
summary:
semantic_probes:  # optional, only with --semantic-probes
captioner_witness:  # optional, only with --captioner-witness
warnings:
```

`timeline.windows` keeps the fixed-window observations. `timeline.regions`
groups adjacent windows into coarse form regions with labels and evidence.
`notable_events` is always present (possibly empty): up to a dozen
moment-level landmarks, each with a start time, an optional end time, a `kind`
(`onset` for a strong onset, `silence_start` / `silence_end` for silences of at
least 0.5 s), a normalized `strength`, and a one-line `detail`.

## Analysis Options

Defaults match the higher-fidelity scoring baseline where possible:

- `--sr 32000`
- `--hop-length 512`
- `--res-type soxr_hq`
- `--tempo-method tempogram`
- `--window-seconds 2.0`
- `--window-hop-seconds 1.0`

For a faster pass:

```bash
python -m macchina_ricordi.cli listen file /path/to/audio.wav \
  --sr 16000 \
  --res-type soxr_qq \
  --tempo-method autocorr
```

## Semantic Probes

Semantic probes add optional CLAP-style audio-text comparisons to a dossier.
They are meant as hints for text agents, not ground truth:

```bash
pip install -e ".[semantic-listening]"
python -m macchina_ricordi.cli listen file /path/to/audio.wav --semantic-probes
```

The first backend is Microsoft CLAP through `msclap`. Model weights may download
the first time probes run. The base install never imports or downloads the model.
Archive indexing initializes the backend once and caches the text prompt
embeddings for the curated vocabulary, so `listen index --semantic-probes`
does not recompute the same text side for every file.

Useful options:

- `--semantic-backend msclap`
- `--semantic-model-version 2023`
- `--semantic-device auto`
- `--semantic-model-path /path/to/model.pt`

The JSON report adds `semantic_probes` with backend metadata, the curated
macchina_ricordi vocabulary version, all probe scores, strong findings, weak negative
controls, and uncertain readings. Markdown renders a compact section. Negative
controls such as `stable_beat`, `clear_melody`, or `sung_voice` are surfaced as
uncertain when they score high, because experimental music can fool semantic
models in interesting ways.

## Captioner Witness

Captioner witnesses add an optional audio-language model summary. The caption is
stored separately from deterministic evidence and cross-examined before it is
shown to agents:

```bash
pip install -e ".[captioner-witness]"
python -m macchina_ricordi.cli listen file /path/to/audio.wav --captioner-witness
python -m macchina_ricordi.cli listen file /path/to/audio.wav --semantic-probes --captioner-witness
```

Two backends are available; each is an independent witness and both go through
the same claim cross-examination:

- `--captioner-backend qwen2-audio` (default) — model id
  `Qwen/Qwen2-Audio-7B-Instruct`, Apache 2.0.
- `--captioner-backend audio-flamingo-3` — model id defaults to
  `nvidia/music-flamingo-hf` (music-specialized Audio Flamingo 3, 8B; NVIDIA
  OneWay Noncommercial license, research use only). Requires `transformers>=5`.
  Music Flamingo is trained on songs and may hallucinate melody/harmony on
  arrhythmic material even more than Qwen — that is what the cross-examination
  warnings are for.

Shared flags:

- `--captioner-model-id ...` (default: backend-specific)
- `--captioner-device auto`
- `--captioner-dtype auto`
- `--captioner-quantization auto`
- `--captioner-max-new-tokens 220`
- `--captioner-max-audio-seconds 120`
- `--captioner-prompt-profile experimental_texture_v1`
- `--captioner-trust-remote-code` / `--no-captioner-trust-remote-code`

Model weights may download the first time the witness runs. The base install
never imports or downloads the captioner model.

Qwen2-Audio is large. A real smoke on this workstation downloaded about 16 GB
of Hugging Face model cache. On CUDA, `--captioner-quantization auto` resolves
to `bnb-4bit` so the model has a better chance to fit on consumer GPUs. On CPU,
it resolves to `none`. Use `--captioner-quantization none` to request the older
full/full-ish precision loading path. 4-bit reduces model-weight memory, but it
does not make Qwen2-Audio lightweight.

Use shorter limits for quick checks:

```bash
python -m macchina_ricordi.cli listen file /path/to/audio.wav \
  --captioner-witness \
  --captioner-device cuda \
  --captioner-dtype float16 \
  --captioner-quantization bnb-4bit \
  --captioner-max-audio-seconds 2 \
  --captioner-max-new-tokens 48
```

The JSON report adds `captioner_witness` with backend metadata, prompt profile,
prompt text, compact summary, raw response, extracted claim flags, agreement
ratings, quantization mode, and warnings. Markdown renders a `Captioner Witness`
section after semantic probes when present.

## Interrogation (`listen ask`)

`listen ask` sends free-form questions to the captioner backend instead of the
fixed witness prompt, so agents can interrogate a clip beyond the one-shot
caption:

```bash
python -m macchina_ricordi.cli listen ask sample.wav "What changes at the halfway point?"
python -m macchina_ricordi.cli listen ask sample.wav "Is there a steady pulse?" "How dense is the texture?"
python -m macchina_ricordi.cli listen ask sample.wav "Describe the transients" --captioner-backend audio-flamingo-3
```

Multiple questions in one invocation reuse the loaded model. Each answer goes
through the same claim extraction and cross-examination as captions, using a
fresh deterministic dossier as evidence. By default every exchange is appended
to an interrogation transcript sidecar (`<stem>_interrogation_report.json`)
next to the audio file; pass `--no-sidecar` to skip it, `--output` to redirect
it, and `--json` to print the report as JSON. The model reloads on every
invocation — if that latency hurts, a resident-model server is the planned next
step.

The witness is deliberately suspicious of conventional claims. If the captioner
claims a steady beat against unstable pulse evidence, mentions voice against
weak or absent `sung_voice` probes, or names melody, harmony, instruments, or
genres without supporting evidence, the dossier marks those claims as
unsupported, unverified, or captioner-only.

## MusicGen Outputs

MusicGen generation can write listening sidecars automatically after the WAVs
are saved:

```bash
python -m macchina_ricordi.cli play generate --listen
python -m macchina_ricordi.cli play continue --prompt-audio seed.wav --listen
python -m macchina_ricordi.cli play sweep --listen
```

For `sample.wav`, MusicGen writes:

- `sample_listening_report.json`
- `sample_listening.md`

`--listen` requires saved WAVs, so `--listen --no-save-output` is rejected
before backend generation starts. Listening failures after generation are
reported but do not delete or invalidate the saved audio.

## Archive Memory

Build a deterministic archive index with listening summaries, compact DSP
vectors, nearest neighbors, novelty scores, and best-effort MusicGen metadata:

```bash
python -m macchina_ricordi.cli listen index artifacts/runs \
  --output artifacts/listening_index.json
```

Missing listening sidecars are analyzed and written by default. Use
`--read-only` to only consume existing `*_listening_report.json` files, or
`--refresh` to re-analyze and overwrite sidecars.

When archive indexing creates or refreshes a listening sidecar, the same loaded
waveform is reused for archive similarity vectors. Existing sidecars still avoid
full dossier creation and only load audio for vector extraction.

Archive indexing can also enrich sidecars with semantic probes:

```bash
python -m macchina_ricordi.cli listen index artifacts/runs \
  --semantic-probes \
  --output artifacts/listening_index.json
```

With `--read-only --semantic-probes`, archive indexing only reads existing
semantic data and never creates missing reports.

Archive indexing can also run the captioner witness. The backend is initialized
once and reused across files:

```bash
python -m macchina_ricordi.cli listen index artifacts/runs \
  --captioner-witness \
  --captioner-quantization bnb-4bit \
  --output artifacts/listening_index.json
```

With `--read-only --captioner-witness`, archive indexing only reads existing
witness fields and never creates or mutates sidecars. `--refresh --captioner-witness`
re-analyzes and overwrites sidecars.

Query the same archive memory:

```bash
python -m macchina_ricordi.cli score similar sample.wav artifacts/runs
python -m macchina_ricordi.cli score novelty artifacts/runs --limit 20
python -m macchina_ricordi.cli listen portrait .dora/xps/<sig>
```

`score similar` and `score novelty` (numbers, so they sit under `score`; they were `listen similar` / `novelty`,
which still work) print Markdown by default and accept
`--json`. `listen portrait` scans generated-audio metadata for a matching XP
path, XP signature, checkpoint path, run id, sweep id, model label, or
workspace, then summarizes repeated labels, representative samples, and
outliers. Archive memory uses the deterministic DSP vectors `score sim` compares (one engine,
`scoring/similarity.py`) for neighbors and novelty; semantic probe and captioner witness fields are attached as report
hints only.

## Python API

The CLI is the primary interface, but the listening package exposes the same
building blocks for scripts:

```python
from macchina_ricordi.listening.archive import build_archive_index
from macchina_ricordi.listening.captioner import attach_captioner_witness
from macchina_ricordi.listening.features import analyze_audio
from macchina_ricordi.listening.semantic import attach_semantic_probes

dossier = analyze_audio("sample.wav")
index = build_archive_index("artifacts/runs", read_only=True)
```

Neural helpers stay lazy: `attach_semantic_probes()` imports/initializes the
semantic backend only when called, and `attach_captioner_witness()` does the
same for the captioner backend.

## Notes

- Dossiers work best on short clips or already-segmented tracks.
- Long recordings should usually go through [Segmentation](segmentation.md)
  first.
- JSON is the canonical machine-readable sidecar. Markdown is a human/agent
  view rendered from the same dossier.
- For fast deterministic sweeps, prefer `--sr 16000 --res-type soxr_qq`; keep
  optional semantic and captioner passes for smaller shortlisted sets when
  model runtime matters.
