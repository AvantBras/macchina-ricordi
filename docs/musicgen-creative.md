# Creative tools — duet, variations, min-p

Inference-side tools for musical interest, built on the fastgen speed
headroom.
For the stream-side creative controls (novelty-aware best-of-N, motif
recapitulation, the `pin` command) see [musicgen-stream.md](musicgen-stream.md).

## Duet — two models playing together

```bash
python -m macchina_ricordi.cli play duet \
    --xp-a .dora/xps/2b0dc2a8 --xp-b .dora/xps/ed386554 \
    --rounds 40 --round-duration 8 --tau 2 --candidates 2 \
    --temperature-a 1.1 --temperature-b 1.4 \
    --listen-a 0.3 --listen-b 0.5 --seed 11
```

Offline renderer (not real-time playback): loads both XPs side by side —
each arms its own fp16 + CUDA-graph fast path — and every round each voice
continues a **blend of its own tail and its partner's tail**:

    prompt_A = (1 - listen_a) * tail_A + listen_a * tail_B

Both voices read the *previous* round's state, so neither leads. Cross-codec
conditioning is free: each model encodes the shared audio with its own
EnCodec, which is how the 25 Hz percussive LM can "hear" the 50 Hz deep
voice at all. `--candidates N` renders N takes per voice per round and keeps
the least silent (batching is near-free).

Output is a [session](sessions.md), `artifacts/sessions/<date>/duet_<utc>/`:

```
voice_a.wav    # mono stem, seamless (60ms linear crossfades at junctions)
voice_b.wav
mix.wav        # stereo master: A panned -0.35, B +0.35, gain 1.0/0.8, peak-safe
rounds.jsonl   # per-round silence fractions, picks, timings
manifest.json
```

Knobs worth playing:

- `--listen-a/--listen-b` are the musical heart. `0` = two independent
  streams; `0.3/0.5` = the percussive voice follows the arc voice more than
  the reverse (recommended start); `0.8+` = call-and-response so tight it
  approaches unison. Asymmetric values create a leader/follower feel.
- Per-voice `--temperature/--top-k/--min-p`: the intended pairing is the
  50 Hz deep at ~1.1 (its arc regime) with the 25 Hz LM at ≥1.4 (below that
  it silence-collapses).
- `--round-duration` must fit *both* models' trained segment lengths — the
  shorter one governs, and each model's cap is its own. The deep and 25 Hz
  voices used here both train at 10 s, so 8 s leaves headroom; a model with
  a tighter cap drives the round down to it.
- Speed on the 3060: a round pair with best-of-2 on both voices renders in
  ~1.3 s for 6 s of new audio (~4.6× realtime) — a 10-minute duet takes
  ~2.5 minutes.

The stems stay sample-locked (both voices contribute identical-length
rounds), so you can remix them in a DAW instead of using `mix.wav`.

## Variations — re-performing a piece

```bash
python -m macchina_ricordi.cli play variations \
    --input stream_20261005T141908Z#42 \
    --xp-path .dora/xps/2b0dc2a8 -n 4 --temperature 1.3 --seed 7
```

`--input` is a take (`<session>#<n>`, recorded as the new session's parent) or any audio file. Encodes the
input with the model's own EnCodec, **freezes the coarsest RVQ
book(s)** — book 0 carries the structural skeleton — and resamples the
remaining books: same piece, different performance. Works because MusicGen's
generation loop only writes over unknown positions, so pre-seeded tokens
both survive *and* condition the sampling around them. The tool asserts the
frozen books come back byte-identical (`frozen_books_preserved: True`).

- `-n` variations render in one batch: 4 × 8 s in ~1.1 s on the 3060.
- `--freeze-books 1` (default) re-voices timbre/detail over fixed structure;
  `--freeze-books 2` (of the 4 books at 50 Hz) stays closer to the original;
  on the 25 Hz stack (8 books) try 2–3.
- `--keep-intro-seconds 1.0` freezes *all* books for the first second so
  variations open identically before diverging.
- `--temperature` is the "how different a performance" knob (1.2–1.5).
- `reference_roundtrip.wav` is the codec round-trip of the input — compare
  against *that*, not the raw input, to hear what the variation changed
  versus what the codec changed.
- Measured character on deep@60: variations correlate with the reference at
  ~0.35 at the waveform level (unrelated renders sit near 0) while staying
  clearly distinct from one another.

Inputs don't have to be generated audio: a training-set segment, or a round
from the *other* model (25 Hz take re-performed with the 50 Hz codec's
palette) both work — anything the codec can encode.

## min-p sampling <a name="min-p"></a>

`--min-p` on `play generate`, `play stream` (live-editable), duet
(per voice) and variations. Keeps tokens with `p >= min_p * p_max` computed
*after* temperature, then renormalises (arXiv:2407.01082). The truncation
adapts to the model's per-step confidence — confident steps stay narrow,
uncertain steps stay wide — which is what fixed top-k can't do.

**Where it belongs: high temperature.** That's its designed regime and the
project's pain point (mega/deep models need temp ≥ 1.4 to avoid silence
collapse, and top-k 250 lets junk through up there). Measured at temp 1.5 /
min-p 0.08 on deep@60: four fully distinct, healthy samples.

**Where it bites: low temperature + best-of-N.** In confident (well)
passages at temp ≈ 1.1, min-p can leave a *single* surviving token per step
— generation goes near-greedy and all `candidates` takes collapse to the
same audio, silently defeating both best-of-N and novelty selection.
Measured on deep@60: with min-p 0.05 at temp 1.1, several rounds produced
four identical candidates; the identical run without min-p diverged every
round. Rule of thumb: `min_p` > 0 pairs with temp ≥ 1.3, or stays 0 when
best-of-N is doing the steering.

Implementation: an instance patch of `_sample_next_token`
(`macchina_ricordi/musicgen/sampling.py`), zero vendored-code edits; `min_p <= 0`
delegates to the stock sampler, so it is inert unless requested.
