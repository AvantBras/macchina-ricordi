# Voice Loops

`loop` finds sung phrases in a song that loop cleanly and renders them. It automates the "pick a vocal phrase in
Audacity and repeat it" edit: the machine proposes a shortlist, and you choose by ear.

## Find candidates

```bash
python -m macchina_ricordi.cli loop find song.mp3 artifacts/loops/song
```

What it does:

1. **Splits the song** into `vocals.wav` + `no_vocals.wav` with Demucs (`htdemucs`), cached in
   `<output_dir>/stems/`. Demucs lives in `.venv-musicgen`, so `find` runs it there as a subprocess. It uses the CPU by
   default (about a minute for a 3–4 minute song), so it never competes with a training run for the GPU. Pass
   `--device cuda` when the GPU is free.
2. **Measures the bar length from the instrumental.** The drums repeat exactly while the voice does not, so the
   onset autocorrelation of `no_vocals.wav` gives the beat period to a fraction of a millisecond. A mild preference
   for tempos near 120 BPM breaks near-ties with dotted riffs (Mauro Picotto's Proximus otherwise reads 92 BPM
   instead of 138). Check the tempo `find` prints: pass `--bpm` if it is still off (half, double, or 2/3 time), and
   `--beats-per-bar` for anything other than 4/4.
3. **Finds phrase starts**: points where the voice comes back after a breath of at least `--min-breath-ms`.
4. **Proposes loops** that start `--pre-roll-ms` before a phrase (so the first word keeps its attack) and last exactly
   `--bars` bars (1, 2, 4, 8 by default). Whole bars mean the beat never stumbles at the seam.
5. **Ranks them**: mostly-voice loops that end while the singer is silent (no word cut in half) and start after a clear
   breath come first. Starts closer than one bar apart are deduplicated.

It writes:

- `candidates/<rank>_<start>_<bars>bar.flac`: each shortlisted loop repeated to about `--seconds` (60 by default),
  for listening. `04_0m42s56_8bar` means rank 4, starts at 0:42.56, 8 bars long.
- `loop_report.json`: tempo, bar length, and every shortlisted candidate (start, length, voice fraction, voice level
  at the loop end, breath before the phrase).

The ranking weights were tuned on one song. Treat the order as a suggestion, and listen to the whole shortlist.

### Voiced sections only

To keep only loops that are sung all the way through (no instrumental break inside), and one loop per section:

```bash
python -m macchina_ricordi.cli loop find song.mp3 artifacts/loops/song --bars 4,8 --min-voice 0.8 --max-gap-ms 1800 --no-overlap --top 50
```

- `--min-voice 0.8`: the voice is present at least 80% of the loop.
- `--max-gap-ms 1800`: no stretch without voice longer than 1.8 s (about one bar at eurodance tempos). Gaps
  between sung lines run from 0.2 to 1.6 s; instrumental breaks run from 2 s up.
- `--no-overlap`: shortlisted loops never share audio, so each one is a distinct section of the song.

These values were calibrated on Hit My Heart, where loops that sound voiced to the ear measured 87–98% voice with gaps
of 1.6 s at most. The report lists each candidate's `voice_fraction` and `longest_gap_seconds`.

Re-running `find` into the same folder reuses the stems and replaces the previous auditions. Stems that belong to a
different song are refused; `--force-separate` re-runs Demucs.

## Render the one you like

```bash
python -m macchina_ricordi.cli loop render artifacts/loops/song/loop_report.json song_loop.flac --pick 4 --seconds 115
python -m macchina_ricordi.cli loop render artifacts/loops/song/loop_report.json song_loop.flac --start 42.6 --bars 8 --repeats 8
```

`--pick N` renders candidate N from the report (default 1). `--start` + `--bars` renders your own choice using the
song's measured bar length, which is handy to fix an Audacity loop whose length is a few milliseconds off the grid.
`--bars` also works with `--pick` to change a candidate's length.

`--layer` chooses what gets looped:

| Layer | Result |
|---|---|
| `mix` (default) | The song itself |
| `vocals` | The voice alone (Demucs stem) |
| `instrumental` | The band alone |
| `stuck` | The song plays normally up to the loop start, then the voice keeps repeating while the band carries on to the end. Always runs to the end of the song, so `--seconds`/`--repeats` don't apply |

Each render writes `<output>_report.json` next to the audio file.

## Seams

Every repeat starts exactly one loop length after the previous one. At each seam, the audio that followed the loop
end in the original fades out over `--fade-ms` (10 ms), while the loop start fades in. That removes clicks without
shifting the groove. The fade is linear (equal-gain): both sides of a bar-aligned seam carry the same drum hits, and
an equal-power fade of such correlated audio would overshoot by up to 3 dB and clip loud masters.

Loud masters often decode from MP3 slightly above full scale (peaks of 1.00–1.09). Those samples are clipped when
written to 16-bit FLAC/WAV, the same as any other export would do.

Output defaults to 44.1 kHz and keeps the source channels (`--mono` to downmix).

## Feeding a loop to training

A rendered loop is an ordinary audio file. Put it in a corpus folder like any other clip and slice or prepare it as
usual (see [Augmentation](augmentation.md) for speed variants and [MusicGen Training](musicgen.md) for `data
prepare`). A model trained on a loop learns the seam too, so a bar-exact loop is worth the extra step.

## Helper scripts

`scripts/loops/` holds the analyses behind the loops made so far. Run them with `.venv/bin/python` from the repo
root; each one's `--help` explains what it measures.

| Script | Use it to |
|---|---|
| `match_manual_loop.py LOOP SONG --report R` | Work out a hand-made loop (period, source start, ms error per seam, clipped first word) and print the bar-exact `loop render` fix |
| `vocal_map.py R [--anchor S] [--around T]` | See where the voice is (one row per two bars, phrase starts in bars from an anchor), or zoom on one moment |
| `compare_starts.py R --bars N [--reference S] --starts a,b` | Choose a start where the voice never pauses: where a sung line recurs, and seam quality for each start |
| `batch_voiced_loops.py SRC OUT [--dry-run]` | Voiced-only loops for a whole folder, with a CLAP check that the "voice" isn't a synth, and a `loops_manifest.json` |
| `verify_loops.py OUT_OR_FILES` | Check rendered loops: length, beat drift across repeats, and whether clipping comes from the source or the render |

`R` is the `loop_report.json` that `loop find` writes. The usual agent workflow is:
1. `match_manual_loop.py` to check a hand-made Audacity loop and get the bar-exact
   `loop render` fix; 2. `vocal_map.py` to see where the voice sits before picking a
   start; 3. `compare_starts.py` to choose a start where the voice never pauses;
   4. `verify_loops.py` on the renders to confirm length, beat drift, and whether
   clipping comes from the source or the render. `batch_voiced_loops.py` is for
   turning a whole folder into voiced-only loops in one pass.
