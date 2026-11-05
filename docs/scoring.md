# Scoring

`score` gives numbers; `listen` gives text and `judge` gives verdicts ([the split](cli.md)). Perceptual words for a
clip (`listen describe`, `diff`, `compare`) moved to `listen` in October 2026; `score describe` and friends still
work and say where they went. Similarity has one engine: `score sim` compares two files, and `score similar` /
`score novelty` compare a file against a listening archive with the same vectors.

Use scoring when you want a quick arrhythmic/percussive signal for a clip or a whole library.
If you want a richer text report with form regions, timeline windows, and
evidence, use [Listening Dossiers](listening.md).

The score on this page measures busyness and transients — it is blind to whether the
music *develops*. For development, section structure, and for finding out which metric
actually matches your ear, see
[Structure Scoring & Metric Calibration](scoring-structure.md).

## Single File

```bash
python -m macchina_ricordi.cli score file /path/to/audio.wav --json
```

Default scoring uses the higher-fidelity baseline:

- `--sr 32000`
- `--res-type soxr_hq`
- `--tempo-method tempogram`

For a faster pass you can still switch to the lighter prototype settings:

```bash
python -m macchina_ricordi.cli score file /path/to/audio.wav \
  --sr 16000 \
  --res-type soxr_qq \
  --tempo-method autocorr
```

## Directory Scoring

```bash
python -m macchina_ricordi.cli score dir /path/to/library \
  --recursive \
  --output artifacts/scores/library.csv
```

Useful flags:

- `--recursive` includes subdirectories.
- `--extensions wav,flac,mp3` narrows file discovery.
- `--quiet` disables the progress bar.

The output CSV is sorted by descending score and includes the full reward component breakdown.

## Calibration

Use calibration to compare known steady clips against known arrhythmic clips.

```bash
python -m macchina_ricordi.cli score calibrate \
  refs/steady \
  refs/arrhythmic \
  --recursive \
  --output artifacts/scores/calibration.csv \
  --summary-json artifacts/scores/calibration_summary.json
```

Calibration prints a small separation summary and writes one row per scored file.

## Notes

- Scoring works best on short clips or already-segmented tracks.
- If your source files are long recordings, segment them first with the commands in [Segmentation](segmentation.md).
