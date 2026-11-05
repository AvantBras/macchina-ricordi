# Segmentation

Segmentation splits long recordings into scorer-friendly WAVs by looking for sustained quiet gaps. The tracks it
writes are sources, so they belong in `library/` ([glossary](glossary.md#source)), where the GUI's library lists
them for a new dataset.

## Segment One Recording

```bash
python -m macchina_ricordi.cli segment file \
  /path/to/full_album.mp3 \
  library/my_album
```

This writes one subdirectory per source file and creates a per-album `segments.csv` manifest.

## Segment A Directory

```bash
python -m macchina_ricordi.cli segment dir \
  /path/to/archive \
  library/my_archive \
  --recursive
```

This writes a combined manifest at `library/my_archive/segments.csv`.

## Main Tuning Knobs

- `--min-silence-seconds` requires longer quiet gaps before a split is allowed.
- `--silence-threshold-db` makes gap detection stricter or looser.
- `--min-track-seconds` prevents tiny accidental fragments.
- `--output-sr` and `--mono/--no-mono` control the exported WAV format.

Default exports are `32kHz` mono WAVs so they are ready for scoring and MusicGen dataset prep.

## Typical Flow

```bash
python -m macchina_ricordi.cli segment dir /path/to/archive library/my_archive --recursive
python -m macchina_ricordi.cli score dir library/my_archive --recursive --output artifacts/scores/my_archive.csv
python -m macchina_ricordi.cli data build my_dataset library/my_archive --slice 10
```

## Notes

- The gap-based approach works best when tracks have real pauses between them.
- Gapless albums and live recordings may still need manual review or a second-pass boundary method.
