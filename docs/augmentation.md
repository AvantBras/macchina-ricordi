# Augmentation

Augmentation grows a dataset by writing transformed copies of existing clips. For now there is one transform:
changing speed.

## Speed variants of a file or folder

```bash
python -m macchina_ricordi.cli augment speed clip.wav artifacts/augmented --ratios 0.8,1,1.25
python -m macchina_ricordi.cli augment speed curated_clips artifacts/augmented --ratios 0.9,1,1.1 --recursive
```

Each source produces one `<stem>__speed<ratio>.flac` per ratio (`1` = original speed, `2` = twice as fast; the dot
becomes `p`, e.g. `__speed1p25`). Ratio `1` is written too, so the output folder is a complete set. Folder input
mirrors the source tree. An `augment_report.json` in the output directory lists every variant with its duration.

Options:

- `--mode varispeed` (default): tape/turntable speed. Pitch moves with speed (`0.8` ≈ −3.9 semitones, `1.25` ≈ +3.9)
  and no processing artefacts are introduced.
- `--mode stretch`: phase-vocoder time stretch. Pitch is kept, but transients get smeared, and a model trained on it
  will learn that smear. It is also far heavier: the phase-vocoder work dominates the pass. On the author's
  workstation an early run measured a 27-minute stereo file at ~75 s and 8 GB RAM, versus ~9 s for varispeed. Treat
  those as a rough relative guide, not a guarantee — the hardware and exact load were not recorded.
- `--format flac|wav`, `--output-sr`, `--mono`: output encoding. By default each source keeps its own sample rate and
  channels.

## Augmenting a training workspace

For training data, augment inside `data prepare` instead of before it:

```bash
python -m macchina_ricordi.cli data prepare curated_clips \
  --workspace-dir .workspace_curated --speed-ratios 0.9,1.1
```

`--workspace-dir` is required: it names a new workspace folder that `prepare` wipes and
refills (an old one is refused unless you pass `--overwrite`).

`prepare` shuffles files into train/valid. If you augment first, variants of a validation clip end up in train, and
validation perplexity turns into a memorisation score. With `--speed-ratios`, variants are written after the split,
next to the train clips only (ratio `1` is ignored because the original is already there). `logs/prepare_dataset.json`
records the ratios, mode, and variant count. The added files change the dataset fingerprint, so training picks a
fresh Dora signature automatically.

## Watch the clip length

A ratio above 1 shortens clips. With pre-sliced data (e.g. 10 s slices trained at `segment_duration=10`), `1.25` turns
them into 8 s clips that get silence-padded, which is the "music, then silence" hazard. Run
`data audit --workspace-dir <workspace>` after preparing. The training guard will also warn or refuse. When
clips are sliced right at `segment_duration`, either use only ratios ≤ 1 or augment the long recordings before slicing
them.
