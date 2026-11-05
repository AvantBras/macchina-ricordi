# The appliance — a trained model as a program you can hand to someone

`appliance` takes one trained model out of this repository and turns it into a
program that plays it, endlessly, on a CPU: no GPU, no Python install, no
AudioCraft, no torch, no experiment folder.

It exists for a different reason from [the stream](musicgen-stream.md). The
stream is an instrument for whoever has this repo checked out and a suitable NVIDIA
GPU in the machine. This is for handing the work to somebody else — a gallery, a friend, a
laptop in a room — where nothing can be installed, configured or explained.

The size and speed figures below are historical author-reported measurements of
specific models and machines, not guarantees for a new build or another computer.
This checkout does not include those trained models.

---

## What gets handed over

A built application folder, optionally with model files bundled inside it. An empty
app opens the panel, but needs a `.nrbox` added before it can play.

- **The app** — about 110 MB on disk, about 30 MB as a `.tar.xz`. Python, ONNX
  Runtime, numpy and PortAudio, and nothing else. No torch: the first version of
  this needed about a gigabyte of dependencies, 711 MB of it torch.
- **A box** (`.nrbox`) — 194 MB for a nano model. A zip holding `spec.json`, the
  language model as `lm.onnx` (15 MB), the audio decoder as `decoder.onnx`
  (118 MB), and the audio *encoder* as `encoder.onnx` (61 MB), which only the
  controls that reshape the past need (133 MB without it, packed with
  `--no-encoder`). The decoder and encoder are Meta's EnCodec, the same bytes in
  every box; the model itself is the 15 MB.

Whoever receives it double-clicks `macchina-ricordi`, a page opens in their browser, and
they press Play. Sound comes out of the computer's speakers; the page is only a
control surface. A `README.txt` in the folder says exactly that.

On macOS the app is `macchina-ricordi.app`, zipped with its README, built for Apple
Silicon. It has no window and no Dock icon — the page is the whole interface —
so the page carries what a window would: **Add a model…**, which copies a `.nrbox`
into the person's models folder, and **Quit**. Opening the app a second time
brings the page back instead of starting a second player.

## The pipeline

A box is exported from a model, the way everything else names one ([model references](models.md)).
The reference may be a trained model *or* a [portable bundle](models.md#the-portable-bundle) you
received — so the box pipeline starts from whatever someone handed you. `--weights-state
model|ema|best` chooses which snapshot a box is lifted from (default: `model`, the raw saved weights;
the tools never silently swap in an EMA copy).
The examples use illustrative local models and boxes: replace `midwife@best`, the
export signature, and the box paths with your own. They are not included downloads.

```bash
python -m macchina_ricordi.cli model export midwife@best --box --dry-run   # prints the plan/card; does not export or pack
python -m macchina_ricordi.cli model export midwife@best --box             # boxes/midwife_ep<N>.nrbox
```

It reads the model record, writes it as a card (`boxes/src/<name>.card.json`), and runs steps 1 and 2 below
in their own environments, recording each in `boxes/<name>.export_status.json`. The card goes into the box's
`spec.json` as `model` (sig, name, dataset, recipe, parents, epoch, valid ce, measured behaviour), and its name
and title become the box's name and note, so a box says which model it is. Export refuses a checkpoint whose
architecture is not the record's. `--no-pack` stops at the source; `--no-probe` skips the attractor probe.
There is one `ModelSpec` in the repo, the box's (`appliance/spec.py`); the per-model study numbers the stream
reads are `StudiedModel` in `macchina_ricordi/models/specs.py`.

A portable bundle never carries the codec: the standard EnCodec graph that every recipe uses is downloaded
separately from its public hosted weights (expect a download on first use), and a `.nrbox` holds a copy of the
codec graphs inside it. The bundle's v1 schema refuses a custom-codec or text-conditioned model until those
are supported, so such a model packs as `.nrbox` for now.

Three steps, because three different sets of dependencies are involved and only
the last one ships.

| step | what it does | environment | writes |
|---|---|---|---|
| **export** | lifts a model out of its Dora experiment | `.venv-musicgen` (AudioCraft) | `.nrsrc` source |
| **pack** | converts to ONNX, checks it against torch, probes it | `.venv-appliance` (torch, onnx) | `.nrbox` |
| **build** | freezes the player into a folder | `.venv-appliance-build` (no torch) | `dist/appliance/macchina-ricordi/` |

### 1. Export

```bash
AUDIOCRAFT_DORA_DIR=$PWD/.dora PYTHONPATH=. \
  .venv-musicgen/bin/python -m appliance.export 31f7067b --out boxes/src/disco.nrsrc
```

A `.nrsrc` is a torch file — fp16 weights, the decoder as TorchScript, the spec.
It is an intermediate, not something to hand out.

### 2. Pack

The default pack includes an encoder and loads Meta's `facebook/encodec_32khz`
from Hugging Face through `transformers`; expect a model download on first use.
These are packing dependencies, not requirements for someone receiving the built app.

```bash
uv venv --python 3.11 .venv-appliance
uv pip install --python .venv-appliance/bin/python --index-url https://download.pytorch.org/whl/cpu torch
uv pip install --python .venv-appliance/bin/python onnx onnxruntime numpy sounddevice transformers

PYTHONPATH=. .venv-appliance/bin/python -m appliance.pack boxes/src/*.nrsrc --out-dir boxes
```

Packing refuses to write a box until the exported graphs have been run beside
the torch reference and agree with it — the same sampled token at every position
tested, the decoder within 1e-3. The encoder comes from the pretrained EnCodec on
Hugging Face rather than from the experiment, so packing first checks that the
model's own decoder is that codec's, then
that the exported encoder picks the same codes as the torch one. The language
model and the encoder are then **stored at half precision** and checked again:
the model's logits come out identical to the last bit (its weights were float16
already), the encoder picks the same codes. The decoder, which is what people
hear, stays at full precision (see [Things worth knowing](#things-worth-knowing)). Then it **probes the box for the silent
attractor** using the runtime that ships, and records the lowest temperature
that kept every take alive. `--no-probe` skips that (seconds instead of minutes);
`--keep-tuning` reuses a temperature already in the source.

A box packed before 2026-09-19 is halved in place, after the same check, with
`PYTHONPATH=. .venv-appliance/bin/python -m appliance.halve boxes/*.nrbox`.

### 3. Build

```bash
uv venv --python-preference only-managed --python 3.11 .venv-appliance-build
uv pip install --python .venv-appliance-build/bin/python onnxruntime numpy sounddevice pyinstaller

PYTHONPATH=. .venv-appliance-build/bin/python packaging/appliance/build.py --boxes boxes/*.nrbox
```

Writes the folder to `dist/appliance/macchina-ricordi/` and an archive beside it. Two
things the script enforces or works around:

- **It refuses to run if torch is importable.** PyInstaller ships whatever it can
  import, and an environment with torch quietly adds 700 MB.
- **It needs a uv-managed Python**, hence `only-managed`: Debian's system Python
  has no `libpython3.11.so`, which PyInstaller embeds.

On Linux the build carries its own PortAudio, and the app loads that copy rather
than the system's, so it plays on a machine without `libportaudio2` installed.

### 4. macOS, without a Mac

PyInstaller cannot cross-compile, so the macOS app is built on a Mac — GitHub's
Apple Silicon runners, by `.github/workflows/appliance-macos.yml`. Relevant
pushes to `main` and pull requests run it; it can also be started manually.
Pull requests build an empty app and verify it with the random test model below,
without downloading or bundling the release models:

1. A Linux job makes a **test model** with `packaging/appliance/make_test_box.py`:
   the nano architecture with random language-model weights, and Meta's real
   EnCodec decoder downloaded from Hugging Face — numerically the same decoder a
   real box carries (within 8e-7). Nobody's trained weights leave this machine.
2. The Mac job runs the unit tests on arm64, builds `macchina-ricordi.app` from an
   environment without torch, checks its signature, times it at 1, 2 and 4
   threads, and runs `packaging/appliance/smoke_test.py`, which uses the built
   app the way a person would — starts empty, adds a model through the page,
   plays, quits — and then launches it through LaunchServices, as a double-click
   does.
3. The zipped app is kept as the run's artifact for 30 days:

   ```bash
   gh run download --name macchina-ricordi-macos-arm64
   ```

On pushes to `main` and manual runs on `main`, models are built in from the
repository's **`appliance-models`** release. Pull requests and manual runs on
other branches skip this download and build an empty app.
Every `.nrbox` attached to the release goes inside the trusted main build, so a friend receives one
download and presses Play. Upload one with
`gh release upload appliance-models boxes/<name>.nrbox`; with no release, the app
ships empty and the person adds a model through the page.

**In a public repository, those release assets and the models bundled into the
app are public.** Workflow artifacts are also downloadable by signed-in GitHub
users with read access to the repository, including public repositories. Only
upload models you intend to distribute publicly and have the rights to share;
`.gitignore` does not protect release uploads or workflow artifacts. Keep private
models out of the `appliance-models` release. The README inside the zip tells the
person how to get past the first-launch warning.

Historical build measurements reported an Apple Silicon runner on macOS 26.6:
108 MB for the app, **34.5 MB zipped**, and 1.91x real time on one thread
(worst round 1.61x) or 2.65x on four, without dropouts in that test. Those results
are not a verification of a build or release from this public checkout. That build
required **macOS 14 or newer**; the current minimum depends on the wheels installed
for the build and is printed in its generated README.

Two things the build handles that are easy to get wrong by hand: the zip is made
with `ditto`, because an ordinary zip flattens the symlinks inside a Mac bundle and
breaks its signature; and the minimum macOS version is read off the bundled
wheels rather than assumed — numpy's Apple Silicon wheels, for one, are built
against Accelerate and need a recent macOS.

## Play without building

Any environment with `onnxruntime`, `numpy` and `sounddevice` plays a box directly:

```bash
PYTHONPATH=. python -m appliance panel                 # the browser panel
PYTHONPATH=. python -m appliance play boxes/nano_disco_60ep.nrbox
PYTHONPATH=. python -m appliance render boxes/nano_disco_60ep.nrbox --out out.wav --seconds 300
PYTHONPATH=. python -m appliance list boxes
```

The built app takes the same commands; with none, it opens the panel.

Models added through the panel, and recordings, go to one folder a person can
find: `~/macchina-ricordi/` on Linux and Windows, **Music › macchina-ricordi** on macOS
(`MACCHINA_RICORDI_HOME` moves it). When the app runs without a terminal it writes its
output to `macchina-ricordi.log` there — the first thing to ask a remote tester for.

## Speed and memory

Historical measurements on a Ryzen 9 5900X with `nano_disco_60ep`, 4-second rounds,
each configuration in its own process on an otherwise idle machine. The trained
model is not included here, so these figures are context rather than a reproducible
benchmark supplied by this checkout:

| threads | default | `--fast` | torch, for reference |
|---|---|---|---|
| 1 | 1.71 xRT · 495 MB | 2.21 xRT · 1314 MB | 1.58 xRT · 961 MB |
| 2 | 2.25 xRT · 512 MB | 3.05 xRT · 1328 MB | 2.26 xRT · 877 MB |
| 4 | 2.66 xRT · 485 MB | 3.60 xRT · 1310 MB | 3.18 xRT · 971 MB |

In this measurement, the default matched torch on one or two threads at half the memory. `--fast`
keeps ONNX Runtime's memory arena, which beats torch everywhere and holds on to
about 850 MB more. The worst single round on one thread, default settings, is
1.62 xRT — the number that decides whether it survives the night.

## Controls

The panel, the flags and [the stream](musicgen-stream.md) share one vocabulary,
`macchina_ricordi/instrument/controls.py`: each control's name, meaning, default and range, the steering crossovers, and
the weights of the "bring it back" gestures. The appliance keeps narrower ranges where it has a reason (memory up
to 5 s, a box's context; tape speed 0.9–1.1, past which EnCodec's round trip adds its own artefacts); the stream
goes further (tape speed 0.5–2, memory up to the model's horizon). Temperature stops at 3 on both.

| panel | flag | what it does |
|---|---|---|
| Wildness | `--temperature` | Low is careful and can fall silent. Defaults to the box's probed value. |
| Variety | `--top-k` | How many options it considers each step. |
| Memory | `--tau` | Seconds of what just played the model re-reads, 0.5–5 (default 5). |
| Stride | `--round-seconds` | New seconds each round adds, 1–10 (default 4). Longer answers the controls more slowly. |
| Low / Mid / High | `--tilt-low` `--tilt-mid` `--tilt-high` | ±12 dB on the past the model re-reads, below 250 Hz / between / above 4 kHz. |
| Tape speed | `--tape-speed` | 0.9–1.1: the past played back to the model slower and lower, or faster and higher. |
| Back to neutral | — | Tilt flat, tape speed 1. Touches nothing else. |
| Pin this moment | — | Remembers the round that is audible now. |
| Bring it back, lightly / strongly | — | Blends the newest pin into the next round's past, 20% / 60%, once. |
| Recall a pin | `--recall-every` | Every N rounds, blends a pin back in at 50%, cycling through them. |
| — | `--fast` | About 30% quicker, about 850 MB more memory. |
| — | `--threads` | Leave unset to let ONNX Runtime choose. |

Memory and stride together have to fit inside the length the model was trained
on — past it, the model's positions were never seen — so the stride gives way
when they would not, and the panel says so.

There is one take per round, always. The GPU stream can generate several and
keep the best, but choosing means scoring them as they are made, and on a CPU
that costs more than the loop has to spare; so the appliance has no candidates
and nothing that scores.

## Steering the past

The last six rows are the MusicGen stream's prompt steering, with the same
constants: the same tilt curve (identical to the stream's, point for point), the
same 20/60% bring-back and 50% recall. They do not change how the model samples.
They change **what it believes its recent past was**: each round the tail it is
about to continue from is decoded, blended with a pin, played back at tape speed,
tilted, and encoded back into codes. What is *played* is never touched.

Two consequences are the whole character of these controls. **They compound**: a
round's output becomes the next round's past, so a tilt held for a few rounds
drags the whole stream. And **neutral does not snap back**: the stream stays
wherever it drifted, because the model is now genuinely making that music.

The re-encoding costs something, and only in rounds where one of these is off
neutral. Historical measurements on the same Ryzen with `nano_disco_60ep`, a
4-second round:

| threads | neutral | steering |
|---|---|---|
| 1 | 1.78x real time | 1.38x (0.64 s of it encoding) |
| 2 | 2.58x | 2.07x |
| 4 | 3.03x | 2.58x |

Holding +12 dB on the lows, the low band of the played output rose round after
round, to 5.3 dB above an identical neutral run by rounds 7–12. **Tape speed, on
disco, did nothing measurable to pitch**: the past was shifted by 165 cents, and
the output, over four seeds, by 0. That model hears a sped-up past and continues
in its own register. These are observations about that model and test, not a
guarantee of how another model responds to tape speed.

## Things worth knowing

**Silence is absorbing in a stream.** A rolling context that has gone quiet
predicts more quiet from itself; at temperature 1.0 a five-minute stream of
`nano_disco_60ep` went silent in its first minute and never recovered. The probed
temperature in each box avoids this, and a watchdog drops a dead context after
two silent rounds as a backstop.

**int8 changes the model.** In the historical test, dynamic int8 quantisation of
the language model was 40% faster and picked a different token 18% of the time. That is a different
instrument, not a smaller one, so boxes compute in fp32.

**fp16 storage and fp16 arithmetic are different trade-offs.** On the tested CPU,
computing the decoder in fp16 made decoding 35 times slower; this is a hardware-
and runtime-specific result, not a property of every CPU.
Storing weights in fp16 with a cast back to fp32, which ONNX Runtime folds when
the box opens, costs nothing at run time — that is how the language model and
encoder are stored. The decoder is not: halved, it usually rendered the same
codes within 70 dB, but in one second of midwife in about 600 the difference was
only 24 dB under the music, and that has not been listened to. It would take a
box from 194 MB to 135 MB. Those measurements motivated keeping this decoder at
full precision; they were not rerun for this checkout.

**Decoding less context is not free.** Each round re-decodes a ten-second window
so every emitted sample has audio to its left. Cutting that to one second keeps
the joins clean but changes the rendering throughout (the difference sits only
18.5 dB below the signal) — the decoder remembers further back than the seam
metric can see. It stays at the full window.

## macOS and Windows

**macOS** has a build-and-test workflow above, targeting Apple Silicon only
(an Intel build would need an Intel runner). Check that workflow's result before
distributing a build; the minimum macOS version is in its generated README
(macOS 14 for the historical build described above). The app is signed ad hoc, which Apple
Silicon requires to run at all, but not with a Developer ID, so the first launch
is refused once: the person clicks Done, then **System Settings › Privacy &
Security › Open Anyway**. The README in the zip walks through it, with the
Terminal fallback for the "damaged" message. Removing that step needs an Apple
Developer account ($99 a year) for signing and notarisation.

A downloaded app that has not been moved is run by macOS from a randomised
read-only copy (App Translocation), so a `boxes` folder next to the app cannot be
relied on there. Models are therefore added through the page, or built into the
bundle with `--boxes` at build time, both of which survive it.

**Windows** is written for — the build script and the panel are the same — but
not built or tested.

## Limits

- **Only small models were real-time in the reported tests.** A 126 M-parameter
  model ran below real time on the Ryzen above; roughly 8 M parameters at 50 Hz
  was the practical range there. This is not a hardware-independent size limit.
- **Each box carries its own codec.** 179 MB of every 194 MB box is the same
  EnCodec decoder and encoder; ten models are 1.9 GB where they could be 330 MB.
- **Unconditional models only.**
