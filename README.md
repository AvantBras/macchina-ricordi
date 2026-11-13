# macchina ricordi

A personal, experimental toolbox for training small music models on records you own, then playing them:

- **the stream**: live, steerable generation on a GPU, round by round, from a keyboard TUI
- **the app**: a trained model packed into one `.nrbox` file and played on a CPU, with no Python,
  torch or GPU to install
- the tools around them: training and comparing models, blind A/B listening, and scoring, describing, segmenting and
  looping audio

It grew as an art project, not a product, so expect rough edges. Training and the stream need Linux and an NVIDIA GPU;
the app has build support for Linux and macOS (Apple Silicon). Windows support is experimental: it has not
been built or tested. See [the app guide](docs/appliance.md#macos-and-windows) for platform limits.

## Install

`macchina_ricordi` is installed from a source checkout — the workspace commands read the checkout's
`musicgen/pyproject.toml` for its recipes, codecs and split defaults, so a bare wheel/PyPI install of the
package alone is not supported. Clone the repository, then install it from there:

```bash
git clone https://github.com/AvantBras/macchina-ricordi macchina-ricordi
cd macchina-ricordi
uv sync --extra dev      # base tools, tests, and the `ricordi` command
make setup-musicgen      # the AudioCraft/Dora runtime, in its own environment (.venv-musicgen)
uv run ricordi runtime check
```

The examples below use `ricordi …`; run them as `uv run ricordi …` from the checkout,
or activate `.venv` first. `python -m macchina_ricordi.cli …` is the equivalent module command
when using that environment. Optional model-backed listening and audio playback need their
extras; `uv sync --all-extras` (also `make install`) installs all of them and can download large
GPU dependencies. The base tools and tests do not need them.

Audio and analysis commands
(`score`, `listen`, `segment`, `loop`, `augment`, `judge`) run fine anywhere; anything that touches the
MusicGen workspace (`data`, `train`, `model`, `play`, `tidy`, `runtime`, `gui`, `session`) needs the
checkout and prints how to get one if it is missing. If the package is installed non-editable (copied into
site-packages rather than an editable dev install), point it at the checkout with
`MACCHINA_RICORDI_ROOT=/path/to/macchina-ricordi`.

## Development checks

```bash
uv sync --locked --extra dev
make test
make lint
```

The fast CI workflow runs base-plus-dev tests on Python 3.11 and 3.12, checks the root
lockfile, and runs correctness lint and archive builds. Formatting is optional; the
MusicGen lockfile check is a separate, non-blocking advisory. CI runs on pull requests,
pushes to `main`, and manual dispatch. The separate macOS workflow builds and exercises
the appliance for relevant changes; PR builds use only its random test model, not your
release models. If you choose a default branch other than `main`, update the branch
filters and trusted-release condition in `.github/workflows/` to match.

See [Contributing](CONTRIBUTING.md) for the small-change workflow and
[repository protection](docs/repository-protection.md) for the prepared publication guardrails.

## A model to start with

**[Download a9220dd9-ep200 on Hugging Face](https://huggingface.co/AvantBras/a9220dd9-ep200)** —
a small unconditional model trained on my own music: `reshape-nano-wide`, 7.35 million parameters,
epoch 200, 15-second context, about **29.4 MB** of full-precision weights. The model bundle is
**CC BY-SA 4.0**; the toolbox code remains MIT.

### Download and generate your first sounds

From the source checkout, after [installing the base tools and MusicGen runtime](#install):

```bash
uv run python scripts/download_example_model.py
uv run ricordi model show a9220dd9-ep200
uv run ricordi play generate a9220dd9-ep200 --device cuda \
  --duration-seconds 10 --num-samples 2 --seed 42 \
  --save-output --output-dir artifacts/example-model
```

Open the generated WAV files under `artifacts/example-model/<date>/generate_<runid>/takes/` in your
audio player (the default session layout). For batch generation
without a GPU, use `--device cpu` instead (slower; the MusicGen runtime is still required). The runtime
setup above targets Linux; this is not a standalone Windows/macOS installer.

The small download script needs only Python 3.11/3.12 and internet access: no Hugging Face account,
token, or extra download package. It fetches a pinned published revision, verifies **all four files**
by SHA-256, and refuses to overwrite different or incomplete existing files. Running it again checks
an already-installed copy. It downloads the model only, not the AudioCraft runtime.

To download manually, take these four files from the model repository and put them here:

```text
models/a9220dd9-ep200/
├── model.safetensors
├── config.json
├── README.md
└── LICENSE
```

Do not put the files in `.dora/` or add another nested model folder. `models/` is ignored by Git;
weights are hosted on Hugging Face, not bundled in the source repository. No dataset, training history,
or original experiment is needed. On first generation, AudioCraft downloads the separate standard
EnCodec codec; its weights retain their own **CC BY-NC 4.0** terms. This custom bundle loads through
macchina-ricordi, not Transformers' `AutoModel.from_pretrained()` or hosted HF inference.

### Play and explore

Use `uv run ricordi gui --open` for the browser interface and select **a9220dd9-ep200**. For live GPU
streaming with keyboard controls, install the audio extra (and system PortAudio if needed), then:

```bash
uv sync --extra dev --extra instrument
uv run ricordi play stream --xp-path models/a9220dd9-ep200 --tui
```

For batch experiments, change `--seed`, `--temperature`, `--top-k`, `--top-p`, or `--min-p` on
`play generate`; use `uv run ricordi play generate --help` for the full controls. This model is
unconditional: it does not take text prompts. See [the stream guide](docs/musicgen-stream.md) and
[creative tools](docs/musicgen-creative.md) for continuation, sweeps and variations.

The portable bundle is **not** a `.nrbox` app model. For the CPU app,
[export and pack a compatible model](docs/appliance.md#the-pipeline) or obtain a `.nrbox` separately.
The app can open without a bundled model, but needs one before playing.

## Command map

```bash
ricordi score file path/to/audio.wav --json
ricordi score dir path/to/library --recursive --output artifacts/scores/library.csv
ricordi listen file path/to/audio.wav
ricordi listen ask path/to/audio.wav "Is there a steady pulse?" "What changes over time?"
ricordi segment file path/to/full_album.mp3 artifacts/segments
ricordi loop find path/to/song.mp3 artifacts/loops/song

ricordi data build my_dataset path/to/your/music --slice 10
ricordi train run --recipe reshape-nano --dataset my_dataset --dry-run
ricordi train queue add --recipe reshape-nano --dataset my_dataset --epochs 30 --run-label a
ricordi train queue run  # foreground worker: runs queued jobs on the GPU
ricordi train queue list # confirm the job completed before generating from a@best
ricordi model list
ricordi play generate a@best --duration-seconds 20 --num-samples 2 --listen
ricordi play stream --tui
ricordi model export a@best --bundle --out-dir models --name a   # a portable bundle to host, move or hand over
ricordi model export a@best --box
ricordi gui --open
```

Replace the audio paths with your own inputs. The training sequence requires the MusicGen runtime;
wait for the queue worker to finish and check that the job completed before using `a@best`.
Every command takes `--help`. Training `--dry-run` shows the plan without launching training,
but can refresh runtime dataset configuration and fetch a referenced checkpoint; it is not a no-write mode.

## Documentation

- [The command line](docs/cli.md) and the [glossary](docs/glossary.md): the six nouns (source, dataset, recipe, model,
  session, judgement)
- [MusicGen](docs/musicgen.md): datasets, recipes, training · [Models](docs/models.md) · [Sessions](docs/sessions.md)
- [The stream](docs/musicgen-stream.md) · [Creative tools](docs/musicgen-creative.md) · [The app](docs/appliance.md)
- [Scoring](docs/scoring.md) · [Listening](docs/listening.md) · [Segmentation](docs/segmentation.md) ·
  [Voice loops](docs/looping.md) · [The GUI](docs/gui.md)

## Licence

The code is MIT ([LICENSE](LICENSE)). The runtime uses the MIT-licensed
[aaronabebe AudioCraft fork](https://github.com/aaronabebe/audiocraft), pinned to
[`1ca32f28d2d6c223b4e2335042ac558c308f2ea2`](https://github.com/aaronabebe/audiocraft/commit/1ca32f28d2d6c223b4e2335042ac558c308f2ea2),
with the patches in `patches/`. It derives from [Meta's AudioCraft](https://github.com/facebookresearch/audiocraft).
Runtime setup needs that external repository to remain available; the pin does not mirror its contents.

This is a local, trusted-machine toolbox, not an authenticated network service. Read the
[security and trust boundaries](SECURITY.md) before loading legacy checkpoints, enabling neural
captioners, or exposing the GUI.

Model weights have a separate licence chosen by their publisher; check the bundle's `LICENSE` before use or redistribution.
The default EnCodec codec is released by Meta under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Portable bundles reference it as a separate download; `.nrbox` files include codec weights/graphs and carry those
licence obligations. The code's MIT licence does not replace the model or codec licences.

## About the name

I didn't invent the name *macchina ricordi*: it comes from the work of
[Pierre-André Arcand](https://fr.wikipedia.org/wiki/Pierre-Andr%C3%A9_Arcand), who created an
instrument of that name for recording and layering evolving sound loops. That connection to
repetition, recorded memory and transformation felt fitting for this toolbox.

The name is a nod to his work, not a claim of affiliation or endorsement. If you feel its use
here is inappropriate, please open an issue—I'm open to changing it.
