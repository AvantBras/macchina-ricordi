"""Turn a Dora experiment (or a portable model bundle) into a ``.nrbox`` a plain torch install can play.

Usually reached through ``python -m macchina_ricordi.cli model export <model> --box``, which resolves the model,
writes its record as ``--card`` and runs this, then ``appliance.pack``. Run directly, it needs the AudioCraft
interpreter (``.venv-musicgen``); it is the last step that needs one. It writes a ``.nrsrc`` source;
``appliance.pack`` turns that into the ``.nrbox`` that plays without torch.

    AUDIOCRAFT_DORA_DIR=$PWD/.dora PYTHONPATH=. \\
      .venv-musicgen/bin/python -m appliance.export 31f7067b --out boxes/src/disco.nrsrc

A *bundle directory* may also be given positionally (in place of the Dora
signature): the model is then loaded through ``macchina_ricordi.models.bundle_runtime``
with no solver or XP involved, and the spec is built from the bundle's own
manifest. Bundles carry whichever snapshot they were written from, so
``--weights-state`` is ignored for a bundle input (the bundle already chose).

For a legacy Dora export, ``--weights-state {model,ema,best}`` selects the
snapshot, defaulting to the raw ``model`` state — there is no silent EMA
default. The historical ``use_ema`` boolean remains available to explicit callers
(``use_ema=True`` means ``weights_state="ema"``).

The codec is traced rather than reimplemented. Its decoder is Meta's, its
encoder is not needed at play time, and a traced graph cannot drift from the
weights it was traced with.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from .format import write_source
from .spec import ModelSpec, check_card

LEGACY_WEIGHTS_STATES = ("model", "ema", "best")


class _DecoderOnly(torch.nn.Module):
    """``[B, n_q, T]`` codes -> ``[B, samples]``, with the encoder left behind.

    Half of EnCodec's 56.9 M parameters are its encoder, and a player never
    encodes anything. Replacing it before the trace is what takes a box from
    192 MB to 78 MB; tracing alone does not, because freezing keeps the weights
    of modules that are merely *reachable*, not only those that ran.
    """

    def __init__(self, compression_model) -> None:
        super().__init__()
        inner = getattr(compression_model, "model", None)
        if inner is not None and hasattr(inner, "encoder"):
            inner.encoder = torch.nn.Identity()
        self.model = compression_model

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes)[:, 0]


def _lm_spec(solver, name: str, origin: str, note: str) -> ModelSpec:
    cfg = solver.cfg
    lm_cfg = cfg.transformer_lm
    compression = solver.compression_model
    pattern = cfg.codebooks_pattern
    delays = list(pattern.delay.delays)[: int(lm_cfg.n_q)]
    segment = float(cfg.dataset.segment_duration)
    return ModelSpec(
        dim=int(lm_cfg.dim),
        num_heads=int(lm_cfg.num_heads),
        num_layers=int(lm_cfg.num_layers),
        n_q=int(lm_cfg.n_q),
        card=int(lm_cfg.card),
        hidden_scale=float(lm_cfg.hidden_scale),
        layer_scale=None if not lm_cfg.layer_scale else float(lm_cfg.layer_scale),
        delays=delays,
        frame_rate=float(compression.frame_rate),
        sample_rate=int(compression.sample_rate),
        channels=int(compression.channels),
        segment_duration=segment,
        name=name,
        origin=origin,
        max_period=float(getattr(lm_cfg, "max_period", 10_000.0)),
        note=note,
    )


def _bundle_spec(model_config: dict, audio: dict, *, name: str, origin: str, note: str) -> ModelSpec:
    """Build a ``ModelSpec`` straight from a bundle's manifest — no solver, no XP."""
    lm_cfg = model_config.get("transformer_lm") or {}
    pattern = model_config.get("codebooks_pattern") or {}
    delays = _pattern_delays(pattern, int(lm_cfg.get("n_q", 4)))
    return ModelSpec(
        dim=int(lm_cfg.get("dim", 0)),
        num_heads=int(lm_cfg.get("num_heads", 0)),
        num_layers=int(lm_cfg.get("num_layers", 0)),
        n_q=int(lm_cfg.get("n_q", 0)),
        card=int(lm_cfg.get("card", 0)),
        hidden_scale=float(lm_cfg.get("hidden_scale", 1.0)),
        layer_scale=None if lm_cfg.get("layer_scale") is None else float(lm_cfg.get("layer_scale")),
        delays=delays,
        frame_rate=float(audio.get("frame_rate", 50.0)),
        sample_rate=int(audio.get("sample_rate", 32_000)),
        channels=int(audio.get("channels", 1)),
        segment_duration=float(model_config.get("dataset", {}).get("segment_duration", 10.0)),
        name=name,
        origin=origin,
        max_period=float(lm_cfg.get("max_period", 10_000.0)),
        note=note,
    )


def _pattern_delays(pattern: dict, n_q: int) -> list[int]:
    if isinstance(pattern, dict) and isinstance(pattern.get("delay"), dict):
        delays = pattern["delay"].get("delays")
        if isinstance(delays, list) and delays:
            return [int(d) for d in delays][:n_q]
    return list(range(n_q))


def _trace_decoder(compression_model, n_q: int) -> torch.jit.ScriptModule:
    decoder = _DecoderOnly(compression_model).eval()
    example = torch.zeros(1, n_q, 16, dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(decoder, example, check_trace=False)
        traced = torch.jit.freeze(traced)
    return traced


def export(
    sig: str,
    out: Path,
    *,
    checkpoint: str = "checkpoint.th",
    name: str | None = None,
    note: str = "",
    half: bool = True,
    card: dict | None = None,
    weights_state: str = "model",
    use_ema: bool | None = None,
) -> Path:
    """Export ``sig`` (a Dora signature or a bundle directory) to a ``.nrsrc`` source.

    ``weights_state`` selects ``model`` (raw), ``ema`` or ``best`` for a legacy
    Dora checkpoint, defaulting to the raw model. ``use_ema`` is kept for
    explicit legacy callers: ``True`` maps to ``weights_state='ema'`` and
    overrides ``weights_state`` when it is still ``model``; it never silently
    turns on EMA by default.
    """
    from macchina_ricordi.models.bundle_runtime import bundle_dir_or_none

    bundle_path = bundle_dir_or_none(sig)
    if bundle_path is not None:
        return _export_bundle(
            bundle_path,
            out,
            name=name,
            note=note,
            half=half,
            card=card,
        )

    resolved_weights_state = _resolve_weights_state(weights_state, legacy_use_ema=use_ema)
    return _export_legacy(
        sig,
        out,
        checkpoint=checkpoint,
        name=name,
        note=note,
        half=half,
        card=card,
        weights_state=resolved_weights_state,
    )


def _resolve_weights_state(weights_state: str, *, legacy_use_ema: bool | None) -> str:
    state = weights_state
    if legacy_use_ema is True and state == "model":
        state = "ema"
    if state not in LEGACY_WEIGHTS_STATES:
        raise ValueError(f"unsupported weights_state {state!r}; expected one of {', '.join(LEGACY_WEIGHTS_STATES)}")
    return state


def _export_legacy(
    sig: str,
    out: Path,
    *,
    checkpoint: str,
    name: str | None,
    note: str,
    half: bool,
    card: dict | None,
    weights_state: str,
) -> Path:
    from audiocraft import train
    from audiocraft.utils import checkpoint as checkpoint_utils

    dora_dir = Path(__import__("os").environ.get("AUDIOCRAFT_DORA_DIR", ".dora"))
    xp_path = dora_dir / "xps" / sig
    ckpt_path = xp_path / checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    solver = train.get_solver_from_sig(
        sig,
        override_cfg={
            "device": "cpu",
            "autocast": False,
            "execute_inplace": True,
            "logging": {"log_wandb": False, "log_tensorboard": False},
            "dataset": {"num_workers": 0},
        },
        restore=False,
        load_best=False,
        disable_fsdp=True,
    )
    state = checkpoint_utils.load_checkpoint(ckpt_path)
    sources = getattr(getattr(solver, "stateful", None), "sources", {})
    if isinstance(sources, dict) and "scaler" in state and "scaler" not in sources:
        state = dict(state)
        state.pop("scaler", None)
    solver.load_state_dict(state)
    solver.model.eval()

    # Select the snapshot with the shared core helper so a raw export resolves
    # exactly the same weights the bundle exporter would.
    from macchina_ricordi.models.bundle import select_checkpoint_weights

    lm_state = select_checkpoint_weights(state, weights_state)

    label = name or (card or {}).get("name") or _label_for(xp_path, sig)
    note = note or (card or {}).get("note") or ""
    spec = _lm_spec(solver, label, f"{sig}/{checkpoint}", note)
    if card:
        check_card(spec, card)
        spec = replace(spec, model=dict(card))

    traced = _trace_decoder(solver.compression_model, spec.n_q)
    path = write_source(
        Path(out),
        spec=spec,
        lm_state=lm_state,
        codec_module=traced,
        half=half,
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, weights={weights_state}, fp16={half})")
    print(f"next: python -m appliance.pack {path}   (needs torch, onnx and onnxruntime)")
    return path


def _export_bundle(
    bundle_path: Path,
    out: Path,
    *,
    name: str | None,
    note: str,
    half: bool,
    card: dict | None,
) -> Path:
    """Export a bundle's already-selected snapshot; no Dora solver or XP is involved."""
    from macchina_ricordi.models.bundle import read_bundle
    from macchina_ricordi.models.bundle_runtime import load_bundle_model

    bundle = read_bundle(bundle_path, verify=True)
    model, _source = load_bundle_model(bundle_path, device="cpu")

    label = name or bundle.name
    spec = _bundle_spec(
        bundle.model_config,
        bundle.audio,
        name=label,
        origin=bundle_path.name,
        note=note,
    )
    if card:
        check_card(spec, card)
        spec = replace(spec, model=dict(card))

    lm_state = {k: v.cpu() for k, v in model.lm.state_dict().items()}
    traced = _trace_decoder(model.compression_model, spec.n_q)

    path = write_source(
        Path(out),
        spec=spec,
        lm_state=lm_state,
        codec_module=traced,
        half=half,
    )
    print(f"wrote {path}  ({path.stat().st_size / 1e6:.1f} MB, weights={bundle.selection}, fp16={half})")
    print(f"next: python -m appliance.pack {path}   (needs torch, onnx and onnxruntime)")
    return path


def _label_for(xp_path: Path, sig: str) -> str:
    meta = xp_path / "macchina_ricordi_train_metadata.json"
    if meta.exists():
        try:
            return str(json.loads(meta.read_text()).get("effective_label") or sig)
        except (OSError, json.JSONDecodeError):
            pass
    return sig


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sig", help="Dora experiment signature or a model bundle directory")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--checkpoint", default="checkpoint.th")
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--fp32", action="store_true", help="store weights at full precision")
    ap.add_argument(
        "--weights-state",
        choices=LEGACY_WEIGHTS_STATES,
        default="model",
        help="which checkpoint snapshot to export for a legacy Dora checkpoint (bundles carry their own)",
    )
    ap.add_argument(
        "--ema",
        action="store_true",
        help="legacy alias for --weights-state ema (no longer the default)",
    )
    ap.add_argument(
        "--card",
        type=Path,
        default=None,
        help="the model record as JSON (written by `model export --box`); goes into spec.json",
    )
    args = ap.parse_args()
    card = json.loads(args.card.read_text(encoding="utf-8")) if args.card else None
    use_ema = True if args.ema else None
    export(
        args.sig,
        args.out,
        checkpoint=args.checkpoint,
        name=args.name,
        note=args.note,
        half=not args.fp32,
        card=card,
        weights_state=args.weights_state,
        use_ema=use_ema,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
