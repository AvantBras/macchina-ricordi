from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..audio_files import DEFAULT_EXTENSIONS, iter_audio_files, parse_extensions
from ..cli.common import MacchinaRicordiHelpFormatter, write_csv
from .captioner import (
    DEFAULT_CAPTIONER_BACKEND,
    DEFAULT_CAPTIONER_DEVICE,
    DEFAULT_CAPTIONER_DTYPE,
    DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
    DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
    DEFAULT_CAPTIONER_MODEL_ID,
    DEFAULT_CAPTIONER_PROMPT_PROFILE,
    DEFAULT_CAPTIONER_QUANTIZATION,
    DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
    SUPPORTED_CAPTIONER_BACKENDS,
    CaptionerBackendUnavailable,
    attach_captioner_witness,
)
from .features import analyze_audio
from .render import (
    default_json_report_path,
    default_markdown_report_path,
    render_dossier_json,
    render_dossier_markdown,
)
from .semantic import (
    DEFAULT_SEMANTIC_BACKEND,
    DEFAULT_SEMANTIC_DEVICE,
    DEFAULT_SEMANTIC_MODEL_VERSION,
    SemanticBackendUnavailable,
    attach_semantic_probes,
)
from .spectrogram import (
    attach_spectrogram_path,
    default_spectrogram_path,
    render_mel_spectrogram_png_from_file,
)


def _analysis_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "sr": args.sr,
        "hop_length": args.hop_length,
        "res_type": args.res_type,
        "tempo_method": args.tempo_method,
        "window_seconds": args.window_seconds,
        "window_hop_seconds": args.window_hop_seconds,
    }


def _semantic_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "semantic_probes": getattr(args, "semantic_probes", False),
        "semantic_backend": getattr(args, "semantic_backend", DEFAULT_SEMANTIC_BACKEND),
        "semantic_model_version": getattr(args, "semantic_model_version", DEFAULT_SEMANTIC_MODEL_VERSION),
        "semantic_device": getattr(args, "semantic_device", DEFAULT_SEMANTIC_DEVICE),
        "semantic_model_path": getattr(args, "semantic_model_path", "") or None,
    }


def _captioner_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "captioner_witness": getattr(args, "captioner_witness", False),
        "captioner_backend": getattr(args, "captioner_backend", DEFAULT_CAPTIONER_BACKEND),
        "captioner_model_id": getattr(args, "captioner_model_id", DEFAULT_CAPTIONER_MODEL_ID),
        "captioner_device": getattr(args, "captioner_device", DEFAULT_CAPTIONER_DEVICE),
        "captioner_dtype": getattr(args, "captioner_dtype", DEFAULT_CAPTIONER_DTYPE),
        "captioner_quantization": getattr(args, "captioner_quantization", DEFAULT_CAPTIONER_QUANTIZATION),
        "captioner_max_new_tokens": getattr(args, "captioner_max_new_tokens", DEFAULT_CAPTIONER_MAX_NEW_TOKENS),
        "captioner_max_audio_seconds": getattr(
            args, "captioner_max_audio_seconds", DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS
        ),
        "captioner_prompt_profile": getattr(args, "captioner_prompt_profile", DEFAULT_CAPTIONER_PROMPT_PROFILE),
        "captioner_trust_remote_code": getattr(
            args,
            "captioner_trust_remote_code",
            DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
        ),
    }


def _attach_semantic_for_cli(dossier: object, args: argparse.Namespace) -> object:
    if not getattr(args, "semantic_probes", False):
        return dossier
    try:
        return attach_semantic_probes(
            dossier,
            audio_path=args.audio_path,
            backend=args.semantic_backend,
            model_version=args.semantic_model_version,
            device=args.semantic_device,
            model_path=args.semantic_model_path or None,
        )
    except SemanticBackendUnavailable as exc:
        raise SystemExit(f"semantic probes unavailable: {exc}") from exc


def _attach_captioner_for_cli(dossier: object, args: argparse.Namespace) -> object:
    if not getattr(args, "captioner_witness", False):
        return dossier
    try:
        return attach_captioner_witness(
            dossier,
            audio_path=args.audio_path,
            backend=args.captioner_backend,
            model_id=args.captioner_model_id,
            device=args.captioner_device,
            dtype=args.captioner_dtype,
            quantization=getattr(args, "captioner_quantization", DEFAULT_CAPTIONER_QUANTIZATION),
            max_new_tokens=args.captioner_max_new_tokens,
            max_audio_seconds=args.captioner_max_audio_seconds,
            prompt_profile=args.captioner_prompt_profile,
            trust_remote_code=args.captioner_trust_remote_code,
        )
    except CaptionerBackendUnavailable as exc:
        raise SystemExit(f"captioner witness unavailable: {exc}") from exc


def cmd_listen_file(args: argparse.Namespace) -> int:
    dossier = analyze_audio(args.audio_path, **_analysis_kwargs(args))
    dossier = _attach_semantic_for_cli(dossier, args)
    dossier = _attach_captioner_for_cli(dossier, args)

    if args.sidecars and getattr(args, "spectrogram", False):
        spectrogram_path = default_spectrogram_path(args.audio_path).resolve()
        render_mel_spectrogram_png_from_file(
            args.audio_path,
            output_path=spectrogram_path,
            sr=args.sr,
            res_type=args.res_type,
        )
        dossier = attach_spectrogram_path(dossier, path=spectrogram_path.name)

    json_text = render_dossier_json(dossier)
    markdown_text = render_dossier_markdown(dossier)
    output_path = Path(args.output).expanduser().resolve() if args.output else None
    markdown_path = Path(args.markdown).expanduser().resolve() if args.markdown else None
    if args.sidecars:
        if output_path is None:
            output_path = default_json_report_path(args.audio_path).resolve()
        if markdown_path is None:
            markdown_path = default_markdown_report_path(args.audio_path).resolve()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text, encoding="utf-8")
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown_text, encoding="utf-8")

    if args.json:
        print(json_text)
    else:
        print(markdown_text, end="")
    return 0


def cmd_listen_ask(args: argparse.Namespace) -> int:
    from .interrogate import (
        append_interrogation_report,
        interrogate_audio,
        render_interrogation_markdown,
    )

    dossier = analyze_audio(args.audio_path, **_analysis_kwargs(args))
    try:
        report = interrogate_audio(
            args.audio_path,
            args.questions,
            dossier=dossier,
            backend=args.captioner_backend,
            model_id=args.captioner_model_id,
            device=args.captioner_device,
            dtype=args.captioner_dtype,
            quantization=args.captioner_quantization,
            max_new_tokens=args.captioner_max_new_tokens,
            max_audio_seconds=args.captioner_max_audio_seconds,
            trust_remote_code=args.captioner_trust_remote_code,
        )
    except CaptionerBackendUnavailable as exc:
        raise SystemExit(f"captioner backend unavailable: {exc}") from exc
    if args.sidecar:
        append_interrogation_report(report, report_path=args.output or None)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_interrogation_markdown(report), end="")
    return 0


def cmd_listen_index(args: argparse.Namespace) -> int:
    from .archive import archive_extensions, build_archive_index, render_archive_index_json

    try:
        index = build_archive_index(
            args.root,
            output_path=args.output or None,
            recursive=args.recursive,
            extensions=archive_extensions(args.extensions),
            neighbors=args.neighbors,
            read_only=args.read_only,
            refresh=args.refresh,
            **_semantic_kwargs(args),
            **_captioner_kwargs(args),
            **_analysis_kwargs(args),
        )
    except SemanticBackendUnavailable as exc:
        raise SystemExit(f"semantic probes unavailable: {exc}") from exc
    except CaptionerBackendUnavailable as exc:
        raise SystemExit(f"captioner witness unavailable: {exc}") from exc
    if not args.output:
        print(render_archive_index_json(index))
    return 0


def cmd_listen_portrait(args: argparse.Namespace) -> int:
    from ..models.record import ModelNotFound, resolve_model
    from .archive import build_archive_portrait, render_portrait_markdown

    ref = args.ref
    try:
        resolved = resolve_model(ref)
    except ModelNotFound:
        resolved = None
    if resolved is not None and resolved.model is not None and resolved.model.name == ref:
        ref = resolved.model.sig

    try:
        portrait = build_archive_portrait(
            ref,
            generated_root=args.generated_root,
            neighbors=args.neighbors,
            read_only=args.read_only,
            refresh=args.refresh,
            **_semantic_kwargs(args),
            **_captioner_kwargs(args),
            **_analysis_kwargs(args),
        )
    except SemanticBackendUnavailable as exc:
        raise SystemExit(f"semantic probes unavailable: {exc}") from exc
    except CaptionerBackendUnavailable as exc:
        raise SystemExit(f"captioner witness unavailable: {exc}") from exc
    if args.json:
        print(json.dumps(portrait.to_dict(), indent=2))
    else:
        print(render_portrait_markdown(portrait), end="")
    return 0


def cmd_listen_describe(args: argparse.Namespace) -> int:
    from .describe import describe_audio, load_default_thresholds

    path = Path(args.audio_path).expanduser().resolve()
    load_default_thresholds()
    desc = describe_audio(path, sr=args.sr, res_type=args.res_type)

    if args.json:
        import json

        print(json.dumps({"path": str(path), **desc.to_dict()}))
    else:
        print(f"path: {path}")
        print(desc.to_sentence())
    return 0


def cmd_listen_diff(args: argparse.Namespace) -> int:
    from .describe import describe_audio, diff_audio, load_default_thresholds

    path_a = Path(args.path_a).expanduser().resolve()
    path_b = Path(args.path_b).expanduser().resolve()
    load_default_thresholds()

    if args.json:
        import json

        desc_a = describe_audio(path_a, sr=args.sr, res_type=args.res_type)
        desc_b = describe_audio(path_b, sr=args.sr, res_type=args.res_type)
        diff = diff_audio(path_a, path_b, sr=args.sr, res_type=args.res_type)
        print(
            json.dumps(
                {
                    "path_a": str(path_a),
                    "description_a": desc_a.to_dict(),
                    "path_b": str(path_b),
                    "description_b": desc_b.to_dict(),
                    "diff": diff,
                },
                indent=2,
            )
        )
    else:
        diff = diff_audio(path_a, path_b, sr=args.sr, res_type=args.res_type)
        print(diff)
    return 0


def cmd_listen_describe_calibrate(args: argparse.Namespace) -> int:
    from .describe import auto_calibrate

    auto_calibrate(args.directory, sr=args.sr, max_files=args.max_files)
    print(f"Calibrated from directory: {args.directory}")
    print("Thresholds saved to ~/.macchina_ricordi/describe_thresholds.json")
    print("Active for this session and all future sessions.")
    return 0


def cmd_listen_compare(args: argparse.Namespace) -> int:
    import json

    from ..scoring.similarity import compute_similarity_with_baseline
    from .describe import describe_audio, diff_audio, load_default_thresholds

    path_a = Path(args.path_a).expanduser().resolve()
    path_b = Path(args.path_b).expanduser().resolve()

    load_default_thresholds()
    desc_a = describe_audio(path_a, sr=args.sr, res_type=args.res_type)
    desc_b = describe_audio(path_b, sr=args.sr, res_type=args.res_type)
    diff = diff_audio(path_a, path_b, sr=args.sr, res_type=args.res_type)
    sim = compute_similarity_with_baseline(path_a, path_b, sr=args.sr, res_type=args.res_type)

    payload = {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "description_a": desc_a.to_dict(),
        "description_b": desc_b.to_dict(),
        "diff": diff,
        "similarity": {
            "mfcc_cosine": round(sim.mfcc_cosine, 4),
            "spectral_cosine": round(sim.spectral_cosine, 4),
            "rhythm_cosine": round(sim.rhythm_cosine, 4),
            "combined": round(sim.combined, 4),
            "self_similarity_a": round(sim.self_similarity_a, 4) if sim.self_similarity_a is not None else None,
            "self_similarity_b": round(sim.self_similarity_b, 4) if sim.self_similarity_b is not None else None,
        },
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"--- {path_a.name} ---")
        print(desc_a.to_sentence())
        print(f"\n--- {path_b.name} ---")
        print(desc_b.to_sentence())
        print("\n--- diff ---")
        print(diff)
        print("\n--- similarity ---")
        print(f"  combined={payload['similarity']['combined']:.4f}")
        if sim.self_similarity_a is not None:
            print(f"  self_sim_a={payload['similarity']['self_similarity_a']:.4f}")
        if sim.self_similarity_b is not None:
            print(f"  self_sim_b={payload['similarity']['self_similarity_b']:.4f}")
    return 0


def cmd_listen_describe_dir(args: argparse.Namespace) -> int:
    import json

    from .describe import AudioDescription, describe_audio, load_default_thresholds

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    load_default_thresholds()
    extensions = parse_extensions(args.extensions) if args.extensions else DEFAULT_EXTENSIONS
    files = list(iter_audio_files(input_dir, recursive=args.recursive, extensions=extensions))

    rows: list[dict[str, Any]] = []
    empty_description = {field: "" for field in AudioDescription.fieldnames()}
    for path in files:
        try:
            desc = describe_audio(path, sr=args.sr, res_type=args.res_type)
            rows.append({"path": str(path), **desc.to_dict(), "error": ""})
        except Exception as exc:
            rows.append({"path": str(path), **empty_description, "error": str(exc)})

    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix == ".json" or args.json:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2))
    else:
        fieldnames = ["path", *AudioDescription.fieldnames(), "error"]
        write_csv(output_path, fieldnames, rows)

    print(f"Described {len(rows)} files → {output_path}")
    return 0


def _add_analysis_args(parser: argparse.ArgumentParser) -> None:
    analysis = parser.add_argument_group("deterministic analysis")
    analysis.add_argument("--sr", type=int, default=32_000, help="Target sample rate")
    analysis.add_argument("--hop-length", type=int, default=512, help="Hop length for feature analysis")
    analysis.add_argument("--res-type", default="soxr_hq", help="librosa resampler type")
    analysis.add_argument(
        "--tempo-method",
        choices=("autocorr", "tempogram"),
        default="tempogram",
        help="Tempo clarity penalty backend",
    )
    analysis.add_argument("--window-seconds", type=float, default=2.0, help="Fixed timeline window size")
    analysis.add_argument("--window-hop-seconds", type=float, default=1.0, help="Fixed timeline window hop")


def _add_semantic_args(parser: argparse.ArgumentParser) -> None:
    semantic = parser.add_argument_group("semantic probes")
    semantic.add_argument(
        "--semantic-probes",
        action="store_true",
        help="Run optional CLAP audio-text semantic probes; requires the semantic-listening extra",
    )
    semantic.add_argument(
        "--semantic-backend",
        default=DEFAULT_SEMANTIC_BACKEND,
        choices=("msclap",),
        help="Semantic probe backend",
    )
    semantic.add_argument(
        "--semantic-model-version",
        default=DEFAULT_SEMANTIC_MODEL_VERSION,
        help="Semantic backend model version",
    )
    semantic.add_argument(
        "--semantic-device",
        default=DEFAULT_SEMANTIC_DEVICE,
        choices=("auto", "cpu", "cuda"),
        help="Semantic backend device",
    )
    semantic.add_argument(
        "--semantic-model-path",
        default="",
        help="Optional semantic backend model weight path",
    )


def _add_captioner_args(parser: argparse.ArgumentParser, *, witness: bool = True) -> None:
    captioner = parser.add_argument_group("captioner witness" if witness else "captioner backend")
    if witness:
        captioner.add_argument(
            "--captioner-witness",
            action="store_true",
            help=(
                "Run optional audio-language captioner witness; requires the captioner-witness extra and a large model"
            ),
        )
    captioner.add_argument(
        "--captioner-backend",
        default=DEFAULT_CAPTIONER_BACKEND,
        choices=SUPPORTED_CAPTIONER_BACKENDS,
        help="Captioner backend",
    )
    captioner.add_argument(
        "--captioner-model-id",
        default=DEFAULT_CAPTIONER_MODEL_ID,
        help="Captioner model id (default: backend-specific)",
    )
    captioner.add_argument(
        "--captioner-device",
        default=DEFAULT_CAPTIONER_DEVICE,
        choices=("auto", "cpu", "cuda"),
        help="Captioner backend device",
    )
    captioner.add_argument(
        "--captioner-dtype",
        default=DEFAULT_CAPTIONER_DTYPE,
        choices=("auto", "float32", "float16", "bfloat16"),
        help="Captioner model dtype",
    )
    captioner.add_argument(
        "--captioner-quantization",
        default=DEFAULT_CAPTIONER_QUANTIZATION,
        choices=("auto", "none", "bnb-4bit"),
        help="Captioner weight quantization; auto uses bnb-4bit on CUDA and none on CPU",
    )
    captioner.add_argument(
        "--captioner-max-new-tokens",
        type=int,
        default=DEFAULT_CAPTIONER_MAX_NEW_TOKENS,
        help="Maximum captioner tokens to generate",
    )
    captioner.add_argument(
        "--captioner-max-audio-seconds",
        type=float,
        default=DEFAULT_CAPTIONER_MAX_AUDIO_SECONDS,
        help="Maximum audio seconds to pass to captioner",
    )
    if witness:
        captioner.add_argument(
            "--captioner-prompt-profile",
            default=DEFAULT_CAPTIONER_PROMPT_PROFILE,
            choices=(DEFAULT_CAPTIONER_PROMPT_PROFILE,),
            help="Captioner witness prompt profile",
        )
    captioner.add_argument(
        "--captioner-trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_CAPTIONER_TRUST_REMOTE_CODE,
        help="Allow captioner model remote code",
    )


def _add_archive_build_args(parser: argparse.ArgumentParser) -> None:
    archive = parser.add_argument_group("archive building")
    archive.add_argument("--read-only", action="store_true", help="Skip audio without existing listening sidecars")
    archive.add_argument("--refresh", action="store_true", help="Re-analyze and overwrite existing listening sidecars")
    archive.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated audio extensions to scan",
    )
    archive.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan archive directories recursively",
    )
    _add_analysis_args(parser)
    _add_semantic_args(parser)
    _add_captioner_args(parser)


def register_listening_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    listen = subparsers.add_parser(
        "listen",
        help="Text about audio: dossiers, descriptions, diffs, questions, portraits",
        description=(
            "Text out: listening dossiers, perceptual descriptions and diffs, free questions to an audio-language "
            "model, archive indexes and model portraits. Numbers are `score`; verdicts are `judge`."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli listen file sample.wav --sidecars
  python -m macchina_ricordi.cli listen describe sample.wav
  python -m macchina_ricordi.cli listen compare before.wav after.wav
  python -m macchina_ricordi.cli listen index artifacts/generated_audio --output listening_index.json
  python -m macchina_ricordi.cli listen portrait ec64a919""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_subparsers = listen.add_subparsers(dest="listen_command", required=True)

    listen_file = listen_subparsers.add_parser(
        "file",
        help="Create a listening dossier for one audio file",
        description="Analyze one audio file and render a Markdown or JSON listening dossier.",
        epilog="""examples:
  python -m macchina_ricordi.cli listen file sample.wav
  python -m macchina_ricordi.cli listen file sample.wav --sidecars
  python -m macchina_ricordi.cli listen file sample.wav --semantic-probes --json""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_file.add_argument("audio_path", help="Path to audio file")
    listen_file.add_argument("--output", default="", help="Optional JSON report path")
    listen_file.add_argument("--markdown", default="", help="Optional Markdown report path")
    listen_file.add_argument("--sidecars", action="store_true", help="Write default JSON and Markdown sidecars")
    listen_file.add_argument(
        "--spectrogram",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --sidecars, also write a mel-spectrogram PNG next to the audio file",
    )
    listen_file.add_argument("--json", action="store_true", help="Print JSON to stdout instead of Markdown")
    _add_analysis_args(listen_file)
    _add_semantic_args(listen_file)
    _add_captioner_args(listen_file)
    listen_file.set_defaults(func=cmd_listen_file)

    listen_ask = listen_subparsers.add_parser(
        "ask",
        help="Ask an audio-language model ad-hoc questions about one audio file",
        description=(
            "Interrogate one audio file with free-form questions; answers go through the same "
            "claim cross-examination as captioner witnesses and append to an interrogation sidecar."
        ),
        epilog="""examples:
  python -m macchina_ricordi.cli listen ask sample.wav "What changes at the halfway point?"
  python -m macchina_ricordi.cli listen ask sample.wav "Is there a steady pulse?" "How dense is the texture?"
  python -m macchina_ricordi.cli listen ask sample.wav "Describe the transients"
  --captioner-backend audio-flamingo-3""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_ask.add_argument("audio_path", help="Path to audio file")
    listen_ask.add_argument("questions", nargs="+", help="One or more questions to ask about the audio")
    listen_ask.add_argument("--output", default="", help="Optional interrogation report JSON path")
    listen_ask.add_argument(
        "--sidecar",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append the exchange to an interrogation transcript sidecar next to the audio file",
    )
    listen_ask.add_argument("--json", action="store_true", help="Print JSON to stdout instead of Markdown")
    _add_analysis_args(listen_ask)
    _add_captioner_args(listen_ask, witness=False)
    listen_ask.set_defaults(func=cmd_listen_ask)

    listen_index = listen_subparsers.add_parser(
        "index",
        help="Build an archive-level listening memory index",
        description=(
            "Scan an audio archive, create missing listening sidecars by default, and compute neighbors/novelty."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_index.add_argument("root", help="Audio archive root")
    listen_index.add_argument("--output", default="", help="Optional archive index JSON path")
    listen_index.add_argument("--neighbors", type=int, default=3, help="Nearest neighbors per entry")
    _add_archive_build_args(listen_index)
    listen_index.set_defaults(func=cmd_listen_index)

    listen_describe = listen_subparsers.add_parser(
        "describe",
        help="Describe audio in perceptual terms",
        description="Generate compact deterministic perceptual labels for one audio file.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_describe.add_argument("audio_path", help="Path to audio file")
    listen_describe.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    listen_describe.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    listen_describe.add_argument("--json", action="store_true", help="Output as JSON")
    listen_describe.set_defaults(func=cmd_listen_describe)

    listen_diff = listen_subparsers.add_parser(
        "diff",
        help="Describe what changed between two audio files",
        description="Describe deterministic feature changes between two audio files.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_diff.add_argument("path_a", help="First audio file")
    listen_diff.add_argument("path_b", help="Second audio file")
    listen_diff.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    listen_diff.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    listen_diff.add_argument("--json", action="store_true", help="Output as JSON")
    listen_diff.set_defaults(func=cmd_listen_diff)

    desc_calibrate = listen_subparsers.add_parser(
        "describe-calibrate",
        help="Calibrate perceptual thresholds from a directory and save for auto-loading",
        description="Calibrate descriptor thresholds from a recursive audio directory.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    desc_calibrate.add_argument("directory", help="Directory of audio files (recursive)")
    desc_calibrate.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    desc_calibrate.add_argument("--max-files", type=int, default=100, help="Max files to sample")
    desc_calibrate.set_defaults(func=cmd_listen_describe_calibrate)

    listen_compare = listen_subparsers.add_parser(
        "compare",
        help="Full comparison: similarity, descriptions, and diff between two files",
        description="Run similarity, per-file descriptions, and feature-diff analysis together.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_compare.add_argument("path_a", help="First audio file")
    listen_compare.add_argument("path_b", help="Second audio file")
    listen_compare.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    listen_compare.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    listen_compare.add_argument("--json", action="store_true", help="Output as JSON")
    listen_compare.set_defaults(func=cmd_listen_compare)

    describe_dir = listen_subparsers.add_parser(
        "describe-dir",
        help="Describe all audio files in a directory",
        description="Describe every audio file in a directory and write CSV or JSON output.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    describe_dir.add_argument("input_dir", help="Directory containing audio files")
    describe_dir.add_argument(
        "--extensions",
        default="",
        help="Comma-separated list, e.g. wav,flac,mp3 (default: all audio)",
    )
    describe_dir.add_argument("--sr", type=int, default=16_000, help="Target sample rate")
    describe_dir.add_argument("--res-type", default="soxr_qq", help="librosa resampler type")
    describe_dir.add_argument("--recursive", action="store_true", help="Include subdirectories")
    describe_dir.add_argument("--output", default="artifacts/scores/descriptions.csv", help="Output CSV or JSON path")
    describe_dir.add_argument("--json", action="store_true", help="Force JSON output")
    describe_dir.set_defaults(func=cmd_listen_describe_dir)

    listen_portrait = listen_subparsers.add_parser(
        "portrait",
        help="Summarize generated audio for one model or XP",
        description=(
            "Build a listening portrait for generated audio matching an XP signature, path, checkpoint, or label."
        ),
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    listen_portrait.add_argument("ref", help="XP path, XP signature, checkpoint path, model label, or model name")
    listen_portrait.add_argument(
        "--generated-root",
        default="artifacts/runs",
        help="Generated audio root to search",
    )
    listen_portrait.add_argument("--json", action="store_true", help="Print JSON instead of Markdown")
    listen_portrait.add_argument("--neighbors", type=int, default=3, help="Nearest neighbors per indexed entry")
    listen_portrait.add_argument(
        "--read-only",
        action="store_true",
        help="Skip audio without existing listening sidecars",
    )
    listen_portrait.add_argument("--refresh", action="store_true", help="Re-analyze and overwrite listening sidecars")
    _add_analysis_args(listen_portrait)
    _add_semantic_args(listen_portrait)
    _add_captioner_args(listen_portrait)
    listen_portrait.set_defaults(func=cmd_listen_portrait)
