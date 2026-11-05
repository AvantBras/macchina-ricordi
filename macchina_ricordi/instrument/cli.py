"""`play`: generate, sweep, stream, monitor, duet, variations, replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ..cli.common import MacchinaRicordiHelpFormatter, bool_env
from ..models.cli import (
    apply_model_reference,
    model_folder_or_text,
)
from ..musicgen.backend import DEFAULT_EXTEND_STRIDE_SECONDS
from ..musicgen.config import (
    recipe_choices,
)
from .replay import load_source_run, resolve_replay_metadata, run_replay_audit


def cmd_play_replay(args: argparse.Namespace) -> int:
    from ..sessions.record import SessionNotFound

    try:
        metadata_path = resolve_replay_metadata(args.metadata_path)
    except SessionNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    source = load_source_run(metadata_path)
    report_path = Path(args.report_path).expanduser().resolve() if args.report_path else None
    report = run_replay_audit(
        source,
        output_dir=args.output_dir,
        report_path=report_path,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "all_match": report.all_match,
                    "report_path": str(report.report_path),
                    "replay_dir": str(report.replay_run.output_dir),
                    "samples": [
                        {
                            "index": sample.index,
                            "matches": sample.matches,
                            "source_sha256": sample.source_sha256,
                            "replay_sha256": sample.replay_sha256,
                        }
                        for sample in report.samples
                    ],
                },
                indent=2,
            )
        )
    else:
        print(f"source_metadata: {report.source.metadata_path}")
        print(f"source_run_id: {report.source.run_id}")
        print(f"replay_dir: {report.replay_run.output_dir}")
        print(f"report: {report.report_path}")
        print(f"all_match: {report.all_match}")
        for sample in report.samples:
            status = "match" if sample.matches else "MISMATCH"
            print(
                f"sample_{sample.index:03d}: {status} "
                f"source={sample.source_sha256[:10]} "
                f"replay={sample.replay_sha256[:10]}"
            )

    if args.strict and not report.all_match:
        return 1
    return 0


def _print_generation_result(args: argparse.Namespace, request, result) -> None:  # type: ignore[no-untyped-def]
    from ..instrument.saving import save_generation_run

    print(f"backend: {result.backend}")
    print(f"mode: {request.mode}")
    print(f"model_id: {result.model_id}")
    print(f"device: {result.device}")
    print(f"sample_rate: {result.sample_rate}")
    print(f"audio_shape: {tuple(result.audio_values.shape)}")
    source = result.metadata.get("source", {})
    if isinstance(source, dict):
        source_kind = source.get("kind")
        if source_kind is not None:
            print(f"source: {source_kind}")
        checkpoint_path = source.get("checkpoint_path")
        if checkpoint_path is not None:
            print(f"checkpoint_path: {checkpoint_path}")
    for warning in result.warnings:
        print(f"warning: {warning}")

    if args.save_output:
        saved = save_generation_run(
            result,
            request,
            output_dir=args.output_dir,
            prefix=args.output_prefix,
            output_layout=args.output_layout,
        )
        print(f"saved_files: {len(saved.audio_paths)}")
        print(f"saved_dir: {saved.output_dir}")
        print(f"metadata_json: {saved.metadata_path}")
        if args.output_layout == "session":
            print(f"session: {saved.output_dir.name}")
        if getattr(args, "listen", False):
            from ..instrument.listening import write_listening_sidecars_for_paths

            listening_results = write_listening_sidecars_for_paths(saved.audio_paths)
            listening_reports = [result for result in listening_results if result.ok]
            listening_failed = [result for result in listening_results if not result.ok]
            print(f"listening_reports: {len(listening_reports)}")
            print(f"listening_failed: {len(listening_failed)}")
            for listening_result in listening_results:
                if listening_result.ok:
                    print(f"listening_report: {listening_result.report_path}")
                    print(f"listening_markdown: {listening_result.markdown_path}")
                else:
                    print(f"listening_error: {listening_result.audio_path}: {listening_result.error}")

    try:
        import torch

        if result.device.startswith("cuda"):
            print(f"vram_used_gb: {torch.cuda.max_memory_allocated() / 1e9:.2f}")
    except Exception:
        pass


def cmd_play_generate(args: argparse.Namespace) -> int:
    from ..musicgen.backend import GenerationRequest, build_audiocraft_backend

    if getattr(args, "listen", False) and not args.save_output:
        raise ValueError("--listen requires --save-output so generated WAVs can be analyzed.")
    apply_model_reference(args, positional_attr="model")

    prompt_audio = getattr(args, "prompt_audio", None)
    prompt_duration = getattr(args, "prompt_duration_seconds", None)
    backend = build_audiocraft_backend(
        model_id=args.model_id,
        device=args.device,
        profile_name=args.profile,
        xp_path=args.xp_path,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
    )
    request = GenerationRequest(
        prompts=[""],
        batch_size=args.num_samples,
        mode="continuation" if args.generation_mode == "continue" else args.generation_mode,
        duration_seconds=args.duration_seconds,
        seed=args.seed,
        prompt_audio_path=prompt_audio,
        prompt_start_seconds=getattr(args, "prompt_start_seconds", 0.0),
        prompt_duration_seconds=prompt_duration,
        use_sampling=args.use_sampling,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=getattr(args, "min_p", 0.0),
        temperature=args.temperature,
        cfg_coef=args.cfg_coef,
        extend_stride=args.extend_stride,
    )
    result = backend.generate(request)
    _print_generation_result(args, request, result)
    return 0


def cmd_play_sweep(args: argparse.Namespace) -> int:
    from ..instrument.sweep import run_musicgen_sweep

    apply_model_reference(args)
    run = run_musicgen_sweep(
        mode=args.sweep_mode,
        model_id=args.model_id,
        profile_name=args.profile,
        device=args.device,
        xp_path=args.xp_path,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        prompt_audio_path=args.prompt_audio,
        prompt_start_seconds=args.prompt_start_seconds,
        prompt_duration_seconds=args.prompt_duration_seconds,
        num_samples=args.num_samples,
        duration_seconds=args.duration_seconds,
        seed=args.seed,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        cfg_coef=args.cfg_coef,
        use_sampling=args.use_sampling,
        extend_stride=args.extend_stride,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        duration_seconds_values=args.duration_seconds_values,
        prompt_duration_seconds_values=args.prompt_duration_seconds_values,
        temperature_values=args.temperature_values,
        top_k_values=args.top_k_values,
        top_p_values=args.top_p_values,
        cfg_coef_values=args.cfg_coef_values,
        extend_stride_values=args.extend_stride_values,
        seed_values=args.seed_values,
        use_sampling_values=args.use_sampling_values,
        max_combinations=args.max_combinations,
        dry_run=args.dry_run,
        score=args.score,
        index_html=args.index_html,
        listen=args.listen,
    )
    print(f"sweep_id: {run.sweep_id}")
    print(f"mode: {args.sweep_mode}")
    print(f"combinations: {run.combo_count}")
    print(f"samples_written: {run.sample_count}")
    print(f"dry_run: {run.dry_run}")
    print(f"sweep_dir: {run.output_dir}")
    print(f"sweep_plan: {run.plan_path}")
    print(f"sweep_report: {run.report_path}")
    print(f"index_csv: {run.index_csv_path}")
    if run.index_html_path is not None:
        print(f"index_html: {run.index_html_path}")
    return 0


def cmd_play_monitor(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ..instrument.stream.monitor import latest_session_dir, run_monitor
    from ..sessions.record import SessionNotFound
    from ..sessions.store import find_session, latest_stream_dir

    if args.session:
        session = Path(args.session).expanduser()
        if not session.is_dir():
            try:
                session = find_session(args.session).folder
            except SessionNotFound as exc:
                raise FileNotFoundError(f"No stream session {args.session!r}: {exc}") from exc
    else:
        found = latest_session_dir(Path(args.streams_dir).expanduser()) if args.streams_dir else latest_stream_dir()
        if found is None:
            print(f"no_sessions: no stream session with a performance.jsonl under {args.streams_dir or 'artifacts/'}")
            return 1
        session = found

    run_monitor(
        session,
        interval=float(args.interval),
        history=max(1, int(args.history)),
        events_height=max(0, int(args.events)),
        follow=args.follow,
        once=bool(args.once),
        width=int(args.width) if args.width else None,
    )
    return 0


def cmd_play_stream(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ..instrument.stream.playback import NullPlaybackWorker, PlaybackWorker
    from ..instrument.stream.session import (
        DEFAULT_NOVELTY_CALIBRATION,
        PLAYLIST_FILENAME,
        StreamControl,
        build_stream_session_paths,
        fresh_run_id,
        load_novelty_calibration,
        normalize_model_xp_path,
    )
    from ..musicgen.shared import write_text_atomic

    apply_model_reference(args)
    if args.seed_mode == "excerpt" and args.seed_audio is None:
        raise ValueError("--seed-mode excerpt requires --seed-audio PATH.")
    if not 1.0 <= float(args.completion_duration) <= 60.0:
        raise ValueError("--completion-duration must be between 1 and 60 seconds.")

    seed_audio = Path(args.seed_audio).expanduser().resolve() if args.seed_audio is not None else None
    if seed_audio is not None and not seed_audio.exists():
        raise FileNotFoundError(f"Seed audio does not exist: {seed_audio}")

    # Digit keys 1-9 pick from here; the loop compares resolved paths when it
    # decides whether a swap is a swap, so resolve once up front.
    playlist = [
        normalize_model_xp_path(model_folder_or_text(entry.strip())) or entry
        for entry in (args.playlist or "").split(",")
        if entry.strip()
    ]
    if len(playlist) > 9:
        print(f"playlist_truncated: {len(playlist)} entries given; only the first 9 get a digit key")
    for entry in playlist:
        if not Path(entry).is_dir():
            print(f"playlist_missing: {entry} is not a directory; that key will be skipped mid-session")
    # A playlist with no --xp-path would otherwise start on the base model and
    # only reach slot 1 once you pressed it.
    if playlist and not args.xp_path:
        args.xp_path = playlist[0]
        print(f"playlist_start: {playlist[0]}")

    initial_control = StreamControl(
        command="continue",
        tau_seconds=float(args.tau),
        completion_duration_seconds=float(args.completion_duration),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        min_p=float(args.min_p),
        cfg_coef=float(args.cfg_coef),
        extend_stride=float(args.extend_stride),
        model_xp=args.xp_path if args.xp_path else None,
        candidates=int(args.candidates),
        novelty_weight=float(args.novelty),
        stuck_weight=float(args.stuck_weight),
        stuck_kick_rounds=int(args.stuck_kick),
        motif_dir=args.motif_dir if args.motif_dir else None,
        motif_period=int(args.motif_period),
        motif_mix=float(args.motif_mix),
        tail_rescue=not args.no_tail_rescue,
        prompt_tilt_low_db=float(args.prompt_tilt_low),
        prompt_tilt_mid_db=float(args.prompt_tilt_mid),
        prompt_tilt_high_db=float(args.prompt_tilt_high),
        prompt_rate=float(args.prompt_rate),
        prompt_rate_mode=args.prompt_rate_mode,
        round_delay_seconds=float(args.round_delay_seconds),
    )

    run_id = fresh_run_id()
    if args.output_dir:
        paths = build_stream_session_paths(Path(args.output_dir).expanduser(), run_id=run_id)
    else:
        from ..sessions.store import new_session_dir

        folder = new_session_dir("stream", run_id=run_id)
        paths = build_stream_session_paths(folder.parent, run_id=run_id, name=folder.name)
    if playlist:
        write_text_atomic(paths.session_dir / PLAYLIST_FILENAME, json.dumps(playlist, indent=2) + "\n")

    if args.play:
        try:
            player: PlaybackWorker | NullPlaybackWorker = PlaybackWorker(
                crossfade_seconds=float(args.crossfade_seconds),
            )
        except RuntimeError as exc:
            print(f"playback_unavailable: {exc}")
            player = NullPlaybackWorker()
    else:
        player = NullPlaybackWorker()

    print(f"session_dir: {paths.session_dir}")
    print(f"control_file: {paths.control_path}")
    print(f"performance_log: {paths.performance_path}")
    print(f"manifest: {paths.manifest_path}")
    print(f"playback: {'sounddevice' if isinstance(player, PlaybackWorker) else 'off'}")
    print("hint: edit the control file mid-session to steer the next round")

    calibration = None
    if not args.no_novelty_calibration:
        candidate = Path(args.novelty_calibration) if args.novelty_calibration else DEFAULT_NOVELTY_CALIBRATION
        calibration = load_novelty_calibration(candidate)
        if calibration is not None:
            print(f"novelty_calibration: {candidate} (corpus: {calibration.source})")
        elif args.novelty_calibration:
            print(f"novelty_calibration_unreadable: {candidate}; novelty falls back to raw cosine")
        else:
            print(
                f"novelty_calibration: none at {candidate}; novelty is raw cosine (near-zero range). "
                "Build one with: score sim-calibrate analysis/calibration_corpus "
                f"--output {candidate}"
            )

    tui = None
    if args.tui:
        from ..instrument.stream.tui import StreamStatus, StreamTui

        tui = StreamTui(
            paths.session_dir,
            paths.control_path,
            status=StreamStatus(),
            follow=args.follow,
            blind=args.blind,
            playlist=playlist,
            playback=player,
        )
        if not tui.terminal.is_interactive():
            print("tui_unavailable: stdin/stdout is not an interactive terminal; using plain output")
            tui = None
        else:
            print(f"worker_log: {paths.session_dir / 'worker.log'}")

    try:
        if tui is None:
            result = _run_stream(
                args, paths=paths, initial_control=initial_control, player=player, calibration=calibration
            )
        else:
            # The terminal is restored before the summary prints, so it lands on the
            # shell the user gets back rather than on the discarded alt-screen.
            with tui.terminal:
                tui.start()
                try:
                    result = _run_stream(
                        args,
                        paths=paths,
                        initial_control=initial_control,
                        player=player,
                        status=tui.status,
                        quiet=True,
                        calibration=calibration,
                    )
                finally:
                    tui.stop()
    finally:
        _seal_quietly(paths.session_dir)

    print(f"session: {paths.session_dir.name}")
    print(f"rounds_written: {result.rounds_written}")
    print(f"stopped_by: {result.stopped_by}")
    return 0


def _seal_quietly(folder: Any, *, parent: str | None = None) -> None:
    """Write session.json for a finished session; a failure here must not hide the session's own error."""
    from ..sessions.store import seal_session

    try:
        seal_session(folder, parent=parent)
    except Exception as exc:
        print(f"session_record_failed: {folder}: {exc}")


def _resume_from_args(args: argparse.Namespace) -> float | None:
    if args.resume_buffer_ahead_seconds is not None:
        return float(args.resume_buffer_ahead_seconds)
    if args.buffer_floor_ratio is not None and args.max_buffer_ahead_seconds is not None:
        return min(1.0, max(0.0, float(args.buffer_floor_ratio))) * float(args.max_buffer_ahead_seconds)
    return None


def _run_stream(
    args: argparse.Namespace,
    *,
    paths: Any,
    initial_control: Any,
    player: Any,
    status: Any = None,
    quiet: bool = False,
    calibration: Any = None,
) -> Any:
    from pathlib import Path

    from ..instrument.stream.session import run_stream_session

    seed_audio = Path(args.seed_audio).expanduser().resolve() if args.seed_audio is not None else None
    return run_stream_session(
        paths=paths,
        initial_control=initial_control,
        seed_mode=args.seed_mode,
        seed_audio=seed_audio,
        model_id=args.model_id,
        device=args.device,
        profile_name=args.profile,
        xp_path=args.xp_path,
        checkpoint=args.checkpoint,
        checkpoint_path=args.checkpoint_path,
        base_seed=args.seed,
        max_rounds=int(args.max_rounds),
        score_rounds=bool(args.score_rounds),
        player=player,
        max_buffer_ahead_seconds=(
            float(args.max_buffer_ahead_seconds) if args.max_buffer_ahead_seconds is not None else None
        ),
        resume_buffer_ahead_seconds=_resume_from_args(args),
        status=status,
        quiet=quiet,
        novelty_calibration=calibration,
    )


def _run_musicgen_script(script_name: str, script_args: list[str]) -> int:
    import subprocess
    from pathlib import Path

    from ..musicgen.config import musicgen_script_env
    from ..musicgen.runtime import load_musicgen_project_config, require_musicgen_python

    config = load_musicgen_project_config()
    script = config.repo_root / "scripts" / script_name
    if not script.exists():
        raise FileNotFoundError(f"Helper script not found: {script}")
    musicgen_python = str(Path(require_musicgen_python()).expanduser())
    completed = subprocess.run(
        [musicgen_python, str(script), *script_args],
        check=False,
        env=musicgen_script_env(),
    )
    return int(completed.returncode)


def _resolve_voice_args(args: argparse.Namespace, voice: str) -> list[str]:
    from pathlib import Path

    from ..instrument.provenance import resolve_generation_profile
    from ..musicgen.backend import resolve_generation_checkpoint

    xp_path = getattr(args, f"xp_{voice}")
    resolved_checkpoint = resolve_generation_checkpoint(
        xp_path=xp_path,
        checkpoint=getattr(args, f"checkpoint_{voice}", None) or "latest",
        checkpoint_path=None,
    )
    profile = resolve_generation_profile(
        profile_name=getattr(args, f"profile_{voice}", None),
        xp_path=xp_path,
        checkpoint_path=resolved_checkpoint,
    )
    out: list[str] = []
    if xp_path:
        out.extend([f"--xp-path-{voice}", str(Path(xp_path).expanduser().resolve())])
    if resolved_checkpoint is not None:
        out.extend([f"--checkpoint-path-{voice}", str(resolved_checkpoint)])
    if profile:
        out.extend([f"--profile-{voice}", str(profile)])
    return out


def _run_and_seal(script: str, script_args: list[str], output_dir: Any, *, parent: str | None = None) -> int:
    code = _run_musicgen_script(script, script_args)
    if code == 0:
        _seal_quietly(output_dir, parent=parent)
        print(f"session: {output_dir.name}")
    return code


def cmd_play_duet(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ..sessions.store import new_session_dir

    for voice in ("a", "b"):
        apply_model_reference(
            args, xp_attr=f"xp_{voice}", checkpoint_attr=f"checkpoint_{voice}", checkpoint_path_attr=None
        )
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else new_session_dir("duet")
    script_args = [
        "--output-dir",
        str(output_dir),
        "--rounds",
        str(args.rounds),
        "--round-duration",
        str(args.round_duration),
        "--tau",
        str(args.tau),
        "--candidates",
        str(args.candidates),
        "--pan-a",
        str(args.pan_a),
        "--pan-b",
        str(args.pan_b),
        "--gain-a",
        str(args.gain_a),
        "--gain-b",
        str(args.gain_b),
    ]
    if args.seed is not None:
        script_args.extend(["--seed", str(args.seed)])
    if args.device:
        script_args.extend(["--device", args.device])
    for voice in ("a", "b"):
        script_args.extend(_resolve_voice_args(args, voice))
        for flag, attr in (
            ("temperature", "temperature"),
            ("top-k", "top_k"),
            ("top-p", "top_p"),
            ("min-p", "min_p"),
            ("listen", "listen"),
        ):
            script_args.extend([f"--{flag}-{voice}", str(getattr(args, f"{attr}_{voice}"))])
        seed_audio = getattr(args, f"seed_audio_{voice}")
        if seed_audio:
            script_args.extend([f"--seed-audio-{voice}", str(Path(seed_audio).expanduser().resolve())])
    return _run_and_seal("run_musicgen_duet.py", script_args, output_dir)


def cmd_play_variations(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ..instrument.provenance import resolve_generation_profile
    from ..musicgen.backend import resolve_generation_checkpoint
    from ..sessions.record import SessionNotFound
    from ..sessions.store import find_take, new_session_dir

    apply_model_reference(args, checkpoint_path_attr=None)
    input_path = Path(args.input).expanduser().resolve()
    parent = None
    if not input_path.exists():
        try:
            parent_session, take = find_take(args.input)
        except SessionNotFound as exc:
            raise FileNotFoundError(f"Input is neither an audio file nor a take: {args.input} ({exc})") from exc
        if take.audio is None or not take.audio.exists():
            raise FileNotFoundError(f"Take {args.input} has no audio on disk")
        input_path = take.audio.resolve()
        parent = f"{parent_session.id}#{take.index}"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else new_session_dir("variations")
    resolved_checkpoint = resolve_generation_checkpoint(
        xp_path=args.xp_path,
        checkpoint=args.checkpoint or "latest",
        checkpoint_path=None,
    )
    profile = resolve_generation_profile(
        profile_name=args.profile,
        xp_path=args.xp_path,
        checkpoint_path=resolved_checkpoint,
    )
    script_args = [
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--n",
        str(args.n),
        "--freeze-books",
        str(args.freeze_books),
        "--keep-intro-seconds",
        str(args.keep_intro_seconds),
        "--temperature",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--min-p",
        str(args.min_p),
    ]
    if args.xp_path:
        script_args.extend(["--xp-path", str(Path(args.xp_path).expanduser().resolve())])
    if resolved_checkpoint is not None:
        script_args.extend(["--checkpoint-path", str(resolved_checkpoint)])
    if profile:
        script_args.extend(["--profile", str(profile)])
    if args.seconds is not None:
        script_args.extend(["--seconds", str(args.seconds)])
    if args.seed is not None:
        script_args.extend(["--seed", str(args.seed)])
    if args.device:
        script_args.extend(["--device", args.device])
    return _run_and_seal("run_musicgen_variations.py", script_args, output_dir, parent=parent)


def register_play_group(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    play = subparsers.add_parser(
        "play",
        help="Play models: generate, sweep, stream, duet, variations",
        description="Make sound with models: one-off takes, sweeps over settings, the live stream and its monitor, "
        "two-model duets, variations of a piece, and byte-exact replays.",
        epilog="""examples:
  python -m macchina_ricordi.cli play generate midwife@best --num-samples 4 --listen
  python -m macchina_ricordi.cli play continue midwife@best --prompt-audio seed.wav
  python -m macchina_ricordi.cli play stream --xp-path midwife@best --tui
  python -m macchina_ricordi.cli play sweep --dry-run""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    play_subparsers = play.add_subparsers(dest="play_command", required=True)

    def add_generation_common(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "model",
            nargs="?",
            default=None,
            help="The model: a name, sig, label or path, optionally @best/@last/@final/@<epoch> (same as --xp-path)",
        )
        parser.add_argument(
            "--model-id",
            default=os.getenv("MACCHINA_RICORDI_MODEL_ID"),
            help="HuggingFace model ID or local exported MusicGen model path",
        )
        parser.add_argument(
            "--recipe",
            "--profile",
            dest="profile",
            choices=recipe_choices(),
            default=os.getenv("MACCHINA_RICORDI_PROFILE"),
            help="Recipe to record in metadata; read from the model when omitted",
        )
        parser.add_argument(
            "--device",
            default=os.getenv("MACCHINA_RICORDI_DEVICE"),
            help="Explicit torch device override, e.g. cpu or cuda",
        )
        parser.add_argument(
            "--num-samples",
            "--batch-size",
            dest="num_samples",
            type=int,
            default=int(os.getenv("MACCHINA_RICORDI_BATCH_SIZE", "1")),
            help="Number of samples to generate",
        )
        parser.add_argument(
            "--duration-seconds",
            type=float,
            default=float(os.getenv("MACCHINA_RICORDI_DURATION_SECONDS", "10.0")),
            help="Target generated sample length in seconds",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=int(os.getenv("MACCHINA_RICORDI_SEED")) if os.getenv("MACCHINA_RICORDI_SEED") else None,
            help="Random seed for generation",
        )
        parser.add_argument(
            "--temperature",
            type=float,
            default=float(os.getenv("MACCHINA_RICORDI_TEMPERATURE", "1.0")),
            help="Sampling temperature",
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=int(os.getenv("MACCHINA_RICORDI_TOP_K", "250")),
            help="Top-k sampling value",
        )
        parser.add_argument(
            "--top-p",
            type=float,
            default=float(os.getenv("MACCHINA_RICORDI_TOP_P", "0.0")),
            help="Top-p sampling value; 0 uses top-k",
        )
        parser.add_argument(
            "--min-p",
            type=float,
            default=float(os.getenv("MACCHINA_RICORDI_MIN_P", "0.0")),
            help=(
                "min-p sampling threshold (replaces top-k/top-p when > 0): keep "
                "tokens with p >= min_p * p_max after temperature. Meant for high "
                "temperatures; 0.05-0.1 useful (default: 0 = off)"
            ),
        )
        parser.add_argument(
            "--cfg-coef",
            type=float,
            default=float(os.getenv("MACCHINA_RICORDI_CFG_COEF", "3.0")),
            help="Classifier-free guidance coefficient",
        )
        parser.add_argument(
            "--use-sampling",
            action=argparse.BooleanOptionalAction,
            default=bool_env("MACCHINA_RICORDI_USE_SAMPLING", True),
            help="Use sampling instead of argmax decoding",
        )
        parser.add_argument(
            "--extend-stride",
            type=float,
            default=float(os.getenv("MACCHINA_RICORDI_EXTEND_STRIDE", str(DEFAULT_EXTEND_STRIDE_SECONDS))),
            help="Stride used when generating beyond the model segment duration",
        )
        parser.add_argument(
            "--xp-path",
            default=None,
            help="Dora XP folder to sample from",
        )
        parser.add_argument(
            "--checkpoint",
            default="latest",
            help="Checkpoint selector for --xp-path: latest, final, or an epoch number",
        )
        parser.add_argument(
            "--checkpoint-path",
            default=None,
            help="Explicit checkpoint.th or checkpoint_N.th path; overrides --xp-path/--checkpoint",
        )
        parser.add_argument(
            "--save-output",
            action=argparse.BooleanOptionalAction,
            default=bool_env("MACCHINA_RICORDI_SAVE_OUTPUT", True),
            help="Save generated WAVs and metadata to disk",
        )
        parser.add_argument(
            "--output-dir",
            default=os.getenv("MACCHINA_RICORDI_OUTPUT_DIR", "artifacts/sessions"),
            help="Where sessions go (default: artifacts/sessions, as <day>/generate_<run id>/)",
        )
        parser.add_argument(
            "--output-layout",
            choices=("session", "dated-run-dir", "flat"),
            default=os.getenv("MACCHINA_RICORDI_OUTPUT_LAYOUT", "session"),
            help="session (default): <day>/<kind>_<run id>/ with session.json, run.json and takes/; "
            "dated-run-dir and flat are the layouts before October 2026",
        )
        parser.add_argument(
            "--output-prefix",
            default=os.getenv("MACCHINA_RICORDI_OUTPUT_PREFIX", "musicgen"),
            help="A label for the session: generate_<run id>_<label> (and the file prefix in the old layouts)",
        )
        parser.add_argument(
            "--listen",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Write listening dossier JSON/Markdown sidecars for saved WAVs",
        )
        parser.set_defaults(func=cmd_play_generate)

    unconditional = play_subparsers.add_parser(
        "generate",
        help="Generate without text or audio prompting",
        description="Generate one or more unconditional MusicGen samples.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    add_generation_common(unconditional)
    unconditional.set_defaults(generation_mode="unconditional")

    continuation = play_subparsers.add_parser(
        "continue",
        help="Generate continuation from an audio prompt",
        description="Generate audio that continues from a source audio prompt.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    add_generation_common(continuation)
    continuation.set_defaults(generation_mode="continue")
    continuation.add_argument(
        "--prompt-audio",
        required=True,
        help="Audio file used as continuation prompt",
    )
    continuation.add_argument(
        "--prompt-start-seconds",
        type=float,
        default=0.0,
        help="Start time within --prompt-audio",
    )
    continuation.add_argument(
        "--prompt-duration-seconds",
        type=float,
        default=5.0,
        help="Length of audio prompt to use; must be shorter than --duration-seconds",
    )

    sweep = play_subparsers.add_parser(
        "sweep",
        help="Run a grid search over micro-unconditional generation settings",
        description="Run a deterministic grid of MusicGen generation settings and write a sweep index.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    sweep.add_argument(
        "--mode",
        dest="sweep_mode",
        choices=("continue", "unconditional"),
        default="continue",
        help="Sweep generation mode",
    )
    sweep.add_argument(
        "--model-id",
        default=os.getenv("MACCHINA_RICORDI_MODEL_ID"),
        help="HuggingFace model ID or local exported MusicGen model path",
    )
    sweep.add_argument(
        "--recipe",
        "--profile",
        dest="profile",
        choices=recipe_choices(),
        default=os.getenv("MACCHINA_RICORDI_PROFILE"),
        help="Recipe to record in metadata; read from the model when omitted",
    )
    sweep.add_argument(
        "--device",
        default=os.getenv("MACCHINA_RICORDI_DEVICE"),
        help="Explicit torch device override, e.g. cpu or cuda",
    )
    sweep.add_argument(
        "--xp-path",
        default=None,
        help="Dora XP folder to sample from",
    )
    sweep.add_argument(
        "--checkpoint",
        default="latest",
        help="Checkpoint selector for --xp-path: latest, final, or an epoch number",
    )
    sweep.add_argument(
        "--checkpoint-path",
        default=None,
        help="Explicit checkpoint.th or checkpoint_N.th path; overrides --xp-path/--checkpoint",
    )
    sweep.add_argument(
        "--prompt-audio",
        default=None,
        help="Audio file used as continuation prompt",
    )
    sweep.add_argument(
        "--prompt-start-seconds",
        type=float,
        default=0.0,
        help="Start time within --prompt-audio",
    )
    sweep.add_argument(
        "--prompt-duration-seconds",
        type=float,
        default=5.0,
        help="Default audio prompt duration for continuation sweeps",
    )
    sweep.add_argument(
        "--num-samples",
        "--batch-size",
        dest="num_samples",
        type=int,
        default=int(os.getenv("MACCHINA_RICORDI_BATCH_SIZE", "1")),
        help="Number of samples to generate per grid combination",
    )
    sweep.add_argument(
        "--duration-seconds",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_DURATION_SECONDS", "10.0")),
        help="Default target generated sample length in seconds",
    )
    sweep.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("MACCHINA_RICORDI_SEED")) if os.getenv("MACCHINA_RICORDI_SEED") else None,
        help="Default random seed for generation",
    )
    sweep.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_TEMPERATURE", "1.0")),
        help="Default sampling temperature",
    )
    sweep.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("MACCHINA_RICORDI_TOP_K", "250")),
        help="Default top-k sampling value",
    )
    sweep.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_TOP_P", "0.0")),
        help="Default top-p sampling value",
    )
    sweep.add_argument(
        "--cfg-coef",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_CFG_COEF", "3.0")),
        help="Default classifier-free guidance coefficient",
    )
    sweep.add_argument(
        "--use-sampling",
        action=argparse.BooleanOptionalAction,
        default=bool_env("MACCHINA_RICORDI_USE_SAMPLING", True),
        help="Default sampling mode",
    )
    sweep.add_argument(
        "--extend-stride",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_EXTEND_STRIDE", str(DEFAULT_EXTEND_STRIDE_SECONDS))),
        help="Default stride for generation beyond the model segment duration",
    )
    sweep.add_argument("--temperature-values", default=None, help="Comma-separated temperature values")
    sweep.add_argument("--top-k-values", default=None, help="Comma-separated top-k values")
    sweep.add_argument("--top-p-values", default=None, help="Comma-separated top-p values")
    sweep.add_argument("--cfg-coef-values", default=None, help="Comma-separated cfg coefficient values")
    sweep.add_argument("--extend-stride-values", default=None, help="Comma-separated extend stride values")
    sweep.add_argument("--prompt-duration-seconds-values", default=None, help="Comma-separated prompt duration values")
    sweep.add_argument("--duration-seconds-values", default=None, help="Comma-separated generated duration values")
    sweep.add_argument("--seed-values", default=None, help="Comma-separated seed values; use none for an unseeded run")
    sweep.add_argument("--use-sampling-values", default=None, help="Comma-separated boolean sampling values")
    sweep.add_argument(
        "--output-dir",
        default=os.getenv("MACCHINA_RICORDI_SWEEP_OUTPUT_DIR") or None,
        help="Parent folder for the sweep (default: the session home, artifacts/sessions/<day>/sweep_<run id>/)",
    )
    sweep.add_argument(
        "--output-prefix",
        default=os.getenv("MACCHINA_RICORDI_SWEEP_OUTPUT_PREFIX", "sweep"),
        help="A label for the sweep: sweep_<run id>_<label> (with --output-dir: <label>_<run id>)",
    )
    sweep.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the sweep plan without launching AudioCraft generation",
    )
    sweep.add_argument(
        "--max-combinations",
        type=int,
        default=64,
        help="Maximum allowed grid combinations",
    )
    sweep.add_argument(
        "--score",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score generated samples with the arrhythmic/percussive scorer",
    )
    sweep.add_argument(
        "--index-html",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write an HTML listening index",
    )
    sweep.add_argument(
        "--listen",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write listening dossier sidecars for successful sweep WAVs",
    )
    sweep.set_defaults(func=cmd_play_sweep)

    stream = play_subparsers.add_parser(
        "stream",
        help="Play the model: iterative continuation loop with a live-edited control file",
        description=(
            "Run an outer continuation loop forever. Round 0 is an unconditional generation "
            "(or an excerpt seed); each subsequent round continues from the previous round's "
            "tail. A JSON control file in the session dir is re-read at the start of every "
            "round so knobs (temperature, top_k, tau, ...) can be changed mid-session. "
            "Each completed WAV is enqueued to a sounddevice playback worker (one-chunk-ahead "
            "latency) when --play is on."
        ),
        epilog="""examples:
  play stream --xp-path .dora/xps/0444687c --no-play --max-rounds 2
  play stream --xp-path .dora/xps/0444687c --device cuda --completion-duration 10 --tau 5
  play stream --seed-mode excerpt --seed-audio seed.wav --xp-path .dora/xps/0444687c""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    stream.add_argument(
        "--model-id",
        default=os.getenv("MACCHINA_RICORDI_MODEL_ID"),
        help="HuggingFace model id or local export",
    )
    stream.add_argument(
        "--recipe",
        "--profile",
        dest="profile",
        choices=recipe_choices(),
        default=os.getenv("MACCHINA_RICORDI_PROFILE"),
        help="Recipe to record in metadata; read from the model when omitted",
    )
    stream.add_argument("--device", default=os.getenv("MACCHINA_RICORDI_DEVICE"), help="cuda / cpu (auto by default)")
    stream.add_argument("--xp-path", default=None, help="Dora XP folder to sample from")
    stream.add_argument("--checkpoint", default="latest", help="Checkpoint selector: latest, final, or an epoch number")
    stream.add_argument("--checkpoint-path", default=None, help="Direct path to a checkpoint .th file")
    stream.add_argument(
        "--seed-mode",
        choices=("uncond", "excerpt"),
        default="uncond",
        help="Round 0 source: uncond generation or a copied excerpt",
    )
    stream.add_argument(
        "--seed-audio",
        default=None,
        help="Path to seed audio when --seed-mode excerpt",
    )
    stream.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="Cap on the number of continuation rounds beyond the seed; 0 = infinite",
    )
    stream.add_argument(
        "--completion-duration",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_COMPLETION_DURATION_SECONDS", "10.0")),
        help="Fresh audio appended by each round in seconds (live-editable via control file)",
    )
    stream.add_argument(
        "--tau",
        type=float,
        default=5.0,
        help="Tail length used as continuation prompt (live-editable via control file)",
    )
    stream.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_TEMPERATURE", "1.0")),
        help="Initial sampling temperature (live-editable)",
    )
    stream.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("MACCHINA_RICORDI_TOP_K", "250")),
        help="Initial top-k (live-editable)",
    )
    stream.add_argument(
        "--top-p",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_TOP_P", "0.0")),
        help="Initial top-p (live-editable)",
    )
    stream.add_argument(
        "--min-p",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_MIN_P", "0.0")),
        help=(
            "min-p sampling threshold: keep tokens with p >= min_p * p_max after "
            "temperature (replaces top-k/top-p when > 0). Adapts breadth to the "
            "model's per-step confidence, so high temperatures stay coherent. "
            "Useful range 0.05-0.1; live-editable (default: 0 = off)"
        ),
    )
    stream.add_argument(
        "--cfg-coef",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_CFG_COEF", "3.0")),
        help="Initial classifier-free guidance coefficient (live-editable)",
    )
    stream.add_argument(
        "--extend-stride",
        type=float,
        default=float(os.getenv("MACCHINA_RICORDI_EXTEND_STRIDE", str(DEFAULT_EXTEND_STRIDE_SECONDS))),
        help="Initial extend stride (live-editable)",
    )
    stream.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("MACCHINA_RICORDI_SEED")) if os.getenv("MACCHINA_RICORDI_SEED") else None,
        help="Base RNG seed; each round derives a deterministic seed from this",
    )
    stream.add_argument(
        "--output-dir",
        default=None,
        help="Parent folder for the session, as <folder>/<run id>/ "
        "(default: artifacts/sessions/<day>/stream_<run id>/)",
    )
    stream.add_argument(
        "--play",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Play completed rounds via sounddevice; requires the `instrument` extra",
    )
    stream.add_argument(
        "--crossfade-seconds",
        type=float,
        default=0.02,
        help="Micro-crossfade window at round splices; 0 disables (default: 0.02 = 20ms)",
    )
    stream.add_argument(
        "--max-buffer-ahead-seconds",
        type=float,
        default=None,
        help=(
            "Throttle: when the playback buffer is at least this far ahead of the "
            "playback head, pause generation until it drains back to "
            "--resume-buffer-ahead-seconds. Default: unset (no throttle, generate "
            "back-to-back). Useful since the persistent stream worker can run far "
            "ahead of playback. Live-editable as max_buffer_ahead_seconds, and "
            "with -/+ in --tui: a change lands during the wait it shortens. "
            "'t' holds the throttle open (throttle_enabled) without losing the "
            "cap, to bank a buffer cushion across a model swap's reload stall."
        ),
    )
    stream.add_argument(
        "--resume-buffer-ahead-seconds",
        type=float,
        default=None,
        help=(
            "Throttle resume threshold (hysteresis). When throttling, the loop waits "
            "until buffer ahead drops to this value before resuming generation. "
            "Defaults to max(0, --max-buffer-ahead-seconds - 10). Ignored when "
            "--max-buffer-ahead-seconds is unset. Live-editable as "
            "resume_buffer_ahead_seconds; the -/+ keys keep its ratio to the cap."
        ),
    )
    stream.add_argument(
        "--buffer-floor-ratio",
        type=float,
        default=None,
        help=(
            "The resume threshold as a fraction of --max-buffer-ahead-seconds (0-1): 1.0 tops the "
            "buffer up round by round, 0 lets it drain dry. Overridden by --resume-buffer-ahead-seconds. "
            "{ / } in --tui move it in 0.1 steps."
        ),
    )
    stream.add_argument(
        "--round-delay-seconds",
        type=float,
        default=0.0,
        help=(
            "Rest this long before each round (0-3s), to set its knobs; cut short when the buffer "
            "would run dry. Live-editable as round_delay_seconds, and with w/W in --tui (0.1s steps)."
        ),
    )
    stream.add_argument(
        "--score-rounds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Score each round with the arrhythmic-percussive scorer; record in performance.jsonl",
    )
    stream.add_argument(
        "--candidates",
        type=int,
        default=1,
        help=(
            "Best-of-N: generate N candidate rounds per step in one batched call "
            "(the AR loop is launch-bound, so batching is near-free on GPU) and "
            "keep the lowest-silence/highest-arc one. Live-editable via the "
            "control file. Max 8 (default: 1 = off)"
        ),
    )
    stream.add_argument(
        "--novelty",
        type=float,
        default=0.0,
        help=(
            "Anti-déjà-vu weight (0-1) for best-of-N selection: blend arc with "
            "distance from the last 8 rounds' MFCC fingerprints, steering the "
            "stream away from what it just played. Needs --candidates > 1. "
            "Try 0.3-0.5; live-editable as novelty_weight (default: 0 = off)"
        ),
    )
    stream.add_argument(
        "--stuck-weight",
        type=float,
        default=0.0,
        help=(
            "Drone guard (0-1) for best-of-N selection: rank candidates whose tail "
            "barely moves (spectral stationarity) lower, which also catches drones "
            "before they set in. Needs --candidates > 1; live-editable as "
            "stuck_weight (d/D in the TUI) (default: 0 = off)"
        ),
    )
    stream.add_argument(
        "--stuck-kick",
        type=int,
        default=0,
        help=(
            "Drone escape: after this many rounds in a row ending on a drone "
            "(drone_score >= 0.4), run one round at temperature +0.4 and top_k x2, "
            "then return to your settings. Works at --candidates 1; live-editable as "
            "stuck_kick_rounds (g/G in the TUI). Try 2 (default: 0 = off)"
        ),
    )
    stream.add_argument(
        "--motif-dir",
        default=None,
        help=(
            "Motif bank for recapitulation and the one-shot `pin` control "
            "command (default: <session>/motifs). A shared dir gives separate "
            "sessions a common identity"
        ),
    )
    stream.add_argument(
        "--motif-period",
        type=int,
        default=0,
        help=(
            "Every Nth round, blend a pinned motif's tail into the continuation "
            "prompt so themes return. Live-editable (default: 0 = off)"
        ),
    )
    stream.add_argument(
        "--motif-mix",
        type=float,
        default=0.5,
        help="Motif share in the recap prompt blend, 0-1 (default: 0.5; live-editable)",
    )
    stream.add_argument(
        "--no-tail-rescue",
        action="store_true",
        help=(
            "Disable tail rescue. By default, when the tail that would become "
            "the next prompt is mostly silence, the stream re-prompts from the "
            "latest earlier window in that round that still has audio — a jump "
            "cut, but it stops silence from feeding itself. Live-editable as "
            "tail_rescue"
        ),
    )
    stream.add_argument(
        "--novelty-calibration",
        default=None,
        help=(
            "JSON from 'score sim-calibrate' whose mfcc means/stds whiten the "
            "fingerprint before novelty is measured. Without usable calibration, "
            "novelty uses unwhitened cosine distance and tends to stay near zero. "
            "Pass this path explicitly to use "
            "your own file; the legacy default calibration is not shipped. The "
            "corpus behind the file decides what 'novel' is measured against"
        ),
    )
    stream.add_argument(
        "--no-novelty-calibration",
        action="store_true",
        help="Measure novelty with raw cosine, ignoring any calibration file",
    )
    stream.add_argument(
        "--prompt-tilt-low",
        type=float,
        default=0.0,
        help=(
            "Prompt-tilt steering, low band (dB, -12 to +12; crossover 250 Hz). "
            "EQs the tail *before* it is re-prompted, so the model hears a "
            "bass-heavy recent past and continues from it. Compounds while "
            "held; live-editable and bound to j/J in --tui (default: 0 = flat)"
        ),
    )
    stream.add_argument(
        "--prompt-tilt-mid",
        type=float,
        default=0.0,
        help="Prompt-tilt mid band (dB, 250 Hz - 4 kHz); k/K in --tui (default: 0)",
    )
    stream.add_argument(
        "--prompt-tilt-high",
        type=float,
        default=0.0,
        help="Prompt-tilt high band (dB, above 4 kHz); l/L in --tui (default: 0)",
    )
    stream.add_argument(
        "--prompt-rate",
        type=float,
        default=1.0,
        help=(
            "Varispeed of the continuation prompt only (0.5-2.0): 'rate x tau' "
            "seconds of tail are resampled down to exactly tau, so token count "
            "and buffer arithmetic stay put. Only what is fed back moves, not "
            "what is played, so the stream migrates in pitch/tempo over a few "
            "rounds and stays where it drifted when set back to 1.0. Beyond "
            "about +/-15%% EnCodec adds its own artefacts. Live-editable, "
            "bound to < / > in --tui (default: 1.0 = off)"
        ),
    )
    stream.add_argument(
        "--prompt-rate-mode",
        choices=("tape", "tempo"),
        default="tape",
        help=(
            "How --prompt-rate shifts the prompt: 'tape' resamples (pitch and "
            "tempo together, transients intact), 'tempo' phase-vocodes (pitch "
            "preserved, transients smeared — very audible on percussive "
            "material, which may be the point). Live-editable (default: tape)"
        ),
    )
    stream.add_argument(
        "--playlist",
        default=None,
        help=(
            "Comma-separated Dora XP folders bound to the digit keys 1-9 in "
            "--tui, with tab to cycle: a scene change becomes one keystroke "
            "instead of a mid-performance edit to model_xp. The first entry is "
            "the starting model when --xp-path is omitted. Only the first 9 get "
            "a key; missing folders are reported at startup and skipped by the "
            "loop's usual model_swap_skipped path"
        ),
    )
    stream.add_argument(
        "--follow",
        choices=("generated", "playing"),
        default="generated",
        help="For --tui: which round the metrics block describes (see 'play monitor --follow')",
    )
    stream.add_argument(
        "--blind",
        action="store_true",
        help=(
            "For --tui: start with every per-round reading hidden. They are still "
            "computed and logged; you just play by ear. Toggle live with 'b'. The "
            "readings were visible for the whole statistics corpus, which is what "
            "makes its novelty finding circular — this is how that gets tested."
        ),
    )
    stream.add_argument(
        "--tui",
        action="store_true",
        help=(
            "Draw a single-pane dashboard in this terminal and take keyboard "
            "input, instead of scrolling status lines. Keys nudge temperature "
            "(up/down), tau (left/right), completion duration ([ ]), top_k (, .), "
            "novelty_weight (n/N), candidates (c/C) and the buffer cap (- +); "
            "t holds the throttle open to bank a cushion before a swap; "
            "j/k/l tilt the prompt's low/mid/high bands and < / > varispeed it "
            "(0 resets both); "
            "1-9 swap models from --playlist; r reroll, p pin, Q stop, ? for the full list. "
            "Edits go through control.json, so an external editor still works. "
            "Worker diagnostics are redirected to <session>/worker.log so they "
            "cannot scribble on the frame. Falls back to plain output when "
            "stdin/stdout is not an interactive terminal"
        ),
    )
    stream.set_defaults(func=cmd_play_stream)

    monitor = play_subparsers.add_parser(
        "monitor",
        help="Watch a stream session's last generated round: scores, candidates, buffer",
        description=(
            "Read-only dashboard over a session's performance.jsonl and control.json. "
            "Shows the last round the generator produced — not the audio currently "
            "playing, which with a throttle buffer can be a minute behind — plus its "
            "scores, the per-candidate table best-of-N chose from, sparklines over "
            "recent rounds, and the live control values. Attaches to a running "
            "session without touching it, so it is safe mid-performance; also works "
            "on finished sessions. Defaults to the newest session."
        ),
        epilog="""examples:
  play monitor
  play monitor --session 20260729T075418Z
  play monitor --once --history 200""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    monitor.add_argument(
        "--session",
        default=None,
        help="Stream session to watch: its folder or id (default: the newest stream session)",
    )
    monitor.add_argument(
        "--streams-dir",
        default=None,
        help="Watch the newest session in this folder instead (default: every stream session, old and new homes)",
    )
    monitor.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between redraws (default: 2.0)",
    )
    monitor.add_argument(
        "--history",
        type=int,
        default=60,
        help="Rounds of history behind the sparklines (default: 60)",
    )
    monitor.add_argument(
        "--follow",
        choices=("generated", "playing"),
        default="generated",
        help=(
            "Which round the metrics block describes. 'generated' (default) is the "
            "round the generator just finished — correct for steering. 'playing' is "
            "the round you are hearing right now, which with a buffer lags several "
            "rounds behind; wrong for steering, right for correlating a number with "
            "what is actually in your ears. The frame always says which is active"
        ),
    )
    monitor.add_argument(
        "--events",
        type=int,
        default=8,
        help=(
            "Height of the events box in lines; 0 hides it. The box is always this "
            "tall, so the frame never changes size as events arrive (default: 8)"
        ),
    )
    monitor.add_argument(
        "--width",
        type=int,
        default=None,
        help="Frame width in columns (default: terminal width, clamped to 64-110)",
    )
    monitor.add_argument(
        "--once",
        action="store_true",
        help="Print a single frame and exit instead of redrawing",
    )
    monitor.set_defaults(func=cmd_play_monitor)

    duet = play_subparsers.add_parser(
        "duet",
        help="Render two models playing together, each continuing a blend of itself + its partner",
        description=(
            "Offline two-voice renderer. Every round, each voice's continuation prompt is "
            "(1-listen)*own_tail + listen*partner_tail, so the models react to each other "
            "(cross-codec is free: each voice encodes the shared audio with its own EnCodec). "
            "Outputs two mono stems, a panned stereo mix.wav, rounds.jsonl and manifest.json. "
            "Intended pairing: the 50 Hz deep (arc voice, A) with the 25 Hz LM (percussive voice, B)."
        ),
        epilog="""examples:
  play duet --xp-a .dora/xps/2b0dc2a8 --xp-b .dora/xps/ed386554 --rounds 20
  play duet --xp-a .dora/xps/2b0dc2a8 --xp-b .dora/xps/ed386554 \\
      --listen-a 0.3 --listen-b 0.5 --temperature-b 1.4 --candidates 4""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    duet.add_argument(
        "--output-dir", default=None, help="Session folder (default: artifacts/sessions/<day>/duet_<run id>/)"
    )
    duet.add_argument("--rounds", type=int, default=8, help="Rounds to render (default: 8)")
    duet.add_argument("--round-duration", type=float, default=9.0, help="Seconds per round (default: 9)")
    duet.add_argument("--tau", type=float, default=2.0, help="Tail length each voice re-hears (default: 2)")
    duet.add_argument(
        "--candidates",
        type=int,
        default=1,
        help="Best-of-N per voice per round, picked by least silence (default: 1)",
    )
    duet.add_argument("--seed", type=int, default=None, help="Base RNG seed")
    duet.add_argument("--device", default=os.getenv("MACCHINA_RICORDI_DEVICE"), help="cuda / cpu (auto by default)")
    for voice, label in (("a", "voice A (left)"), ("b", "voice B (right)")):
        duet.add_argument(f"--xp-{voice}", required=True, help=f"Dora XP folder for {label}")
        duet.add_argument(f"--checkpoint-{voice}", default="latest", help=f"Checkpoint selector for {label}")
        duet.add_argument(f"--profile-{voice}", default=None, help=f"Profile for {label}; auto-detected")
        duet.add_argument(f"--temperature-{voice}", type=float, default=1.2)
        duet.add_argument(f"--top-k-{voice}", type=int, default=250)
        duet.add_argument(f"--top-p-{voice}", type=float, default=0.0)
        duet.add_argument(f"--min-p-{voice}", type=float, default=0.0)
        duet.add_argument(
            f"--listen-{voice}",
            type=float,
            default=0.35,
            help=f"How much {label} hears its partner: 0 = deaf, 1 = only the partner (default: 0.35)",
        )
        duet.add_argument(f"--seed-audio-{voice}", default=None, help=f"Optional round-0 excerpt for {label}")
    duet.add_argument("--pan-a", type=float, default=-0.35, help="Voice A stereo position (default: -0.35)")
    duet.add_argument("--pan-b", type=float, default=0.35, help="Voice B stereo position (default: 0.35)")
    duet.add_argument("--gain-a", type=float, default=1.0)
    duet.add_argument("--gain-b", type=float, default=0.8)
    duet.set_defaults(func=cmd_play_duet)

    variations = play_subparsers.add_parser(
        "variations",
        help="Re-perform a piece: freeze coarse RVQ book(s), resample the rest",
        description=(
            "Encode a WAV with the model's own EnCodec, freeze the coarsest RVQ book(s) "
            "(book 0 carries the structural skeleton), and resample the remaining books "
            "at your chosen temperature — same piece, different performance. All N "
            "variations render in one batch. Also writes the codec round-trip of the "
            "reference for a fair listening comparison."
        ),
        epilog="""examples:
  play variations --input round_042.wav --xp-path .dora/xps/2b0dc2a8 -n 4
  play variations --input seed.wav --xp-path .dora/xps/2b0dc2a8 \\
      --freeze-books 2 --temperature 1.5 --keep-intro-seconds 1.0""",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    variations.add_argument(
        "--input", required=True, help="What to re-perform: an audio file or a take (stream_20261005T141908Z#12)"
    )
    variations.add_argument(
        "--output-dir", default=None, help="Session folder (default: artifacts/sessions/<day>/variations_<run id>/)"
    )
    variations.add_argument(
        "--recipe",
        "--profile",
        dest="profile",
        choices=recipe_choices(),
        default=os.getenv("MACCHINA_RICORDI_PROFILE"),
        help="Recipe to record in metadata; read from the model when omitted",
    )
    variations.add_argument("--xp-path", default=None, help="Dora XP folder for the model")
    variations.add_argument("--checkpoint", default="latest", help="Checkpoint selector")
    variations.add_argument("--device", default=os.getenv("MACCHINA_RICORDI_DEVICE"))
    variations.add_argument("-n", "--n", type=int, default=4, help="Variations per batch (default: 4)")
    variations.add_argument(
        "--freeze-books",
        type=int,
        default=1,
        help="Coarse RVQ books to freeze; 1 = structure only (default: 1)",
    )
    variations.add_argument(
        "--keep-intro-seconds",
        type=float,
        default=0.0,
        help="Freeze all books for the first N seconds so variations open identically",
    )
    variations.add_argument("--seconds", type=float, default=None, help="Cap the reference length")
    variations.add_argument("--temperature", type=float, default=1.3)
    variations.add_argument("--top-k", type=int, default=250)
    variations.add_argument("--top-p", type=float, default=0.0)
    variations.add_argument("--min-p", type=float, default=0.0)
    variations.add_argument("--seed", type=int, default=None)
    variations.set_defaults(func=cmd_play_variations)

    replay = play_subparsers.add_parser(
        "replay",
        help="Re-run a saved generation and verify byte-identity against the original audio",
        description="Replay a saved generation metadata file and compare output hashes with the original run.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    replay.add_argument(
        "metadata_path",
        metavar="source",
        help="What to replay: a generate or continue session (its id or folder), or a run.json",
    )
    replay.add_argument(
        "--output-dir",
        default="artifacts/replays",
        help="Root directory for replay outputs (default: artifacts/replays)",
    )
    replay.add_argument(
        "--report-path",
        default=None,
        help="Override the replay_report.json path (default: <replay-dir>/replay_report.json)",
    )
    replay.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any sample hashes mismatch",
    )
    replay.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON summary instead of the plain-text table",
    )
    replay.set_defaults(func=cmd_play_replay)
