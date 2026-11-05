from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from run_musicgen_generate import _load_model, _read_prompt_audio, _set_seed, _to_numpy_audio_batch, _write_wav

from macchina_ricordi.instrument.sweep import SWEEP_INDEX_FIELDNAMES, SWEEP_SCHEMA_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MusicGen sweep inside the AudioCraft runtime")
    parser.add_argument("--plan-json", required=True)
    return parser.parse_args()


def _base_row(plan: dict[str, object], combo: dict[str, object], source: dict[str, object]) -> dict[str, object]:
    return {
        "sweep_id": plan["sweep_id"],
        "combo_id": combo["combo_id"],
        "combo_index": combo["combo_index"],
        "sample_index": "",
        "path": "",
        "error": "",
        "mode": plan["mode"],
        "model_id": plan["model_id"],
        "profile": plan.get("profile") or "",
        "xp_path": source.get("xp_path") or "",
        "checkpoint_path": source.get("checkpoint_path") or "",
        "prompt_audio_path": plan.get("prompt_audio_path") or "",
        "prompt_start_seconds": plan.get("prompt_start_seconds", 0.0),
        "prompt_duration_seconds": combo.get("prompt_duration_seconds") or "",
        "duration_seconds": combo["duration_seconds"],
        "num_samples": plan["num_samples"],
        "seed": combo.get("seed") if combo.get("seed") is not None else "",
        "temperature": combo["temperature"],
        "top_k": combo["top_k"],
        "top_p": combo["top_p"],
        "cfg_coef": combo["cfg_coef"],
        "extend_stride": combo["extend_stride"],
        "use_sampling": combo["use_sampling"],
    }


def _metadata_payload(
    *,
    plan: dict[str, object],
    combo: dict[str, object],
    source: dict[str, object],
    device: str,
    sample_rate: int,
    audio_paths: list[Path],
    error: str = "",
) -> dict[str, object]:
    return {
        "run_id": f"{plan['sweep_id']}_{combo['combo_id']}",
        "backend": "audiocraft",
        "model_id": plan["model_id"],
        "device": device,
        "sample_rate": sample_rate,
        "request": {
            "prompts": [""],
            "batch_size": int(plan["num_samples"]),
            "num_samples": int(plan["num_samples"]),
            "mode": plan["mode"],
            "duration_seconds": combo["duration_seconds"],
            "max_new_tokens": 256,
            "seed": combo.get("seed"),
            "prompt_audio_path": plan.get("prompt_audio_path"),
            "prompt_start_seconds": plan.get("prompt_start_seconds", 0.0),
            "prompt_duration_seconds": combo.get("prompt_duration_seconds"),
            "use_sampling": combo["use_sampling"],
            "top_k": combo["top_k"],
            "top_p": combo["top_p"],
            "temperature": combo["temperature"],
            "cfg_coef": combo["cfg_coef"],
            "extend_stride": combo["extend_stride"],
        },
        "generation": {
            "mode": plan["mode"],
            "source": source,
            "sweep": {
                "sweep_id": plan["sweep_id"],
                "combo_id": combo["combo_id"],
                "combo_index": combo["combo_index"],
                "parameters": {
                    key: combo[key]
                    for key in (
                        "duration_seconds",
                        "prompt_duration_seconds",
                        "temperature",
                        "top_k",
                        "top_p",
                        "cfg_coef",
                        "extend_stride",
                        "seed",
                        "use_sampling",
                    )
                },
            },
        },
        "warnings": [],
        "files": [str(path) for path in audio_paths],
        "error": error,
    }


def _generate_combo(
    *,
    model,
    plan: dict[str, object],
    combo: dict[str, object],
    output_dir: Path,
    device: str,
) -> list[Path]:
    if float(combo["extend_stride"]) >= model.max_duration:
        raise ValueError(
            f"extend_stride must be shorter than this model's max duration "
            f"({model.max_duration:g}s); got {float(combo['extend_stride']):g}s."
        )
    _set_seed(combo.get("seed"), device)
    model.set_generation_params(
        use_sampling=bool(combo["use_sampling"]),
        top_k=int(combo["top_k"]),
        top_p=float(combo["top_p"]),
        temperature=float(combo["temperature"]),
        duration=float(combo["duration_seconds"]),
        cfg_coef=float(combo["cfg_coef"]),
        extend_stride=float(combo["extend_stride"]),
    )
    num_samples = int(plan["num_samples"])
    if plan["mode"] == "unconditional":
        audio_values = model.generate_unconditional(num_samples)
    else:
        prompt_args = SimpleNamespace(
            prompt_audio=plan.get("prompt_audio_path"),
            prompt_start_seconds=float(plan.get("prompt_start_seconds", 0.0)),
            prompt_duration_seconds=combo.get("prompt_duration_seconds"),
            duration_seconds=float(combo["duration_seconds"]),
        )
        prompt_audio, prompt_sample_rate = _read_prompt_audio(prompt_args)
        if prompt_audio.dim() == 2:
            prompt_audio = prompt_audio[None]
        prompt_audio = prompt_audio.repeat(num_samples, 1, 1)
        audio_values = model.generate_continuation(prompt_audio, prompt_sample_rate, descriptions=[None] * num_samples)

    batch_audio = _to_numpy_audio_batch(audio_values)
    audio_paths: list[Path] = []
    for index, sample_ch_t in enumerate(batch_audio):
        audio_path = output_dir / f"sample_{index:03d}.wav"
        _write_wav(audio_path, sample_ch_t, int(model.sample_rate))
        audio_paths.append(audio_path)
    return audio_paths


def main() -> int:
    args = _parse_args()
    plan_path = Path(args.plan_json).expanduser().resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    sweep_dir = plan_path.parent
    device = plan.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_args = SimpleNamespace(
        model_id=plan["model_id"],
        profile=plan.get("profile"),
        xp_path=(plan.get("source") or {}).get("xp_path"),
        checkpoint_path=(plan.get("source") or {}).get("checkpoint_path"),
    )
    model, loaded_source = _load_model(model_args, str(device))
    source = dict(plan.get("source") or {})
    source.update(loaded_source)

    rows: list[dict[str, object]] = []
    combo_failures = 0
    samples_written = 0
    for combo in plan["combinations"]:
        combo_dir = sweep_dir / str(combo["combo_id"])
        combo_dir.mkdir(parents=True, exist_ok=True)
        try:
            audio_paths = _generate_combo(model=model, plan=plan, combo=combo, output_dir=combo_dir, device=str(device))
            samples_written += len(audio_paths)
            metadata = _metadata_payload(
                plan=plan,
                combo=combo,
                source=source,
                device=str(device),
                sample_rate=int(model.sample_rate),
                audio_paths=audio_paths,
            )
            for sample_index, audio_path in enumerate(audio_paths):
                row = _base_row(plan, combo, source)
                row["sample_index"] = sample_index
                row["path"] = str(audio_path)
                rows.append(row)
        except Exception as exc:  # pylint: disable=broad-except
            combo_failures += 1
            metadata = _metadata_payload(
                plan=plan,
                combo=combo,
                source=source,
                device=str(device),
                sample_rate=int(getattr(model, "sample_rate", 0)),
                audio_paths=[],
                error=str(exc),
            )
            row = _base_row(plan, combo, source)
            row["error"] = str(exc)
            rows.append(row)
        (combo_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    index_csv_path = sweep_dir / "index.csv"
    with index_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_INDEX_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "schema_version": SWEEP_SCHEMA_VERSION,
        "sweep_id": plan["sweep_id"],
        "status": "completed" if combo_failures == 0 else "completed_with_errors",
        "mode": plan["mode"],
        "combo_count": len(plan["combinations"]),
        "combo_failures": combo_failures,
        "samples_written": samples_written,
        "plan_path": str(plan_path),
        "report_path": str(sweep_dir / "sweep_report.json"),
        "index_csv_path": str(index_csv_path),
        "index_html_path": str(sweep_dir / "index.html"),
        "score_enabled": False,
    }
    (sweep_dir / "sweep_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
