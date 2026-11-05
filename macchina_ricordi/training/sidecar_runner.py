from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ..cli.common import MacchinaRicordiHelpFormatter
from ..musicgen.config import bootstrap_audiocraft_dora_dir
from ..musicgen.xp_datasource import restore_xp_dataset_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU-only AudioCraft generate/evaluate stages against an immutable epoch checkpoint.",
        formatter_class=MacchinaRicordiHelpFormatter,
    )
    parser.add_argument("--sig", required=True, help="Dora XP signature to load config from")
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help="Absolute path to the immutable checkpoint_<epoch>.th file to run on CPU",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("generate", "evaluate"),
        help="Single stage to execute on CPU",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Torch/OMP thread count for CPU-side stage execution",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override for the sidecar stage",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)

    import audiocraft
    import flashy
    import torch
    from audiocraft import train
    from audiocraft.utils import checkpoint as checkpoint_utils
    from audiocraft.utils.utils import with_rank_rng

    torch.set_num_threads(args.num_threads)
    logging.basicConfig(level=logging.INFO)
    flashy.distrib.init()

    override_cfg: dict[str, object] = {
        "device": "cpu",
        "autocast": False,
        "num_threads": args.num_threads,
        "execute_only": args.stage,
        "execute_inplace": True,
        "logging": {
            "log_wandb": False,
            "log_tensorboard": False,
        },
        "dataset": {
            "num_workers": 0,
        },
    }
    if args.batch_size is not None:
        override_cfg["dataset"]["batch_size"] = args.batch_size  # type: ignore[index]

    bootstrap_audiocraft_dora_dir()
    restore_xp_dataset_config(
        Path(os.environ["AUDIOCRAFT_DORA_DIR"]) / "xps" / args.sig,
        audiocraft_dir=Path(audiocraft.__file__).resolve().parent.parent,
        purpose=f"sidecar-{args.stage}",
    )
    solver = train.get_solver_from_sig(
        args.sig,
        override_cfg=override_cfg,
        restore=False,
        load_best=False,
        disable_fsdp=True,
    )
    state = checkpoint_utils.load_checkpoint(checkpoint_path)
    stateful_sources = getattr(getattr(solver, "stateful", None), "sources", {})
    if isinstance(stateful_sources, dict) and "scaler" in state and "scaler" not in stateful_sources:
        state = dict(state)
        state.pop("scaler", None)
    solver.load_state_dict(state)
    solver.model.eval()

    stage_to_fn = {
        "generate": with_rank_rng()(solver.generate),
        "evaluate": solver.evaluate,
    }

    print(f"xp_sig: {args.sig}")
    print(f"checkpoint_path: {checkpoint_path}")
    print(f"stage: {args.stage}")
    print("device: cpu")
    print(f"num_threads: {args.num_threads}")
    if args.batch_size is not None:
        print(f"batch_size: {args.batch_size}")

    solver._start_epoch()
    with torch.no_grad():
        solver.run_stage(args.stage, stage_to_fn[args.stage])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
