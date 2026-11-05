"""Build a token pool: training crops encoded once, then served to training as cached batches.

Draws ``--crops`` crops with a run's own sampler and settings (copied from an XP folder, or composed from the
Dora overrides the run will use) and encodes them with its frozen codec the
way MusicGenSolver does (fp32, outside autocast, the recipe's batch size at a time). It writes what AudioCraft's
``TokenPoolLoader`` reads: ``tokens.npy`` (int16 codes), ``infos.pt`` (each crop's info) and ``order.npy`` (the
crops of every update, epoch after epoch). Batches walk the pool in shuffled passes, so every crop comes back the
same number of times. (Pools before 2026-10-25 stored every epoch's batches as zips under ``cache/``.)

``train run --pool`` runs this and trains with ``cache.path=<out>`` (``macchina_ricordi/training/pool.py``). Only
the train stage reads the cache; validation still encodes its audio. Needs
patches/audiocraft-cached-batch-loader.patch.

Run inside the musicgen venv:

    .venv-musicgen/bin/python scripts/build_token_pool.py .dora/xps/<sig> artifacts/token_pools/<name> \
        --crops 25000 --epochs 50
    .venv-musicgen/bin/python scripts/build_token_pool.py dora artifacts/token_pools/<name> \
        --crops 25000 --epochs 50 -- <the run's dora overrides>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".audiocraft"))


def shuffled_passes(pool_size: int, draws: int, seed: int) -> torch.Tensor:
    """Pool indices for `draws` draws: back-to-back random permutations of the pool, cut to length."""
    gen = torch.Generator().manual_seed(seed)
    passes = -(-draws // pool_size)
    return torch.cat([torch.randperm(pool_size, generator=gen) for _ in range(passes)])[:draws]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="XP folder whose settings to copy, or `dora` to compose them from DORA_ARGV")
    parser.add_argument("out", help="Output folder; refused if it exists")
    parser.add_argument("--crops", type=int, required=True, help="Pool size")
    parser.add_argument("--epochs", type=int, required=True, help="Epochs of batches to write")
    parser.add_argument("--workers", type=int, default=6, help="Audio loading workers")
    parser.add_argument("--crop-seed", type=int, default=0, help="Draw another set of crops (e.g. for a continuation)")
    parser.add_argument("--fingerprint", default=None, help="Dataset fingerprint to record (default: the XP's)")
    raw = sys.argv[1:]
    cut = raw.index("--") if "--" in raw else len(raw)
    args, dora_argv = parser.parse_args(raw[:cut]), raw[cut + 1 :]  # after --: the run's Dora overrides

    from audiocraft.solvers import builders
    from audiocraft.solvers.compression import CompressionSolver
    from audiocraft.utils.cache import TokenPoolLoader
    from audiocraft.utils.utils import get_dataset_from_loader
    from omegaconf import OmegaConf

    out = Path(args.out).resolve()
    if out.exists():
        parser.error(f"{out} exists: a pool is built once, pick a new name")
    work = out.with_name(out.name + ".partial")
    if work.exists():  # an interrupted build of this same pool
        shutil.rmtree(work)
    if args.source == "dora":
        from audiocraft.train import main as dora_main

        xp = None
        cfg = OmegaConf.create(OmegaConf.to_container(dora_main.get_xp(dora_argv).cfg, resolve=True))
    else:
        xp = Path(args.source).resolve()
        cfg = OmegaConf.load(xp / ".hydra" / "config.yaml")
    batch, updates = int(cfg.dataset.batch_size), int(cfg.optim.updates_per_epoch)
    draws = args.epochs * updates * batch

    cfg.execute_only = "train"  # build the train split only
    dataset = get_dataset_from_loader(builders.get_audio_datasets(cfg, builders.DatasetType.MUSIC)["train"])
    # training epochs count from 1 (and stay far below 10,000), so a run never draws these crops itself
    dataset.current_epoch = 10_000 * args.crop_seed
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch, sampler=range(args.crops), num_workers=args.workers, collate_fn=dataset.collater
    )
    codec = CompressionSolver.wrapped_model_from_checkpoint(cfg, cfg.compression_model_checkpoint, device="cuda")
    codec.eval()

    t0 = time.time()
    pool = []
    with torch.no_grad():
        for step, (wav, infos) in enumerate(loader):
            codes, scale = codec.encode(wav.cuda())
            assert scale is None
            for info, one in zip(infos, codes.cpu(), strict=True):
                assert one.max() < 2**15
                info.self_wav = None
                info.audio_tokens = one.short()
                pool.append(info)
            if step % 500 == 0:
                print(f"encoded {len(pool):,}/{args.crops:,} crops in {time.time() - t0:.0f} s", flush=True)
    encode_s = time.time() - t0

    t1 = time.time()
    order = shuffled_passes(len(pool), draws, seed=int(cfg.seed))
    work.mkdir(parents=True)
    np.save(work / "tokens.npy", torch.stack([info.audio_tokens for info in pool]).numpy())
    np.save(work / "order.npy", order.numpy().astype(np.int32))
    tokens = [info.audio_tokens for info in pool]
    for info in pool:
        info.audio_tokens = None  # the loader takes them from tokens.npy
    torch.save(pool, work / "infos.pt")
    write_s = time.time() - t1

    # self-check: the loader returns exactly the planned batches, and repeats are even
    check = TokenPoolLoader(work, batch, min_length=updates)
    for epoch in (1, args.epochs):
        check.start_epoch(epoch)
        (got,) = next(iter(check))
        start = (epoch - 1) * updates * batch
        want = order[start : start + batch].tolist()
        assert all(
            torch.equal(a.audio_tokens, tokens[i]) and a.seek_time == pool[i].seek_time
            for a, i in zip(got, want, strict=True)
        )
    counts = torch.bincount(order, minlength=len(pool))
    assert int(counts.max() - counts.min()) <= 1
    metadata = xp / "macchina_ricordi_train_metadata.json" if xp else None
    fingerprint = args.fingerprint
    if fingerprint is None and metadata is not None and metadata.exists():
        fingerprint = json.loads(metadata.read_text())["dataset_fingerprint_digest"]
    status = {
        "schema": "macchina_ricordi.token_pool.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_xp": str(xp) if xp else None,
        "dataset": str(cfg.datasource.train),
        "dataset_fingerprint": fingerprint,
        "codec": str(cfg.compression_model_checkpoint),
        "segment_duration": float(cfg.dataset.segment_duration),
        "crops": len(pool),
        "epochs": args.epochs,
        "batch_size": batch,
        "updates_per_epoch": updates,
        "draws": draws,
        "repeats_per_crop": draws / len(pool),
        "crop_seed": args.crop_seed,
        "crop_seed_epoch": dataset.current_epoch,
        "order_seed": int(cfg.seed),
        "encode": f"fp32, {batch} crops at a time",
        "encode_seconds": round(encode_s, 1),
        "write_seconds": round(write_s, 1),
        "layout": "tokens.npy + infos.pt + order.npy",
        "pool_bytes": sum(p.stat().st_size for p in work.iterdir()),
        "train_override": f"cache.path={out}",
    }
    (work / "pool_status.json").write_text(json.dumps(status, indent=2) + "\n")
    work.rename(out)  # a pool folder without the suffix is always a finished build
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
