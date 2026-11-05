"""Token pre-seeding for the variations tool ("same piece, new performance").

MusicGen's generation loop only writes over ``unknown`` (-1) positions:
``gen_sequence[...] = torch.where(gen_sequence == unknown, next_token, ...)``
— pre-filled tokens survive *and* get fed back through the delay pattern, so
the model conditions on them. Freezing the coarsest RVQ book(s) of a
reference and resampling the rest therefore re-performs the same structural
skeleton with fresh timbre/detail.

This module holds the pure, CPU-testable part; the generation loop itself
lives in ``scripts/run_musicgen_variations.py`` (AudioCraft venv).
"""

from __future__ import annotations

import torch

UNKNOWN_TOKEN = -1


def seed_frozen_codes(
    ref_codes: torch.Tensor,
    *,
    freeze_books: int,
    total_len: int,
    keep_intro_steps: int = 0,
) -> torch.Tensor:
    """Build the pre-seeded ``gen_codes`` template ``[B, K, total_len]``.

    Books ``0..freeze_books-1`` carry the reference for every timestep; the
    first ``keep_intro_steps`` timesteps carry the reference for *all* books
    (so variations open identically before diverging); everything else is
    ``UNKNOWN_TOKEN`` and will be sampled.
    """
    if ref_codes.ndim != 3:
        raise ValueError(f"ref_codes must be [B, K, T], got shape {tuple(ref_codes.shape)}")
    B, K, T = ref_codes.shape
    if not 0 <= freeze_books < K:
        raise ValueError(f"freeze_books must be in [0, {K - 1}], got {freeze_books}")
    if not 0 < total_len <= T:
        raise ValueError(f"total_len must be in [1, {T}], got {total_len}")
    if keep_intro_steps < 0 or keep_intro_steps > total_len:
        raise ValueError(f"keep_intro_steps must be in [0, {total_len}], got {keep_intro_steps}")
    if freeze_books == 0 and keep_intro_steps == 0:
        raise ValueError("Nothing frozen: freeze_books=0 with keep_intro_steps=0 is plain generation")

    gen_codes = torch.full((B, K, total_len), UNKNOWN_TOKEN, dtype=torch.long, device=ref_codes.device)
    if freeze_books > 0:
        gen_codes[:, :freeze_books, :] = ref_codes[:, :freeze_books, :total_len]
    if keep_intro_steps > 0:
        gen_codes[:, :, :keep_intro_steps] = ref_codes[:, :, :keep_intro_steps]
    return gen_codes
