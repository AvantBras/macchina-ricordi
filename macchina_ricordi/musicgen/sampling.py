"""min-p sampling for MusicGen micro inference (arXiv:2407.01082).

min-p keeps tokens whose post-temperature probability is at least
``min_p * p_max`` and renormalises before sampling. The truncation adapts to
the model's per-step confidence: confident steps stay narrow, uncertain steps
stay wide. That is exactly the regime the mega/deep micros live in — they need
temperature >= 1.4 to avoid silence collapse, and a fixed top-k either lets
junk through (large k) or kills diversity (small k).

Like fastgen, this is an *instance* patch — zero vendored-code edits.
``enable_min_p`` replaces ``lm._sample_next_token`` with a wrapper that:

- delegates to the original method whenever ``lm._nr_min_p <= 0``, greedy
  decoding is requested, or two-step CFG is in play (behaviour identical to
  stock audiocraft in all of those cases);
- otherwise mirrors the original forward + CFG-mix, then samples with min-p
  (min-p *replaces* top-k/top-p when active).

The active threshold is read from ``lm._nr_min_p`` each step, so it is
live-editable mid-stream through the control file.
"""
from __future__ import annotations

import typing as tp

import torch

__all__ = ["sample_min_p", "enable_min_p", "set_min_p"]


def sample_min_p(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Sample one token per distribution, keeping only p >= min_p * p_max.

    ``probs`` has shape ``[..., card]`` and must already be post-temperature
    probabilities. The argmax token always survives, so the renormalised
    distribution is never empty. Returns indices of shape ``[..., 1]``.
    """
    if not 0.0 < min_p <= 1.0:
        raise ValueError(f"min_p must be in (0, 1], got {min_p}")
    max_p = probs.amax(dim=-1, keepdim=True)
    kept = probs.masked_fill(probs < (min_p * max_p), 0.0)
    kept = kept / kept.sum(dim=-1, keepdim=True)
    flat = kept.reshape(-1, kept.shape[-1])
    out = torch.multinomial(flat, num_samples=1)
    return out.reshape(*kept.shape[:-1], 1)


def set_min_p(model: tp.Any, min_p: float) -> None:
    """Record the live min-p threshold on the LM (<=0 disables)."""
    model.lm._nr_min_p = float(min_p)


def enable_min_p(model: tp.Any) -> bool:
    """Patch ``model.lm._sample_next_token`` with a min-p-aware variant."""
    lm = model.lm
    original = getattr(lm, "_sample_next_token", None)
    if original is None or getattr(lm, "_nr_min_p_patched", False):
        return original is not None

    def patched_sample_next_token(
        sequence: torch.Tensor,
        cfg_conditions: tp.Any,
        unconditional_state: tp.Any,
        use_sampling: bool = False,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        cfg_coef: float | None = None,
        two_step_cfg: bool | None = None,
    ) -> torch.Tensor:
        min_p = float(getattr(lm, "_nr_min_p", 0.0) or 0.0)
        two_step = lm.two_step_cfg if two_step_cfg is None else two_step_cfg
        if (
            min_p <= 0.0
            or not use_sampling
            or temp <= 0.0
            or (two_step and cfg_conditions != {})
        ):
            return original(
                sequence,
                cfg_conditions,
                unconditional_state,
                use_sampling,
                temp,
                top_k,
                top_p,
                cfg_coef=cfg_coef,
                two_step_cfg=two_step_cfg,
            )

        # Mirrors the single-pass branch of LMModel._sample_next_token.
        B = sequence.shape[0]
        coef = lm.cfg_coef if cfg_coef is None else cfg_coef
        model_fn = lm if getattr(lm, "_fsdp", None) is None else lm._fsdp
        assert isinstance(cfg_conditions, dict)
        condition_tensors = cfg_conditions
        if condition_tensors:
            sequence = torch.cat([sequence, sequence], dim=0)
        all_logits = model_fn(sequence, conditions=[], condition_tensors=condition_tensors)
        if condition_tensors:
            cond_logits, uncond_logits = all_logits.split(B, dim=0)
            logits = uncond_logits + (cond_logits - uncond_logits) * coef
        else:
            logits = all_logits

        logits = logits.permute(0, 1, 3, 2)[..., -1]  # [B, K, card]
        probs = torch.softmax(logits / temp, dim=-1)
        return sample_min_p(probs, min_p)

    lm._sample_next_token = patched_sample_next_token
    lm._nr_min_p_patched = True
    if not hasattr(lm, "_nr_min_p"):
        lm._nr_min_p = 0.0
    return True
