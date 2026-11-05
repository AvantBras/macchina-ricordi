"""Static-KV-cache + CUDA-graph fast path for MusicGen micro inference.

AudioCraft's streaming transformer grows its KV cache with a ``torch.cat``
per layer per step and re-derives shapes every step, so the AR loop is
launch-bound (~15% GPU utilisation on the 3060, ~337 kernel launches per
step). This module replaces the per-step path at *inference only* with:

- preallocated per-layer K/V buffers (``[B, H, S_max, Dh]``) written in place
  at the current position — no concatenation, no reallocation;
- attention over the full buffer with an additive ``-inf`` bias beyond the
  current position — every step has an identical shape;
- (stage 2) the whole step — embeddings, all layers, out-norm, heads —
  captured as a single CUDA graph and replayed, collapsing per-step kernel
  launches to one replay call. Sampling stays eager in the caller.

Scope guard: the fast path only arms itself for the unconditional micro
models this repo trains (no cross-attention, no condition tensors, sin
positional embedding, custom+memory-efficient attention). Anything else
falls back to the original AudioCraft forward untouched.

Enable via :func:`enable_fast_generation` (wired into
``scripts/run_musicgen_generate.py:_optimize_for_inference``).
``MACCHINA_RICORDI_NO_FASTGEN=1`` disables just this module.
"""

from __future__ import annotations

import math
import typing as tp

import torch


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class FastStepper:
    """Owns the static buffers and the fused per-step computation for one LM."""

    def __init__(
        self,
        lm: torch.nn.Module,
        *,
        max_steps: int,
        graph: bool = True,
    ) -> None:
        self.lm = lm
        transformer = lm.transformer
        first_param = next(lm.parameters())
        self.device = first_param.device
        self.dtype = first_param.dtype
        self.n_layers = len(transformer.layers)
        layer0 = transformer.layers[0]
        self.num_heads = layer0.self_attn.num_heads
        self.dim = lm.dim
        self.head_dim = self.dim // self.num_heads
        self.max_steps = _round_up(max_steps, 8)
        self.max_period = float(transformer.max_period)
        self.positional_scale = float(transformer.positional_scale)
        self.use_graph = graph and self.device.type == "cuda"

        self.batch_size = 0
        self.pos = 0  # python mirror of pos_t, valid outside captured regions
        self.pos_t: torch.Tensor | None = None
        self.k_bufs: list[torch.Tensor] = []
        self.v_bufs: list[torch.Tensor] = []
        self.bias: torch.Tensor | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._graph_tokens: torch.Tensor | None = None
        self._graph_logits: torch.Tensor | None = None
        self._steps_since_reset = 0

    # ── buffer lifecycle ──────────────────────────────────────────────

    def reset(self, batch_size: int | None = None) -> None:
        """Start a new generation: zero the position, re-mask the bias."""
        if batch_size is not None and batch_size != self.batch_size:
            self._allocate(batch_size)
        self.pos = 0
        self._steps_since_reset = 0
        if self.pos_t is not None:
            self.pos_t.zero_()
        if self.bias is not None:
            self.bias.fill_(float("-inf"))

    def _allocate(self, batch_size: int) -> None:
        self.batch_size = batch_size
        shape = (batch_size, self.num_heads, self.max_steps, self.head_dim)
        self.k_bufs = [
            torch.zeros(shape, device=self.device, dtype=self.dtype)
            for _ in range(self.n_layers)
        ]
        self.v_bufs = [
            torch.zeros(shape, device=self.device, dtype=self.dtype)
            for _ in range(self.n_layers)
        ]
        self.bias = torch.full(
            (1, 1, 1, self.max_steps), float("-inf"), device=self.device, dtype=self.dtype
        )
        self.pos_t = torch.zeros((), device=self.device, dtype=torch.long)
        # any previously captured graph points at freed buffers
        self._graph = None
        self._graph_tokens = None
        self._graph_logits = None

    # ── core math (shared by prefill and step) ────────────────────────

    def _sin_embedding(self, positions: torch.Tensor) -> torch.Tensor:
        """AudioCraft's create_sin_embedding for a [B, T, 1] positions tensor."""
        half_dim = self.dim // 2
        positions = positions.to(self.dtype)
        adim = torch.arange(half_dim, device=self.device, dtype=self.dtype).view(1, 1, -1)
        max_period_tensor = torch.full([], self.max_period, device=self.device, dtype=self.dtype)
        phase = positions / (max_period_tensor ** (adim / (half_dim - 1)))
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)

    def _attention(
        self,
        layer_index: int,
        x: torch.Tensor,
        *,
        write_pos: torch.Tensor,
        prefill_len: int | None = None,
    ) -> torch.Tensor:
        """Self-attention for [B, T, D] x, writing K/V at ``write_pos``.

        ``prefill_len`` (T > 1) uses an eager causal mask over the written
        prefix; the T == 1 step path attends over the full buffer with the
        shared additive bias (static shapes, graph-capturable).
        """
        layer = self.lm.transformer.layers[layer_index]
        attn = layer.self_attn
        B, T, _ = x.shape
        projected = torch.nn.functional.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = projected.view(B, T, 3, self.num_heads, self.head_dim).unbind(dim=2)
        q = q.transpose(1, 2)  # [B, H, T, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        k_buf, v_buf = self.k_bufs[layer_index], self.v_bufs[layer_index]
        if prefill_len is not None:
            k_buf[:, :, :prefill_len] = k
            v_buf[:, :, :prefill_len] = v
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        else:
            index = write_pos.view(1)
            k_buf.index_copy_(2, index, k)
            v_buf.index_copy_(2, index, v)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k_buf, v_buf, attn_mask=self.bias
            )
        out = out.transpose(1, 2).reshape(B, T, self.dim)
        return attn.out_proj(out)

    def _forward_tokens(
        self,
        sequence: torch.Tensor,
        *,
        positions: torch.Tensor,
        write_pos: torch.Tensor,
        prefill_len: int | None = None,
    ) -> torch.Tensor:
        """sequence [B, K, T] -> logits [B, K, T, card]."""
        lm = self.lm
        B, K, T = sequence.shape
        x = sum(lm.emb[k](sequence[:, k]) for k in range(K))
        x = x + self.positional_scale * self._sin_embedding(positions).to(x.dtype)
        for layer_index, layer in enumerate(lm.transformer.layers):
            attn_out = self._attention(
                layer_index, layer.norm1(x), write_pos=write_pos, prefill_len=prefill_len
            )
            x = x + layer.layer_scale_1(attn_out)
            ff = layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
            x = x + layer.layer_scale_2(ff)
        if lm.out_norm is not None:
            x = lm.out_norm(x)
        return torch.stack([lm.linears[k](x) for k in range(K)], dim=1)

    # ── public entry points ───────────────────────────────────────────

    def prefill(self, sequence: torch.Tensor) -> torch.Tensor:
        """Process a prompt chunk [B, K, T], T >= 1. Returns [B, K, T, card]."""
        B, K, T = sequence.shape
        if B != self.batch_size:
            self._allocate(B)
            self.pos = 0
        assert self.pos == 0, "prefill must be the first call after reset"
        assert T <= self.max_steps, f"prompt of {T} steps exceeds buffer {self.max_steps}"
        positions = torch.arange(T, device=self.device).view(1, -1, 1).expand(B, -1, -1)
        logits = self._forward_tokens(
            sequence, positions=positions, write_pos=self.pos_t, prefill_len=T
        )
        assert self.bias is not None and self.pos_t is not None
        self.bias[..., :T] = 0.0
        self.pos_t.fill_(T)
        self.pos = T
        return logits

    def _step_impl(self, tokens: torch.Tensor) -> torch.Tensor:
        """One AR step [B, K, 1] -> [B, K, 1, card]; pure tensor ops on pos_t."""
        assert self.bias is not None and self.pos_t is not None
        # open the bias at the position being written, then advance
        self.bias.view(-1).index_fill_(0, self.pos_t.view(1), 0.0)
        positions = self.pos_t.view(1, 1, 1).expand(tokens.shape[0], -1, -1)
        logits = self._forward_tokens(
            tokens, positions=positions, write_pos=self.pos_t, prefill_len=None
        )
        self.pos_t.add_(1)
        return logits

    _GRAPH_WARMUP_STEPS = 3

    def step(self, tokens: torch.Tensor) -> torch.Tensor:
        B, K, T = tokens.shape
        assert T == 1
        if B != self.batch_size:
            self._allocate(B)
            self.reset(B)
        if self.pos >= self.max_steps:
            raise RuntimeError(f"FastStepper buffer exhausted at {self.max_steps} steps")

        self._steps_since_reset += 1
        if not self.use_graph:
            out = self._step_impl(tokens)
        elif self._graph is None:
            if self._steps_since_reset <= self._GRAPH_WARMUP_STEPS:
                out = self._step_impl(tokens)
            else:
                self._capture(tokens)
                out = self._replay(tokens)
        else:
            out = self._replay(tokens)
        self.pos += 1
        return out

    def _capture(self, tokens: torch.Tensor) -> None:
        torch.cuda.synchronize()
        self._graph_tokens = tokens.clone()
        graph = torch.cuda.CUDAGraph()
        # side-stream warmup as recommended before capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._step_impl(self._graph_tokens)
        torch.cuda.current_stream().wait_stream(stream)
        # the warmup executed for real and advanced state; rewind it. Capture
        # below only RECORDS kernels (they do not run), so no rewind after it.
        self._rewind_one()
        with torch.cuda.graph(graph):
            self._graph_logits = self._step_impl(self._graph_tokens)
        self._graph = graph

    def _rewind_one(self) -> None:
        assert self.bias is not None and self.pos_t is not None
        self.pos_t.sub_(1)
        self.bias.view(-1).index_fill_(0, self.pos_t.view(1), float("-inf"))

    def _replay(self, tokens: torch.Tensor) -> torch.Tensor:
        assert self._graph is not None
        assert self._graph_tokens is not None and self._graph_logits is not None
        self._graph_tokens.copy_(tokens)
        self._graph.replay()
        return self._graph_logits


# ── model wiring ──────────────────────────────────────────────────────


def _supported(lm: torch.nn.Module) -> bool:
    transformer = getattr(lm, "transformer", None)
    if transformer is None or not getattr(transformer, "layers", None):
        return False
    layer = transformer.layers[0]
    attn = getattr(layer, "self_attn", None)
    fuser = getattr(lm, "fuser", None)
    return (
        getattr(transformer, "positional_embedding", None) == "sin"
        and getattr(layer, "norm_first", True)
        and getattr(layer, "cross_attention", None) is None
        and attn is not None
        and getattr(attn, "custom", False)
        and getattr(attn, "memory_efficient", False)
        and not getattr(attn, "attention_as_float32", False)
        and getattr(attn, "kv_repeat", 1) == 1
        and getattr(attn, "rope", None) is None
        and getattr(attn, "past_context", None) is None
        and fuser is not None
        and not fuser.fuse2cond.get("prepend")
        and not fuser.fuse2cond.get("cross")
        and not fuser.fuse2cond.get("sum")
    )


def enable_fast_generation(
    model: tp.Any, *, max_steps: int | None = None, graph: bool = True
) -> bool:
    """Route ``model.lm.forward`` through a :class:`FastStepper` at inference.

    Returns True when armed. Non-matching models (cross-attention, rope,
    conditioned) are left untouched — the original forward keeps running.
    """
    lm = model.lm
    if not _supported(lm):
        return False
    if max_steps is None:
        frame_rate = float(model.compression_model.frame_rate)
        max_steps = int(math.ceil(model.max_duration * frame_rate)) + int(lm.n_q) + 8

    stepper = FastStepper(lm, max_steps=max_steps, graph=graph)
    lm._fast_stepper = stepper
    original_forward = lm.forward
    original_set_streaming = getattr(lm, "_set_streaming", None)

    def patched_set_streaming(streaming: bool) -> None:
        if original_set_streaming is not None:
            original_set_streaming(streaming)
        if streaming:
            stepper.reset(stepper.batch_size or None)

    def patched_forward(sequence, conditions=None, condition_tensors=None, stage=-1):
        conditions = conditions or []
        fast_ok = (
            lm._is_streaming
            and not conditions
            and not condition_tensors
            and stage == -1
            and sequence.dim() == 3
            and sequence.shape[-1] >= 1
        )
        if not fast_ok:
            return original_forward(
                sequence, conditions, condition_tensors=condition_tensors, stage=stage
            )
        if sequence.shape[-1] == 1 and stepper.pos > 0:
            return stepper.step(sequence)
        stepper.reset(sequence.shape[0])
        return stepper.prefill(sequence)

    lm.forward = patched_forward
    lm._set_streaming = patched_set_streaming
    return True
