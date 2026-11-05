"""The MusicGen language model, reimplemented to run alone on a CPU.

This is the same arithmetic AudioCraft performs, written out against a plain
``torch`` install. It exists for two reasons that have nothing to do with
disliking the original:

- **Weights per step is the whole cost.** Batch-1 decoding on a CPU is bound by
  how many bytes of weights it must read per step, not by arithmetic. That makes
  fp16 storage worth roughly a factor of two, and it makes the KV cache's layout
  worth caring about. Owning the loop is what makes those reachable.
- **A dependency you cannot install is a program nobody can run.** AudioCraft
  wants a pinned torch, xformers and a Dora experiment directory. The player has
  to survive on a laptop that has none of them.

The model is a pre-norm causal transformer with LayerScale, no biases anywhere,
sinusoidal positions, ``n_q`` parallel codebook streams, and the delay pattern
that lets one step emit all ``n_q`` codebooks at once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .spec import ModelSpec


def sin_embedding(positions: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """``[B, T, 1]`` positions -> ``[B, T, dim]``, matching AudioCraft's layout.

    Note the concatenation order is cos-then-sin, which is not the more common
    sin-then-cos; getting it backwards produces audio that is wrong in a way no
    loss curve would show.
    """
    half = dim // 2
    positions = positions.float()
    adim = torch.arange(half, device=positions.device, dtype=torch.float32).view(1, 1, -1)
    phase = positions / (max_period ** (adim / (half - 1)))
    return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


@dataclass
class LayerCache:
    """Past keys and values for one layer, pre-allocated to the horizon.

    Grown by slicing into a fixed buffer rather than by ``torch.cat``: the
    profile of the AudioCraft loop shows concatenation as the second-largest
    line after the matmuls, and it is pure copying.
    """

    k: torch.Tensor
    v: torch.Tensor
    length: int = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = k.shape[2]
        end = self.length + n
        self.k[:, :, self.length : end] = k
        self.v[:, :, self.length : end] = v
        self.length = end
        return self.k[:, :, :end], self.v[:, :, :end]


class Layer(nn.Module):
    """One pre-norm block: ``x + ls1·attn(norm1(x))`` then ``x + ls2·ff(norm2(x))``."""

    def __init__(self, dim: int, num_heads: int, hidden: int, layer_scale: float | None) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # Initialised, not merely allocated: loading a box overwrites this, but
        # an un-loaded NanoLM must still be a valid module rather than a bag of
        # whatever was in that memory.
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) / math.sqrt(dim))
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.linear1 = nn.Linear(dim, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, dim, bias=False)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.has_scale = layer_scale is not None
        if self.has_scale:
            self.layer_scale_1 = nn.Parameter(torch.full((dim,), float(layer_scale)))
            self.layer_scale_2 = nn.Parameter(torch.full((dim,), float(layer_scale)))

    def forward(self, x: torch.Tensor, cache: LayerCache, causal: bool) -> torch.Tensor:
        B, T, C = x.shape
        h = self.norm1(x)
        qkv = F.linear(h, self.in_proj_weight)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (B, T, self.num_heads, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        k, v = cache.append(k, v)
        # One step attending to the past needs no mask; a multi-step prime does.
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal and T > 1)
        attn = attn.transpose(1, 2).reshape(B, T, C)
        attn = self.out_proj(attn)
        x = x + (attn * self.layer_scale_1 if self.has_scale else attn)
        h = self.norm2(x)
        ff = self.linear2(F.gelu(self.linear1(h)))
        return x + (ff * self.layer_scale_2 if self.has_scale else ff)


class NanoLM(nn.Module):
    """``n_q`` codebook streams over one causal transformer."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        dim = spec.dim
        hidden = int(spec.hidden_scale * dim)
        self.emb = nn.ModuleList([nn.Embedding(spec.card + 1, dim) for _ in range(spec.n_q)])
        self.layers = nn.ModuleList(
            [Layer(dim, spec.num_heads, hidden, spec.layer_scale) for _ in range(spec.num_layers)]
        )
        self.out_norm = nn.LayerNorm(dim, eps=1e-5)
        self.linears = nn.ModuleList([nn.Linear(dim, spec.card, bias=False) for _ in range(spec.n_q)])

    # ---- weight loading -------------------------------------------------

    def load_audiocraft_state(self, state: dict[str, torch.Tensor]) -> None:
        """Map AudioCraft's ``LMModel`` key names onto this module."""
        mapped: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.startswith("transformer.layers."):
                rest = k.removeprefix("transformer.layers.")
                idx, tail = rest.split(".", 1)
                tail = tail.replace("self_attn.in_proj_weight", "in_proj_weight")
                tail = tail.replace("self_attn.out_proj.", "out_proj.")
                tail = tail.replace("layer_scale_1.scale", "layer_scale_1")
                tail = tail.replace("layer_scale_2.scale", "layer_scale_2")
                mapped[f"layers.{idx}.{tail}"] = v
            else:
                mapped[k] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        real_missing = [m for m in missing if "layer_scale" not in m or self.spec.layer_scale]
        if real_missing or unexpected:
            raise ValueError(f"weight mismatch: missing={real_missing} unexpected={unexpected}")

    # ---- inference ------------------------------------------------------

    def fresh_cache(self, batch: int, horizon: int, dtype: torch.dtype) -> list[LayerCache]:
        caches = []
        for layer in self.layers:
            shape = (batch, layer.num_heads, horizon, layer.head_dim)
            caches.append(LayerCache(torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)))
        return caches

    def forward(self, tokens: torch.Tensor, offset: int, caches: list[LayerCache]) -> torch.Tensor:
        """``[B, n_q, T]`` tokens at sequence ``offset`` -> ``[B, n_q, T, card]`` logits."""
        B, K, T = tokens.shape
        x = sum(self.emb[k](tokens[:, k]) for k in range(K))
        positions = torch.arange(T).view(1, -1, 1) + offset
        x = x + sin_embedding(positions, self.spec.dim, self.spec.max_period).to(x.dtype)
        for layer, cache in zip(self.layers, caches, strict=True):
            x = layer(x, cache, causal=True)
        x = self.out_norm(x)
        return torch.stack([lin(x) for lin in self.linears], dim=1)


class TorchBackend:
    """The reference implementation behind the player's backend interface.

    Not what ships — that is ONNX Runtime — but what a packed box is checked
    against, and a way to play a ``.nrsrc`` source during development.
    """

    def __init__(
        self, lm: NanoLM, codec: torch.jit.ScriptModule, spec: ModelSpec, *, threads: int | None = None
    ) -> None:
        if threads:
            torch.set_num_threads(int(threads))
        self.lm = lm.eval()
        self.codec = codec
        self.spec = spec
        self.dtype = next(lm.parameters()).dtype

    can_encode = False

    def encode(self, audio: np.ndarray) -> np.ndarray:
        raise RuntimeError("a .nrsrc source carries no encoder; pack it to steer the prompt")

    @torch.no_grad()
    def start(self, batch: int, horizon: int) -> list[LayerCache]:
        return self.lm.fresh_cache(batch, horizon, self.dtype)

    @torch.no_grad()
    def step(self, tokens: np.ndarray, offset: int, state: list[LayerCache]) -> tuple[np.ndarray, list[LayerCache]]:
        logits = self.lm(torch.from_numpy(np.ascontiguousarray(tokens, dtype=np.int64)), offset, state)
        return logits.float().numpy(), state

    @torch.no_grad()
    def decode(self, codes: np.ndarray) -> np.ndarray:
        audio = self.codec(torch.from_numpy(np.ascontiguousarray(codes, dtype=np.int64)))
        return audio[0].float().numpy().astype(np.float32, copy=False)


def load_source_backend(path, *, threads: int | None = None, prefer_fp32: bool = False) -> TorchBackend:
    """Open a ``.nrsrc``. fp16 weights stay fp16 unless asked: on a CPU the cost is
    bytes of weights read per step, so halving them is close to a free third."""
    from .format import read_source

    spec, weights, codec = read_source(path)
    lm = NanoLM(spec)
    lm.load_audiocraft_state({k: v.float() for k, v in weights.items()})
    if any(v.dtype == torch.float16 for v in weights.values()) and not prefer_fp32:
        lm = lm.half()
    return TorchBackend(lm, codec, spec, threads=threads)
