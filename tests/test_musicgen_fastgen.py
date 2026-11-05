"""CPU tests for the static-KV fast generation path.

The key property: prefill + incremental steps through FastStepper must produce
the same logits as a plain full-sequence causal forward with identical weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from macchina_ricordi.musicgen.fastgen import FastStepper, _supported, enable_fast_generation  # noqa: E402

DIM = 32
HEADS = 4
LAYERS = 2
K_BOOKS = 2
CARD = 17
MAX_PERIOD = 10_000.0


class _Attn(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = HEADS
        self.in_proj_weight = torch.nn.Parameter(torch.randn(3 * DIM, DIM) * 0.05)
        self.in_proj_bias = torch.nn.Parameter(torch.zeros(3 * DIM))
        self.out_proj = torch.nn.Linear(DIM, DIM)
        # flags read by _supported
        self.custom = True
        self.memory_efficient = True
        self.attention_as_float32 = False
        self.kv_repeat = 1
        self.rope = None
        self.past_context = None


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attn()
        self.norm1 = torch.nn.LayerNorm(DIM)
        self.norm2 = torch.nn.LayerNorm(DIM)
        self.linear1 = torch.nn.Linear(DIM, DIM * 2)
        self.linear2 = torch.nn.Linear(DIM * 2, DIM)
        self.activation = torch.nn.functional.gelu
        self.layer_scale_1 = torch.nn.Identity()
        self.layer_scale_2 = torch.nn.Identity()
        self.norm_first = True
        self.cross_attention = None


class _Transformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer() for _ in range(LAYERS)])
        self.max_period = MAX_PERIOD
        self.positional_scale = 1.0
        self.positional_embedding = "sin"


class _Fuser(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fuse2cond = {"prepend": [], "cross": [], "sum": []}


class _MiniLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim = DIM
        self.n_q = K_BOOKS
        self.emb = torch.nn.ModuleList([torch.nn.Embedding(CARD + 1, DIM) for _ in range(K_BOOKS)])
        self.transformer = _Transformer()
        self.out_norm = torch.nn.LayerNorm(DIM)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(DIM, CARD) for _ in range(K_BOOKS)])
        self.fuser = _Fuser()
        self._is_streaming = False

    def forward(self, sequence, conditions, condition_tensors=None, stage=-1):
        raise NotImplementedError("original forward reached (fallback path)")

    # reference implementation: plain full-sequence causal forward
    def reference_forward(self, sequence: torch.Tensor) -> torch.Tensor:
        B, K, T = sequence.shape
        x = sum(self.emb[k](sequence[:, k]) for k in range(K))
        half = DIM // 2
        positions = torch.arange(T).view(1, -1, 1).to(x.dtype)
        adim = torch.arange(half, dtype=x.dtype).view(1, 1, -1)
        phase = positions / (torch.full([], MAX_PERIOD, dtype=x.dtype) ** (adim / (half - 1)))
        x = x + torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        for layer in self.transformer.layers:
            h = layer.norm1(x)
            projected = torch.nn.functional.linear(h, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            q, k, v = projected.view(B, T, 3, HEADS, DIM // HEADS).unbind(2)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
            att = att.transpose(1, 2).reshape(B, T, DIM)
            x = x + layer.self_attn.out_proj(att)
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
        x = self.out_norm(x)
        return torch.stack([self.linears[k](x) for k in range(K)], dim=1)


@pytest.fixture()
def lm() -> _MiniLM:
    torch.manual_seed(0)
    model = _MiniLM()
    model.eval()
    return model


def _tokens(batch: int, steps: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, CARD, (batch, K_BOOKS, steps), generator=g)


def test_incremental_steps_match_full_forward(lm: _MiniLM) -> None:
    tokens = _tokens(2, 12)
    with torch.no_grad():
        reference = lm.reference_forward(tokens)

    stepper = FastStepper(lm, max_steps=16, graph=False)
    stepper.reset(2)
    with torch.no_grad():
        outs = [stepper.prefill(tokens[..., :1])]
        for t in range(1, 12):
            outs.append(stepper.step(tokens[..., t : t + 1]))
    incremental = torch.cat(outs, dim=2)

    assert incremental.shape == reference.shape
    torch.testing.assert_close(incremental, reference, atol=1e-4, rtol=1e-4)


def test_chunked_prefill_matches_full_forward(lm: _MiniLM) -> None:
    tokens = _tokens(1, 10, seed=3)
    with torch.no_grad():
        reference = lm.reference_forward(tokens)
        stepper = FastStepper(lm, max_steps=16, graph=False)
        stepper.reset(1)
        prefill_logits = stepper.prefill(tokens[..., :6])
        step_logits = [stepper.step(tokens[..., t : t + 1]) for t in range(6, 10)]
    incremental = torch.cat([prefill_logits, *step_logits], dim=2)
    torch.testing.assert_close(incremental, reference, atol=1e-4, rtol=1e-4)


def test_reset_gives_reproducible_outputs(lm: _MiniLM) -> None:
    tokens = _tokens(1, 5)
    stepper = FastStepper(lm, max_steps=8, graph=False)
    with torch.no_grad():
        stepper.reset(1)
        first = [stepper.prefill(tokens[..., :1])]
        first += [stepper.step(tokens[..., t : t + 1]) for t in range(1, 5)]
        stepper.reset(1)
        second = [stepper.prefill(tokens[..., :1])]
        second += [stepper.step(tokens[..., t : t + 1]) for t in range(1, 5)]
    torch.testing.assert_close(torch.cat(first, 2), torch.cat(second, 2))


def test_batch_change_reallocates(lm: _MiniLM) -> None:
    stepper = FastStepper(lm, max_steps=8, graph=False)
    stepper.reset(1)
    with torch.no_grad():
        stepper.prefill(_tokens(1, 1))
    assert stepper.batch_size == 1
    with torch.no_grad():
        stepper.prefill(_tokens(3, 2, seed=5))
    assert stepper.batch_size == 3
    assert stepper.k_bufs[0].shape[0] == 3


def test_buffer_exhaustion_raises(lm: _MiniLM) -> None:
    stepper = FastStepper(lm, max_steps=8, graph=False)
    stepper.reset(1)
    with torch.no_grad():
        stepper.prefill(_tokens(1, 8))
        with pytest.raises(RuntimeError, match="exhausted"):
            stepper.step(_tokens(1, 1))


def test_supported_guards(lm: _MiniLM) -> None:
    assert _supported(lm)
    lm.transformer.positional_embedding = "rope"
    assert not _supported(lm)
    lm.transformer.positional_embedding = "sin"
    lm.fuser.fuse2cond["cross"] = ["description"]
    assert not _supported(lm)


class _FakeCompression(torch.nn.Module):
    frame_rate = 50.0


class _FakeModel:
    def __init__(self, lm: _MiniLM) -> None:
        self.lm = lm
        self.compression_model = _FakeCompression()
        self.max_duration = 1.0


def test_enable_patches_streaming_forward(lm: _MiniLM) -> None:
    model = _FakeModel(lm)
    assert enable_fast_generation(model, graph=False)

    tokens = _tokens(1, 6, seed=7)
    with torch.no_grad():
        reference = lm.reference_forward(tokens)

    lm._is_streaming = True
    with torch.no_grad():
        outs = [lm.forward(tokens[..., :1], [], condition_tensors={})]
        for t in range(1, 6):
            outs.append(lm.forward(tokens[..., t : t + 1], [], condition_tensors={}))
    torch.testing.assert_close(torch.cat(outs, 2), reference, atol=1e-4, rtol=1e-4)

    # non-streaming or conditioned calls fall back to the original forward
    lm._is_streaming = False
    with pytest.raises(NotImplementedError):
        lm.forward(tokens, [], condition_tensors={})


def test_enable_rejects_unsupported() -> None:
    torch.manual_seed(0)
    lm = _MiniLM()
    lm.transformer.positional_embedding = "rope"
    assert not enable_fast_generation(_FakeModel(lm), graph=False)
