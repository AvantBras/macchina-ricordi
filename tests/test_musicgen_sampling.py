"""CPU tests for min-p sampling (macchina_ricordi.musicgen.sampling)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from macchina_ricordi.musicgen.sampling import enable_min_p, sample_min_p, set_min_p  # noqa: E402

CARD = 8


def test_sample_min_p_only_draws_from_kept_set() -> None:
    torch.manual_seed(0)
    # p_max = 0.5; with min_p = 0.3 the threshold is 0.15 -> tokens 0, 1 survive
    probs = torch.tensor([[0.5, 0.3, 0.1, 0.06, 0.04]])
    draws = torch.cat([sample_min_p(probs, 0.3) for _ in range(200)]).flatten()
    assert set(draws.tolist()) <= {0, 1}
    assert 0 in draws.tolist() and 1 in draws.tolist()  # renormalised, not argmax


def test_sample_min_p_high_threshold_degenerates_to_argmax() -> None:
    torch.manual_seed(0)
    probs = torch.tensor([[0.6, 0.25, 0.15]])
    draws = torch.cat([sample_min_p(probs, 1.0) for _ in range(50)]).flatten()
    assert set(draws.tolist()) == {0}


def test_sample_min_p_shapes_and_validation() -> None:
    probs = torch.softmax(torch.randn(2, 3, CARD), dim=-1)
    out = sample_min_p(probs, 0.1)
    assert out.shape == (2, 3, 1)
    with pytest.raises(ValueError, match="min_p"):
        sample_min_p(probs, 0.0)
    with pytest.raises(ValueError, match="min_p"):
        sample_min_p(probs, 1.5)


class _FakeLM(torch.nn.Module):
    """Duck-typed stand-in for audiocraft's LMModel sampling surface."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.two_step_cfg = False
        self.cfg_coef = 3.0
        self._fsdp = None
        self._logits = logits  # [B, K, T, card]
        self.original_calls: list[dict] = []
        self.forward_calls = 0

    def forward(self, sequence, conditions=None, condition_tensors=None):
        self.forward_calls += 1
        return self._logits

    def _sample_next_token(
        self,
        sequence,
        cfg_conditions,
        unconditional_state,
        use_sampling=False,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        cfg_coef=None,
        two_step_cfg=None,
    ):
        self.original_calls.append({"use_sampling": use_sampling, "temp": temp, "top_k": top_k})
        return torch.zeros(sequence.shape[0], self._logits.shape[1], 1, dtype=torch.long)


def _fake_model(logits: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(lm=_FakeLM(logits))


def test_enable_min_p_delegates_when_inactive() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 2, 1, CARD)
    model = _fake_model(logits)
    assert enable_min_p(model)

    sequence = torch.zeros(1, 2, 1, dtype=torch.long)
    # min_p unset -> original sampler runs
    model.lm._sample_next_token(sequence, {}, {}, True, 1.2, 50, 0.0)
    assert len(model.lm.original_calls) == 1
    # greedy path delegates even with min_p set
    set_min_p(model, 0.1)
    model.lm._sample_next_token(sequence, {}, {}, False, 1.0, 0, 0.0)
    assert len(model.lm.original_calls) == 2
    assert model.lm.forward_calls == 0


def test_enable_min_p_active_path_samples_from_kept_set() -> None:
    torch.manual_seed(0)
    # One dominant token per (B, K) row: token index == 3
    logits = torch.full((2, 2, 1, CARD), -4.0)
    logits[..., 3] = 4.0
    logits[..., 5] = 3.0  # runner-up survives a mild threshold
    model = _fake_model(logits)
    assert enable_min_p(model)
    set_min_p(model, 0.2)

    sequence = torch.zeros(2, 2, 1, dtype=torch.long)
    draws = [model.lm._sample_next_token(sequence, {}, {}, True, 1.0, 250, 0.0) for _ in range(100)]
    stacked = torch.cat(draws, dim=-1)
    assert stacked.shape == (2, 2, 100)
    assert set(stacked.flatten().tolist()) <= {3, 5}
    assert model.lm.original_calls == []  # min-p path, not the stock sampler
    assert model.lm.forward_calls == 100


def test_enable_min_p_is_idempotent() -> None:
    model = _fake_model(torch.randn(1, 1, 1, CARD))
    assert enable_min_p(model)
    patched = model.lm._sample_next_token
    assert enable_min_p(model)
    assert model.lm._sample_next_token is patched
