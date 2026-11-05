"""The torch reference: the reimplemented LM and the ``.nrsrc`` source format."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from appliance.format import read_source, write_source  # noqa: E402
from appliance.lm import NanoLM, TorchBackend  # noqa: E402
from appliance.player import Knobs, Player  # noqa: E402
from appliance.spec import ModelSpec  # noqa: E402


class StubCodec(torch.nn.Module):
    """``[B, n_q, T]`` -> ``[B, T * hop]``: the shape contract, none of the sound."""

    def __init__(self, hop: int = 8) -> None:
        super().__init__()
        self.hop = hop

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        level = 0.05 + codes.float().mean(dim=1) / 64.0
        return level.unsqueeze(-1).expand(-1, -1, self.hop).flatten(1)


@pytest.fixture
def spec() -> ModelSpec:
    return ModelSpec(
        dim=16,
        num_heads=2,
        num_layers=2,
        n_q=4,
        card=32,
        hidden_scale=2.0,
        layer_scale=1e-4,
        delays=[0, 1, 2, 3],
        frame_rate=10.0,
        sample_rate=80,
        channels=1,
        segment_duration=4.0,
        name="stub",
    )


@pytest.fixture
def codec():
    return torch.jit.trace(StubCodec(8), torch.zeros(1, 4, 5, dtype=torch.long), check_trace=False)


@pytest.fixture
def lm(spec):
    torch.manual_seed(0)
    return NanoLM(spec).eval()


def test_kv_cache_matches_full_forward(spec, lm):
    with torch.no_grad():
        tokens = torch.randint(0, spec.card, (2, spec.n_q, 12))
        full = lm(tokens, 0, lm.fresh_cache(2, 12, torch.float32))
        caches = lm.fresh_cache(2, 12, torch.float32)
        steps = torch.cat([lm(tokens[:, :, t : t + 1], t, caches) for t in range(12)], dim=2)
    assert torch.allclose(steps, full, atol=1e-4, rtol=1e-4)


def test_positions_shift_the_result(spec, lm):
    with torch.no_grad():
        tokens = torch.randint(0, spec.card, (1, spec.n_q, 1))
        a = lm(tokens, 0, lm.fresh_cache(1, 4, torch.float32))
        b = lm(tokens, 5, lm.fresh_cache(1, 4, torch.float32))
    assert not torch.allclose(a, b)


def test_source_round_trip(tmp_path, spec, lm, codec):
    path = write_source(tmp_path / "s.nrsrc", spec=spec, lm_state=lm.state_dict(), codec_module=codec, half=True)
    back_spec, weights, back_codec = read_source(path)
    assert back_spec == spec
    assert all(v.dtype == torch.float16 for v in weights.values())
    codes = torch.randint(0, spec.card, (1, spec.n_q, 9))
    assert torch.equal(back_codec(codes), codec(codes))


def test_source_rejects_foreign_file(tmp_path):
    bad = tmp_path / "bad.nrsrc"
    torch.save({"format": "something/else"}, bad)
    with pytest.raises(ValueError, match="not a"):
        read_source(bad)


def test_torch_backend_plays(spec, lm, codec):
    player = Player(TorchBackend(lm, codec, spec), context_seconds=1.0, round_seconds=0.8, seed=0)
    lengths = [len(player.round(Knobs(top_k=8))[0]) for _ in range(3)]
    assert lengths[1:] == [player.round_frames * player.samples_per_frame] * 2
