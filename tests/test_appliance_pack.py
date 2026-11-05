"""Packing: the exported ONNX graphs must reproduce the torch reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from appliance.lm import NanoLM  # noqa: E402
from appliance.pack import export_decoder, export_lm, verify  # noqa: E402
from appliance.spec import ModelSpec  # noqa: E402


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
def parts(spec):
    torch.manual_seed(0)
    lm = NanoLM(spec).eval()
    codec = torch.jit.trace(StubCodec(8), torch.zeros(1, 4, 5, dtype=torch.long), check_trace=False)
    return lm, codec, export_lm(lm, spec), export_decoder(codec, spec)


class StubCodec(torch.nn.Module):
    def __init__(self, hop: int = 8) -> None:
        super().__init__()
        self.hop = hop

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        level = 0.05 + codes.float().mean(dim=1) / 64.0
        return level.unsqueeze(-1).expand(-1, -1, self.hop).flatten(1)


def test_verify_refuses_a_different_model(spec, parts):
    lm, codec, _, decoder_onnx = parts
    torch.manual_seed(99)
    other = export_lm(NanoLM(spec).eval(), spec)
    with pytest.raises(RuntimeError, match="do not reproduce"):
        verify(lm, codec, spec, other, decoder_onnx, prime=30, steps=12, decode_frames=23)
