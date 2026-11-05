"""Meta's EnCodec 32 kHz from Hugging Face, split into the two halves a box can carry.

AudioCraft's ``//pretrained/facebook/encodec_32khz`` is this same model, and the
decoder rebuilt from here agrees with the one traced out of a real experiment to
within 8e-7. That is what lets a box gain an encoder after the fact: the encoder
comes from here, and ``pack`` checks the source's decoder against this one before
pairing them.

Needs torch and transformers; only packing and test-model building import it.
"""

from __future__ import annotations

from torch import nn

MODEL_ID = "facebook/encodec_32khz"


def _load(name: str):
    from transformers import EncodecModel

    return EncodecModel.from_pretrained(name).eval()


class HFDecoder(nn.Module):
    """``[B, n_q, T]`` codes -> ``[B, samples]``, called the way AudioCraft's wrapper calls it."""

    def __init__(self, name: str = MODEL_ID) -> None:
        super().__init__()
        model = _load(name)
        model.encoder = nn.Identity()
        self.model = model

    def forward(self, codes):
        return self.model.decode(codes[None], [None])[0][:, 0]


class HFEncoder(nn.Module):
    """``[B, samples]`` audio -> ``[B, n_q, T]`` codes, at the bandwidth that gives the model's codebooks."""

    def __init__(self, name: str = MODEL_ID) -> None:
        super().__init__()
        model = _load(name)
        if model.config.normalize:
            # Boxes decode without per-chunk scales; a normalising codec would
            # need them carried alongside the codes.
            raise ValueError(f"{name} normalises its input; boxes cannot carry its scales")
        model.decoder = nn.Identity()
        self.model = model
        self.bandwidth = float(model.config.target_bandwidths[0])

    def forward(self, audio):
        return self.model.encode(audio[:, None], bandwidth=self.bandwidth).audio_codes[0]
