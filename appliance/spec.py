"""Architecture facts a player needs, with no framework attached.

This lives apart from both box formats so that the shipped runtime — which has
no torch — can describe a model using nothing but the standard library.

It is the only ``ModelSpec`` in the repo. Its name, origin, note and ``model`` come from the model record
(``model export --box``); its architecture is read off the trained solver and checked against that record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields


@dataclass(frozen=True)
class ModelSpec:
    """Architecture and provenance, flat enough to serialise as JSON."""

    dim: int
    num_heads: int
    num_layers: int
    n_q: int
    card: int
    hidden_scale: float
    layer_scale: float | None
    delays: list[int]
    frame_rate: float
    sample_rate: int
    channels: int
    segment_duration: float
    name: str = "untitled"
    origin: str = ""
    max_period: float = 10_000.0
    note: str = ""
    tuning: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    """The model as the repo's model record knew it at export: sig, name, dataset, recipe, parents, epoch, measured.

    Written by ``model export --box`` (``macchina_ricordi/models/export.py``); boxes made
    before October 2026 have none.
    """

    @property
    def special_token(self) -> int:
        return self.card

    @property
    def max_delay(self) -> int:
        return max(self.delays)

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> ModelSpec:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


CARD_SHAPE_KEYS = ("dim", "num_heads", "num_layers", "n_q", "card", "hidden_scale")


def check_card(spec: ModelSpec, card: dict) -> None:
    """Refuse a box whose architecture is not the model record's: the record and the weights must be one model."""
    shape = card.get("shape") or {}
    wrong = [
        f"{key}: record {shape[key]!r}, weights {getattr(spec, key)!r}"
        for key in CARD_SHAPE_KEYS
        if shape.get(key) is not None and float(shape[key]) != float(getattr(spec, key))
    ]
    segment = card.get("segment_duration")
    if segment is not None and float(segment) != float(spec.segment_duration):
        wrong.append(f"segment_duration: record {segment!r}, weights {spec.segment_duration!r}")
    if wrong:
        raise ValueError(f"{card.get('sig')}: the model record and the checkpoint disagree: " + "; ".join(wrong))
