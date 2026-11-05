from __future__ import annotations

import re

import pytest

from macchina_ricordi.instrument import controls
from macchina_ricordi.instrument.prompt_transform import MAX_PROMPT_RATE, MAX_TILT_DB, MIN_PROMPT_RATE
from macchina_ricordi.instrument.stream.session import (
    PINNED_INJECTION_HEAVY_WEIGHT,
    PINNED_INJECTION_LIGHT_WEIGHT,
    StreamControl,
)


def test_every_control_names_a_stream_field_and_a_knob() -> None:
    from appliance.player import Knobs

    stream_fields = set(StreamControl.__dataclass_fields__)
    knob_fields = set(Knobs.__dataclass_fields__)
    for control in controls.CONTROLS:
        assert control.stream in stream_fields, control.key
        assert control.key in knob_fields, control.key
        low, high, _ = control.panel
        allowed = control.appliance or control.limits
        assert allowed[0] <= low and (allowed[1] is None or high <= allowed[1]), control.key


def test_both_instruments_read_the_same_limits_and_gestures() -> None:
    from appliance import steering
    from appliance.player import Knobs

    assert (MIN_PROMPT_RATE, MAX_PROMPT_RATE) == controls.TAPE_SPEED.limits
    assert MAX_TILT_DB == steering.MAX_TILT_DB == controls.TILT_LOW.limits[1]
    assert steering.INJECT_LIGHT == PINNED_INJECTION_LIGHT_WEIGHT == controls.BRING_BACK_LIGHTLY.weight
    assert steering.INJECT_HEAVY == PINNED_INJECTION_HEAVY_WEIGHT == controls.BRING_BACK_STRONGLY.weight
    assert (steering.LOW_CROSSOVER_HZ, steering.HIGH_CROSSOVER_HZ) == (250.0, 4000.0)
    wild = Knobs(prompt_rate=1.5, tau_seconds=30.0, tilt_low_db=40.0).clamped()
    assert (wild.prompt_rate, wild.tau_seconds, wild.tilt_low_db) == (1.1, 5.0, 12.0)


def test_mapping_between_the_two_names() -> None:
    assert controls.control("prompt_tilt_high_db") is controls.TILT_HIGH
    assert controls.neutral() == {"tilt_low_db": 0.0, "tilt_mid_db": 0.0, "tilt_high_db": 0.0, "prompt_rate": 1.0}
    with pytest.raises(KeyError):
        controls.control("volume")


def test_clamp_respects_a_model_ceiling() -> None:
    assert controls.TAU.clamp(30.0) == 30.0
    assert controls.TAU.clamp(30.0, ceiling=9.0) == 9.0
    assert controls.TAU.clamp(30.0, appliance=True) == 5.0
    assert controls.TAU.range_text() == "from 0.5"


def test_the_panel_draws_its_sliders_from_the_vocabulary() -> None:
    from appliance.panel import PAGE

    for control in controls.CONTROLS:
        low, high, step = control.panel
        assert re.search(rf'id="{control.key}" min="{low:g}" max="{high:g}" step="{step:g}"', PAGE), control.key
