from __future__ import annotations

import numpy as np

from macchina_ricordi.listening.events import detect_notable_events


def _click_burst(*, sr: int, duration: float) -> np.ndarray:
    samples = int(duration * sr)
    waveform = np.zeros(samples, dtype=np.float32)
    click = np.hanning(64).astype(np.float32)
    for onset_time in np.arange(0.05, duration - 0.05, 0.15):
        idx = int(onset_time * sr)
        if 0 <= idx < samples - len(click):
            waveform[idx : idx + len(click)] += click
    return waveform


def test_detect_notable_events_returns_empty_for_empty_input() -> None:
    assert detect_notable_events(np.asarray([], dtype=np.float32), sr=16_000) == []


def test_detect_notable_events_finds_onsets_and_silence_transitions() -> None:
    sr = 16_000
    first_clicks = _click_burst(sr=sr, duration=2.0)
    middle_silence = np.zeros(int(1.5 * sr), dtype=np.float32)
    second_clicks = _click_burst(sr=sr, duration=2.0)
    waveform = np.concatenate([first_clicks, middle_silence, second_clicks]).astype(np.float32)

    events = detect_notable_events(waveform, sr=sr)

    kinds = [event.kind for event in events]
    assert "onset" in kinds
    assert "silence_start" in kinds
    assert "silence_end" in kinds

    times = [event.time_seconds for event in events]
    assert times == sorted(times)

    silence_starts = [event for event in events if event.kind == "silence_start"]
    assert silence_starts, "expected at least one silence_start event"
    silence_event = silence_starts[0]
    assert silence_event.end_seconds is not None
    assert silence_event.end_seconds > silence_event.time_seconds
    assert 1.5 < silence_event.time_seconds < 2.5
    assert 3.0 < silence_event.end_seconds < 4.0


def test_detect_notable_events_caps_total_count_and_keeps_silence() -> None:
    sr = 16_000
    waveform = np.concatenate(
        [
            _click_burst(sr=sr, duration=2.0),
            np.zeros(int(1.0 * sr), dtype=np.float32),
            _click_burst(sr=sr, duration=2.0),
        ]
    ).astype(np.float32)

    capped = detect_notable_events(waveform, sr=sr, max_events=4)

    assert len(capped) <= 4
    kinds = {event.kind for event in capped}
    assert "silence_start" in kinds
    assert "silence_end" in kinds
