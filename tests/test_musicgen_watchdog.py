from __future__ import annotations

from macchina_ricordi.training.watchdog import parse_gpu_watchdog_sample


def test_parse_gpu_watchdog_sample_reads_temperature() -> None:
    sample = parse_gpu_watchdog_sample("89 C, 131.63 W, 90 %, 3968 MiB, P2")

    assert sample.temperature_c == 89.0
    assert sample.gpu_lost is False
    assert sample.query_failed is False


def test_parse_gpu_watchdog_sample_detects_gpu_lost() -> None:
    sample = parse_gpu_watchdog_sample("87 C, 128.16 W, 99 %, [GPU is lost], P2")

    assert sample.temperature_c == 87.0
    assert sample.gpu_lost is True
