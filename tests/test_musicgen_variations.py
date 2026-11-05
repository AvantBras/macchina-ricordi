"""CPU tests for the variations pre-seeding helper."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from macchina_ricordi.instrument.variations import UNKNOWN_TOKEN, seed_frozen_codes  # noqa: E402


def _ref(batch: int = 1, books: int = 4, steps: int = 10) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return torch.randint(0, 17, (batch, books, steps), generator=g)


def test_seed_frozen_codes_freezes_books_and_intro() -> None:
    ref = _ref(batch=3)
    out = seed_frozen_codes(ref, freeze_books=1, total_len=10, keep_intro_steps=2)
    assert out.shape == ref.shape
    assert (out[:, 0, :] == ref[:, 0, :]).all()  # book 0 fully frozen
    assert (out[:, 1:, :2] == ref[:, 1:, :2]).all()  # intro frozen for all books
    assert (out[:, 1:, 2:] == UNKNOWN_TOKEN).all()  # the rest is to be sampled


def test_seed_frozen_codes_truncates_to_total_len() -> None:
    ref = _ref(steps=10)
    out = seed_frozen_codes(ref, freeze_books=2, total_len=6)
    assert out.shape == (1, 4, 6)
    assert (out[:, :2, :] == ref[:, :2, :6]).all()
    assert (out[:, 2:, :] == UNKNOWN_TOKEN).all()


def test_seed_frozen_codes_intro_only() -> None:
    ref = _ref()
    out = seed_frozen_codes(ref, freeze_books=0, total_len=10, keep_intro_steps=3)
    assert (out[:, :, :3] == ref[:, :, :3]).all()
    assert (out[:, :, 3:] == UNKNOWN_TOKEN).all()


def test_seed_frozen_codes_validation() -> None:
    ref = _ref()
    with pytest.raises(ValueError, match="freeze_books"):
        seed_frozen_codes(ref, freeze_books=4, total_len=10)
    with pytest.raises(ValueError, match="total_len"):
        seed_frozen_codes(ref, freeze_books=1, total_len=11)
    with pytest.raises(ValueError, match="keep_intro_steps"):
        seed_frozen_codes(ref, freeze_books=1, total_len=10, keep_intro_steps=11)
    with pytest.raises(ValueError, match="Nothing frozen"):
        seed_frozen_codes(ref, freeze_books=0, total_len=10)
    with pytest.raises(ValueError, match="B, K, T"):
        seed_frozen_codes(ref[0], freeze_books=1, total_len=10)
