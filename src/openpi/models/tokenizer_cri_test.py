"""Tests for CRI discrete token formatting in PaligemmaTokenizer."""

from __future__ import annotations

import numpy as np

from openpi.models.tokenizer import PaligemmaTokenizer


def test_tokenize_includes_separate_cri_span():
    tok = PaligemmaTokenizer(max_len=220)
    state = np.zeros(8, dtype=np.float32)
    cri = np.linspace(0.0, 2.0, 8, dtype=np.float32)
    tokens, mask = tok.tokenize("pick up the cup", state, cri=cri)
    assert tokens.shape == (220,)
    assert mask.dtype == bool
    # Decode prefix and check structure (ignore padding ids).
    # Re-encode expected prompt fragments via the same tokenizer API by reconstructing text.
    # We only assert that CRI-aware prompts differ from state-only prompts.
    tokens_no_cri, _ = tok.tokenize("pick up the cup", state)
    assert not np.array_equal(tokens, tokens_no_cri)


def test_tokenize_cri_requires_state():
    tok = PaligemmaTokenizer(max_len=48)
    try:
        tok.tokenize("pick", state=None, cri=np.zeros(8))
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "cri tokens require" in str(exc)
