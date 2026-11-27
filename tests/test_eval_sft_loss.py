"""Tes eval_sft_loss.window_nll dengan 'model' palsu (hanya torch + numpy, CPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import data_loader as dl  # noqa: E402
import eval_sft_loss as ev  # noqa: E402


def fake_model(x, y):
    """Loss palsu = rata-rata nilai y pada y != -1 (dapat dihitung independen)."""
    keep = y != -1
    assert keep.any()
    return None, y[keep].double().mean()


def make(d, tokens, mask):
    d.mkdir(parents=True)
    (d / "x_eval.bin").write_bytes(np.asarray(tokens, dtype="<u2").tobytes())
    (d / "x_eval.mask").write_bytes(np.asarray(mask, dtype=np.uint8).tobytes())
    (d / "meta_eval.json").write_text(json.dumps({
        "split": "eval", "tokenizer_json_sha256": "t", "vocab_size": 64, "sft": True,
        "shards": {"x_eval.bin": {"tokens": len(tokens), "mask": "x_eval.mask"}}}))
    return dl.load_split_sft(d, "eval")


def test_agregasi_per_token_dan_tidak_bergantung_batch(tmp_path):
    rng = np.random.default_rng(0)
    toks = rng.integers(1, 60, 203)
    mask = (rng.random(203) < 0.6).astype(np.uint8)
    shards, masks, _ = make(tmp_path / "d", toks, mask)
    T = 16
    vals = []
    for i in range(0, len(toks) - T, T):
        keep = mask[i + 1:i + T + 1] != 0
        vals.extend(toks[i + 1:i + T + 1][keep].tolist())
    r1 = ev.window_nll(fake_model, shards, masks, T, B=1)
    r3 = ev.window_nll(fake_model, shards, masks, T, B=3)
    assert r1["target_tokens"] == len(vals) == r3["target_tokens"]
    assert r1["nll"] == pytest.approx(float(np.mean(vals)))
    assert r3["nll"] == pytest.approx(r1["nll"])


def test_tanpa_target_respons_gagal_jelas(tmp_path):
    shards, masks, _ = make(tmp_path / "d", np.arange(1, 101) % 60 + 1, [0] * 100)
    with pytest.raises(ValueError, match="target"):
        ev.window_nll(fake_model, shards, masks, 16)
