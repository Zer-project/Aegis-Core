"""Tes pipeline SFT Aegis-Core: prepare_sft, get_batch_sft, train --sft.
Data sintetis; tanpa data nyata, tanpa jaringan; hanya CPU."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
for sub in ("", "tokenizer", "scripts"):
    sys.path.insert(0, str(ROOT / sub) if sub else str(ROOT))
import data_loader as dl  # noqa: E402
import format_cyber_sft as fc  # noqa: E402
import prepare_sft as ps  # noqa: E402
import train as tr  # noqa: E402
import train_tokenizer as tt  # noqa: E402
from model import AegisConfig, AegisNano  # noqa: E402

VOCAB = 300
WORDS = ["adversaries", "may", "use", "tool", "to", "access", "the", "system", "and",
         "execute", "code", "weakness", "input", "buffer", "memory", "data", "network"]
NAME = "aegis_sft_train"


def make_pairs(n, seed, tag="Item"):
    rnd = random.Random(seed)
    pairs = []
    for i in range(n):
        body = " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(8, 30)))
        pair, why = fc.build_pair(f"Title: {tag}{i}\nDescription: {body}.")
        assert why is None
        pairs.append(pair)
    return pairs


def write_pairs(path, pairs):
    path.write_text("".join(json.dumps(p) + "\n" for p in pairs), encoding="utf-8")


@pytest.fixture(scope="module")
def sft(tmp_path_factory):
    d = tmp_path_factory.mktemp("sft")
    pairs, eval_pairs = make_pairs(60, 1), make_pairs(25, 2, tag="Held")
    corpus = d / "en_train.jsonl"
    corpus.write_text("".join(json.dumps({"text": p["prompt"] + p["response"]}) + "\n"
                              for p in pairs), encoding="utf-8")
    tokdir, out = d / "tok", d / "tokens"
    tt.train([corpus], VOCAB, tokdir)
    write_pairs(d / f"{NAME}.jsonl", pairs)
    write_pairs(d / "aegis_sft_eval.jsonl", eval_pairs)
    meta = ps.prepare([d / f"{NAME}.jsonl"], "train", tokdir, out)
    ps.prepare([d / "aegis_sft_eval.jsonl"], "eval", tokdir, out)
    return {"d": d, "tokdir": tokdir, "out": out, "meta": meta, "pairs": pairs}


def arrays(sft):
    toks = np.fromfile(sft["out"] / f"{NAME}.bin", dtype="<u2").astype(np.int64)
    mask = np.fromfile(sft["out"] / f"{NAME}.mask", dtype=np.uint8)
    return toks, mask


def test_mask_dan_token_sesuai_batas_prompt_respons(sft):
    tok = Tokenizer.from_file(str(sft["tokdir"] / "tokenizer.json"))
    eos = sft["meta"]["eos_id"]
    pair = sft["pairs"][0]
    p, r = tok.encode(pair["prompt"]).ids, tok.encode(pair["response"]).ids
    toks, mask = arrays(sft)
    n = len(p) + len(r) + 1
    assert toks[:n].tolist() == p + r + [eos]
    assert mask[:n].tolist() == [0] * len(p) + [1] * (len(r) + 1)
    assert toks.size == mask.size
    info = sft["meta"]["shards"][f"{NAME}.bin"]
    assert info["loss_tokens"] == int(mask.sum()) and info["docs"] == 60
    assert sft["meta"]["sft"] is True and set(np.unique(mask)) == {0, 1}


def test_get_batch_sft_sama_dengan_perhitungan_independen(sft):
    toks, mask = arrays(sft)
    T, B = 32, 8
    win = sliding_window_view(toks, T + 1)
    for step in range(5):
        x, y = dl.get_batch_sft(sft["out"], "train", B, T, seed=7, step=step)
        assert x.shape == y.shape == (B, T) and x.dtype == y.dtype == torch.int64
        for row in range(B):
            cand = np.flatnonzero((win[:, :T] == x[row].numpy()).all(axis=1))
            assert cand.size >= 1
            expected = [np.where(mask[i + 1:i + T + 1] != 0, toks[i + 1:i + T + 1], -1)
                        for i in cand]
            assert any(np.array_equal(e, y[row].numpy()) for e in expected)
            assert (y[row] != -1).any()  # tidak ada baris semua-ignore (loss NaN)


def test_get_batch_sft_deterministik_dan_bergantung_pada_langkah(sft):
    a = dl.get_batch_sft(sft["out"], "train", 4, 32, seed=3, step=2)
    b = dl.get_batch_sft(sft["out"], "train", 4, 32, seed=3, step=2)
    c = dl.get_batch_sft(sft["out"], "train", 4, 32, seed=3, step=3)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def write_manual(d, tokens, mask, sft_flag=True, mask_name="x_train.mask"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "x_train.bin").write_bytes(np.asarray(tokens, dtype="<u2").tobytes())
    (d / mask_name).write_bytes(np.asarray(mask, dtype=np.uint8).tobytes())
    info = {"tokens": len(tokens)}
    meta = {"split": "train", "tokenizer_json_sha256": "tokX", "vocab_size": 64,
            "shards": {"x_train.bin": info}}
    if sft_flag:
        meta["sft"] = True
        info["mask"] = mask_name
    (d / "meta_train.json").write_text(json.dumps(meta))


def test_jendela_tanpa_target_respons_ditarik_ulang(tmp_path):
    n = 600
    rng = np.random.default_rng(0)
    # separuh pertama murni prompt (mask 0): jendela di sana harus ditolak dan ditarik ulang
    write_manual(tmp_path, rng.integers(1, 60, n), [0] * 300 + [1] * 300)
    for step in range(20):
        _, y = dl.get_batch_sft(tmp_path, "train", 8, 16, seed=1, step=step)
        assert ((y != -1).any(dim=1)).all()


def test_gagal_jelas_bila_tidak_ada_target_respons(tmp_path):
    write_manual(tmp_path, np.arange(1, 201) % 60 + 1, [0] * 200)
    with pytest.raises(ValueError, match="target respons"):
        dl.get_batch_sft(tmp_path, "train", 2, 16, seed=1)


def test_menolak_data_non_sft_hash_salah_dan_mask_rusak(tmp_path):
    write_manual(tmp_path / "plain", np.arange(1, 201) % 60 + 1, [1] * 200, sft_flag=False)
    with pytest.raises(ValueError, match="SFT"):
        dl.get_batch_sft(tmp_path / "plain", "train", 2, 16, seed=1)
    write_manual(tmp_path / "ok", np.arange(1, 201) % 60 + 1, [1] * 200)
    with pytest.raises(ValueError, match="tokenizer"):
        dl.get_batch_sft(tmp_path / "ok", "train", 2, 16, seed=1, expected_tokenizer_sha256="zzz")
    (tmp_path / "ok" / "x_train.mask").write_bytes(b"\x01" * 150)  # ukuran tidak cocok meta
    with pytest.raises(ValueError, match="mask"):
        dl.get_batch_sft(tmp_path / "ok", "train", 2, 16, seed=1)
    write_manual(tmp_path / "bad", np.arange(1, 201) % 60 + 1, [1] * 200, mask_name="../x.mask")
    with pytest.raises(ValueError, match="mask"):
        dl.get_batch_sft(tmp_path / "bad", "train", 2, 16, seed=1)


def test_loss_model_hanya_menghitung_token_respons(sft):
    torch.manual_seed(0)
    cfg = AegisConfig(vocab_size=VOCAB, block_size=32, n_layer=1, n_head=2, n_embd=32,
                      dropout=0.0)
    model = AegisNano(cfg).eval()
    x, y = dl.get_batch_sft(sft["out"], "train", 8, 32, seed=5, step=0)
    assert (y == -1).any() and (y != -1).any()
    with torch.no_grad():
        logits, loss = model(x, y)
        sel = y != -1
        assert torch.allclose(loss, F.cross_entropy(logits[sel], y[sel]), atol=1e-6)


def test_prepare_sft_melewati_khusus_kosong_dan_menolak_timpa(sft, tmp_path):
    ok = make_pairs(1, 9)[0]
    rows = [ok,
            {"prompt": ok["prompt"], "response": "x <|endoftext|> y"},
            {"prompt": "", "response": "respons tanpa prompt"},
            {"prompt": ok["prompt"], "response": ""}]
    f = tmp_path / "edge_train.jsonl"
    write_pairs(f, rows)
    meta = ps.prepare([f], "train", sft["tokdir"], tmp_path / "o")
    info = meta["shards"]["edge_train.bin"]
    assert (info["docs"], info["skipped_special"], info["skipped_empty"]) == (1, 1, 2)
    assert (tmp_path / "o" / "edge_train.bin").stat().st_size == info["tokens"] * 2
    assert (tmp_path / "o" / "edge_train.mask").stat().st_size == info["tokens"]
    assert not list((tmp_path / "o").glob("*.partial"))
    with pytest.raises(FileExistsError):
        ps.prepare([f], "train", sft["tokdir"], tmp_path / "o")
    bad = tmp_path / "bad_train.jsonl"
    bad.write_text(json.dumps({"text": "bukan format sft"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt"):
        ps.prepare([bad], "train", sft["tokdir"], tmp_path / "o2")
    assert not list((tmp_path / "o2").glob("*.partial"))


def sargs(data_dir, out_dir, *extra):
    return tr.parse_args(["--data-dir", str(data_dir), "--out-dir", str(out_dir),
                          "--block-size", "32", "--batch-size", "4", "--n-layer", "1",
                          "--n-head", "2", "--n-embd", "32", "--dropout", "0.0",
                          "--log-interval", "1", "--ckpt-interval", "1000", *extra])


def test_train_sft_berjalan_evaluasi_dan_menyimpan_status(sft, tmp_path):
    r = tr.run(sargs(sft["out"], tmp_path / "run", "--sft", "--max-steps", "10",
                     "--eval-interval", "5", "--eval-batches", "2"))
    assert all(np.isfinite(r["losses"])) and r["eval_loss"] is not None
    c = torch.load(tmp_path / "run" / "ckpt.pt", weights_only=True)
    assert c["sft"] is True and c["script_version"] == tr.__version__


def test_flag_sft_harus_cocok_dengan_meta(sft, tmp_path):
    with pytest.raises(ValueError, match="--sft"):
        tr.run(sargs(sft["out"], tmp_path / "a", "--max-steps", "3"))  # data SFT tanpa --sft
    write_manual(tmp_path / "plain", np.arange(1, 400) % 60 + 1, [1] * 399, sft_flag=False)
    with pytest.raises(ValueError, match="--sft"):
        tr.run(sargs(tmp_path / "plain", tmp_path / "b", "--sft", "--max-steps", "3"))


def test_resume_sft_identik_dan_menolak_mode_berbeda(sft, tmp_path):
    ra = tr.run(sargs(sft["out"], tmp_path / "A", "--sft", "--max-steps", "10"))
    rb1 = tr.run(sargs(sft["out"], tmp_path / "B", "--sft", "--max-steps", "10",
                       "--stop-after", "4"))
    with pytest.raises(ValueError):  # resume tanpa --sft pada data SFT
        tr.run(sargs(sft["out"], tmp_path / "B", "--max-steps", "10", "--resume"))
    rb2 = tr.run(sargs(sft["out"], tmp_path / "B", "--sft", "--max-steps", "10", "--resume"))
    assert rb1["losses"] + rb2["losses"] == ra["losses"]


def test_checkpoint_pretraining_lama_tanpa_kunci_sft_tetap_bisa_resume(tmp_path):
    write_manual(tmp_path / "plain", np.tile(np.arange(1, 17), 100), [1] * 1600, sft_flag=False)
    tr.run(sargs(tmp_path / "plain", tmp_path / "r", "--max-steps", "6", "--stop-after", "3"))
    p = tmp_path / "r" / "ckpt.pt"
    c = torch.load(p, weights_only=True)
    c.pop("sft")
    torch.save(c, p)
    r = tr.run(sargs(tmp_path / "plain", tmp_path / "r", "--max-steps", "6", "--resume"))
    assert r["start_step"] == 3 and r["end_step"] == 6
