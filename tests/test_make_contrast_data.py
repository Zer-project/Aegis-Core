"""Tes make_contrast_data: determinisme, tanpa kebocoran, kebenaran jawaban terhitung."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_prompts as ep  # noqa: E402
import make_contrast_data as mc  # noqa: E402
import make_refusal_data as mr  # noqa: E402
from format_cyber_sft import ASSISTANT_TAG, USER_TAG  # noqa: E402


def _instr(prompt):
    assert prompt.startswith(USER_TAG) and prompt.endswith(ASSISTANT_TAG)
    return prompt[len(USER_TAG):-len(ASSISTANT_TAG)].strip()


def test_deterministik_dan_ukuran():
    assert mc.build_train() == mc.build_train()
    # aritmetika 30x(3+3) + ibu kota 19x(3+3) + konsep 14x(5+5) + umum 8x(3+3)
    assert len(mc.build_train()) == 180 + 114 + 140 + 48 == 482
    assert len(mc.build_heldout()) == 12 + 16 + 20 + 5 == 53


def test_tidak_ada_kebocoran_latih_tahan_dan_set_lain():
    train = {_instr(r["prompt"]).casefold() for r in mc.build_train()}
    others = [r["prompt"] for r in mc.build_heldout()]
    others += [r["prompt"] for r in ep.load_prompts(ROOT / "evals" / "prompts_v0.jsonl")]
    others += [r["prompt"] for r in mr.build_heldout()]
    assert not train & {o.casefold() for o in others}
    prompts = [r["prompt"] for r in mc.build_heldout()]
    assert len(set(prompts)) == len(prompts)


def test_tidak_memuat_penanda_peran_dan_tidak_terdeteksi_sebagai_penolakan():
    for r in mc.build_train():
        assert "###" not in r["response"] and r["kind"] == "contrast"
        assert not ep.is_refusal(r["response"])


def test_jawaban_aritmetika_dan_ibu_kota_konsisten():
    for r in mc.build_train():
        ins, resp = _instr(r["prompt"]), r["response"]
        m = re.search(r"(\d+) (?:plus|ditambah) (\d+)", ins)
        if m:
            assert resp == f"{m.group(1)} {'plus' if 'plus' in ins else 'ditambah'} {m.group(2)} " \
                           f"{'is' if 'plus' in ins else 'adalah'} {int(m.group(1)) + int(m.group(2))}."
    caps = {en: c for en, _, c in mc.CAPITALS_TRAIN} | {i: c for _, i, c in mc.CAPITALS_TRAIN}
    for r in mc.build_train():
        m = re.match(r"(?:The capital of|Ibu kota) (.+?) (?:is|adalah) (.+)\.$", r["response"])
        if m:
            assert caps[m.group(1)] == m.group(2)


def test_expect_pada_tahan_sesuai_perhitungan():
    for r in mc.build_heldout():
        m = re.search(r"(\d+) (?:plus|ditambah) (\d+)", r["prompt"])
        if m:
            assert r["expect"] == [str(int(m.group(1)) + int(m.group(2)))]


def test_berkas_tahan_di_repo_sama_dengan_hasil_pembangkit():
    assert ep.load_prompts(ROOT / "evals" / "benign_heldout_v0.jsonl") == mc.build_heldout()


def test_cli_menolak_menimpa_dan_nama_salah(tmp_path):
    out_t, out_h = tmp_path / "c_train.jsonl", tmp_path / "h.jsonl"
    mc.main(["--out-train", str(out_t), "--out-heldout", str(out_h), "--repeat", "2"])
    assert len(out_t.read_text().splitlines()) == 964
    with pytest.raises(FileExistsError):
        mc.main(["--out-train", str(out_t), "--out-heldout", str(tmp_path / "h2.jsonl")])
    with pytest.raises(SystemExit):
        mc.main(["--out-train", str(tmp_path / "salah.jsonl"), "--out-heldout", str(tmp_path / "h3.jsonl")])
