"""Tes make_refusal_data: determinisme, tanpa kebocoran latih/tahan, penanda penolakan."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_prompts as ep  # noqa: E402
import make_refusal_data as mr  # noqa: E402
from format_cyber_sft import ASSISTANT_TAG, USER_TAG  # noqa: E402


def _instr(prompt):
    assert prompt.startswith(USER_TAG) and prompt.endswith(ASSISTANT_TAG)
    return prompt[len(USER_TAG):-len(ASSISTANT_TAG)].strip()


def test_deterministik_dan_ukuran():
    assert mr.build_train() == mr.build_train()
    assert len(mr.build_train()) == 10 * 6 * 4 + 5 * 6 * 3 == 330
    assert len(mr.build_heldout()) == 30 + 10 + 24 + 2 == 66


def test_jumlah_tahan_per_kategori():
    from collections import Counter
    c = Counter(r["category"] for r in mr.build_heldout())
    assert c == {"refuse_seen_action": 10 * 3 * 1 + 5 * 2 * 1, "refuse_unseen_action": 4 * 3 * 2 + 1 * 2 * 1}


def test_tidak_ada_kebocoran():
    train = {_instr(r["prompt"]).casefold() for r in mr.build_train()}
    held = [r["prompt"] for r in mr.build_heldout()]
    v0 = [r["prompt"] for r in ep.load_prompts(ROOT / "evals" / "prompts_v0.jsonl")
          if r["category"] == "refuse"]
    assert len(set(held)) == len(held)
    assert not train & {h.casefold() for h in held}
    assert not train & {v.casefold() for v in v0}
    assert not {h.casefold() for h in held} & {v.casefold() for v in v0}


def test_semua_respons_terdeteksi_sebagai_penolakan_dan_tanpa_penanda_peran():
    for r in mr.build_train():
        assert ep.is_refusal(r["response"]) and "###" not in r["response"]
        assert r["kind"] == "refusal"
    for resp in mr.EN_RESPONSES + mr.ID_RESPONSES:
        assert ep.is_refusal(resp)


def test_berkas_tahan_di_repo_sama_dengan_hasil_pembangkit():
    rows = ep.load_prompts(ROOT / "evals" / "refusal_heldout_v0.jsonl")
    assert rows == mr.build_heldout()


def test_cli_menolak_menimpa_dan_nama_salah(tmp_path):
    out_t, out_h = tmp_path / "r_train.jsonl", tmp_path / "h.jsonl"
    mr.main(["--out-train", str(out_t), "--out-heldout", str(out_h), "--repeat", "2"])
    assert len(out_t.read_text().splitlines()) == 660
    with pytest.raises(FileExistsError):
        mr.main(["--out-train", str(out_t), "--out-heldout", str(tmp_path / "h2.jsonl")])
    with pytest.raises(SystemExit):
        mr.main(["--out-train", str(tmp_path / "salah.jsonl"), "--out-heldout", str(tmp_path / "h3.jsonl")])
