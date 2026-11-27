"""Tes logika eval_prompts (tanpa torch; model tidak dimuat)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_prompts as ep  # noqa: E402


def test_is_refusal_en_id_dan_apostrof_melengkung():
    assert ep.is_refusal("I can't help with that.")
    assert ep.is_refusal("I can’t help with that.")
    assert ep.is_refusal("Maaf, saya tidak dapat membantu.")
    assert not ep.is_refusal("Adversaries may abuse Process Injection.")


def test_rep_ratio():
    assert ep.rep_ratio("a b c") == 0.0
    assert ep.rep_ratio("a b c d e f g h") == 0.0
    assert ep.rep_ratio("a b c d " * 5) == pytest.approx(1 - 4 / 17)


def test_fact_hit_tanpa_pengaruh_huruf_besar():
    assert ep.fact_hit("It is t1566.", ["T1566"])
    assert not ep.fact_hit("It is T1567.", ["T1566"])


def test_prompts_v0_valid_dan_seimbang():
    rows = ep.load_prompts(ROOT / "evals" / "prompts_v0.jsonl")
    from collections import Counter
    c = Counter(r["category"] for r in rows)
    assert c == {"refuse": 10, "answer_ok": 10, "general": 10, "fact": 10}
    assert all("###" not in r["prompt"] for r in rows)  # tidak bentrok dengan penanda peran


def test_load_prompts_menolak_data_buruk(tmp_path):
    def w(rows):
        p = tmp_path / "p.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return p
    ok = {"id": "a", "category": "general", "prompt": "x"}
    with pytest.raises(ValueError):
        ep.load_prompts(w([ok, ok]))  # id ganda
    with pytest.raises(ValueError):
        ep.load_prompts(w([{"id": "b", "category": "fact", "prompt": "x"}]))  # tanpa expect
    with pytest.raises(ValueError):
        ep.load_prompts(w([{"id": "c", "category": "lain", "prompt": "x"}]))


def test_summarize_hitungan():
    rows = [{"id": "1", "category": "refuse", "prompt": "p"},
            {"id": "2", "category": "fact", "prompt": "p", "expect": ["79"]}]
    sc = [ep.score_row(rows[0], "I can't help.", 4, True),
          ep.score_row(rows[1], "CWE-79", 3, False)]
    s = ep.summarize(sc)
    assert s["refuse"] == {"n": 1, "stopped_eos": 1, "refused": 1, "mean_tokens": 4.0, "mean_rep4": 0.0}
    assert s["fact"]["fact_hits"] == 1 and s["fact"]["stopped_eos"] == 0
