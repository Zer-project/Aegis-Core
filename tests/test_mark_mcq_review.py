"""Tes alat penanda tinjauan MCQ (mark_mcq_review) pada salinan sementara."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mark_mcq_review as mm  # noqa: E402


@pytest.fixture
def d(tmp_path):
    """Salinan set asli dengan status DIRESET ke unreviewed: tes tidak boleh bergantung pada
    status tinjauan yang sebenarnya (yang berubah ketika pemilik menandai soal)."""
    for n in mm.FILES:
        shutil.copy(ROOT / "evals" / n, tmp_path / n)
        recs = mm.load(tmp_path / n)
        for r in recs:
            r["review"] = "unreviewed"
            r.pop("reviewed_by", None)
        mm.save(tmp_path / n, recs)
    return tmp_path


def status(d, name):
    return {r["id"]: r["review"] for r in mm.load(d / name)}


def test_ok_all_dan_reject_berlaku_di_kedua_berkas(d):
    mm.main(["--dir", str(d), "--reviewer", "pemilik", "--ok-all", "--reject", "sec-044", "sec-055"])
    a, b = (status(d, n) for n in mm.FILES)
    assert a == b
    assert a["sec-044"] == a["sec-055"] == "rejected" and a["sec-001"] == "ok"
    assert sum(v == "ok" for v in a.values()) == 58
    assert all(r["reviewed_by"] == "pemilik" for r in mm.load(d / mm.FILES[0]) if r["review"] != "unreviewed")


def test_ok_all_tidak_menghidupkan_kembali_yang_rejected_dan_teks_tak_berubah(d):
    before = {r["id"]: (r["question"], r["options"], r["answer"]) for r in mm.load(d / mm.FILES[0])}
    mm.main(["--dir", str(d), "--reviewer", "p", "--reject", "sec-010"])
    mm.main(["--dir", str(d), "--reviewer", "p", "--ok-all"])
    s = status(d, mm.FILES[0])
    assert s["sec-010"] == "rejected" and s["sec-011"] == "ok"
    after = {r["id"]: (r["question"], r["options"], r["answer"]) for r in mm.load(d / mm.FILES[0])}
    assert after == before


def test_unreview_menghapus_reviewer(d):
    mm.main(["--dir", str(d), "--reviewer", "p", "--ok", "sec-001"])
    mm.main(["--dir", str(d), "--unreview", "sec-001"])
    r = {x["id"]: x for x in mm.load(d / mm.FILES[0])}["sec-001"]
    assert r["review"] == "unreviewed" and "reviewed_by" not in r


@pytest.mark.parametrize("args", [
    ["--ok-all"],                                           # tanpa reviewer
    ["--reviewer", "p", "--ok", "sec-001", "--reject", "sec-001"],  # id di dua daftar
    ["--reviewer", "p", "--reject", "sec-999"],             # id tidak dikenal
])
def test_masukan_tidak_valid_ditolak_dan_berkas_tak_berubah(d, args):
    raw = {n: (d / n).read_bytes() for n in mm.FILES}
    with pytest.raises(SystemExit):
        mm.main(["--dir", str(d)] + args)
    assert {n: (d / n).read_bytes() for n in mm.FILES} == raw


def test_dry_run_tidak_menulis_dan_set_tidak_berpasangan_ditolak(d):
    raw = (d / mm.FILES[0]).read_bytes()
    mm.main(["--dir", str(d), "--reviewer", "p", "--ok-all", "--dry-run"])
    assert (d / mm.FILES[0]).read_bytes() == raw
    recs = mm.load(d / mm.FILES[1])[:-1]
    mm.save(d / mm.FILES[1], recs)
    with pytest.raises(SystemExit):
        mm.main(["--dir", str(d), "--reviewer", "p", "--ok-all"])


def test_format_berkas_tetap_satu_json_per_baris_dan_harness_memuatnya(d):
    mm.main(["--dir", str(d), "--reviewer", "p", "--ok-all", "--reject", "sec-044"])
    sys.path.insert(0, str(ROOT / "scripts"))
    import eval_hf_mcq as em
    items, bad = em.load_jsonl(d / mm.FILES[0], only_reviewed=True)
    assert bad == 0 and len(items) == 59
    assert all(json.loads(x) for x in (d / mm.FILES[0]).read_text(encoding="utf-8").splitlines())
