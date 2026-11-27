"""Tes format_cyber_sft.py. Data sintetis; hanya stdlib, tanpa jaringan."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import format_cyber_sft as fc  # noqa: E402

MITRE = "Title: Demo Technique\nDescription: Adversaries may do a demo thing."
CWE = ("CWE-79: Demo Weakness\n\nDescription: Short description.\n\n"
       "Extended Description: Longer explanation.")
CWE_NO_EXT = "CWE-80: Other Weakness\n\nDescription: Only short.\n\nExtended Description:"


def write_jsonl(p, texts):
    p.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")


def test_render_prompt_format_tepat():
    assert fc.render_prompt("X") == "### User:\nX\n### Assistant:\n"


def test_parse_mitre_dan_cwe():
    assert fc.parse_doc(MITRE) == ("mitre", "Demo Technique", "Adversaries may do a demo thing.")
    assert fc.parse_doc(CWE) == ("cwe", "CWE-79: Demo Weakness",
                                 "Short description. Longer explanation.")


def test_cwe_extended_kosong_setelah_strip_tetap_terbaca():
    # clean_text() men-strip akhir teks sehingga "Extended Description: " kehilangan spasi.
    assert fc.parse_doc(CWE_NO_EXT) == ("cwe", "CWE-80: Other Weakness", "Only short.")


def test_pasangan_berisi_prompt_berakhir_tag_asisten_dan_respons_tanpa_tag():
    pair, why = fc.build_pair(MITRE)
    assert why is None and pair["kind"] == "mitre"
    assert pair["prompt"].startswith(fc.USER_TAG) and pair["prompt"].endswith(fc.ASSISTANT_TAG)
    assert "Demo Technique" in pair["prompt"]
    assert pair["response"] == "Adversaries may do a demo thing."


def test_pemilihan_template_deterministik():
    assert fc.build_pair(MITRE) == fc.build_pair(MITRE)
    names = [f"Name {i}" for i in range(60)]
    used = {fc._pick(fc.MITRE_TEMPLATES, n) for n in names}
    assert used == set(fc.MITRE_TEMPLATES)  # ketiga template benar-benar terpakai


@pytest.mark.parametrize("text,why", [
    ("teks bebas tanpa pola", "unparsed"),
    ("Title: Kosong\nDescription: x\n### User:\nbocor", "role_marker"),
    ("CWE-1: A\n\nDescription:\n\nExtended Description:", "empty"),
])
def test_dokumen_dilewati_dengan_alasan(text, why):
    pair, got = fc.build_pair(text)
    assert pair is None and got == why


def test_format_file_statistik_dan_isi(tmp_path):
    src = tmp_path / "aegis_train.jsonl"
    write_jsonl(src, [MITRE, CWE, CWE_NO_EXT, "tidak cocok", MITRE.replace("Demo", "### Assistant:")])
    out = tmp_path / "aegis_sft_train.jsonl"
    st = fc.format_file(src, out)
    assert st == {"read": 5, "written": 3, "mitre": 1, "cwe": 2, "skipped_unparsed": 1,
                  "skipped_empty": 0, "skipped_role_marker": 1}
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and all(set(r) == {"prompt", "response", "kind"} for r in rows)
    assert not list(tmp_path.glob("*.partial"))


def test_menolak_menimpa_dan_nama_split_tidak_cocok(tmp_path):
    src = tmp_path / "a_train.jsonl"
    write_jsonl(src, [MITRE])
    out = tmp_path / "a_sft_train.jsonl"
    fc.format_file(src, out)
    with pytest.raises(FileExistsError):
        fc.format_file(src, out)
    fc.format_file(src, out, overwrite=True)
    with pytest.raises(ValueError):  # train tidak boleh keluar sebagai eval
        fc.format_file(src, tmp_path / "a_sft_eval.jsonl")
    bad = tmp_path / "a_holdout_eval.jsonl"
    write_jsonl(bad, [MITRE])
    with pytest.raises(ValueError):
        fc.format_file(bad, tmp_path / "b_eval.jsonl")
    with pytest.raises(ValueError):
        fc.format_file(tmp_path / "x.jsonl", tmp_path / "y_train.jsonl")


def test_json_rusak_membatalkan_tanpa_meninggalkan_output(tmp_path):
    src = tmp_path / "a_train.jsonl"
    src.write_text(json.dumps({"text": MITRE}) + "\n{rusak\n", encoding="utf-8")
    out = tmp_path / "a_sft_train.jsonl"
    with pytest.raises(ValueError):
        fc.format_file(src, out)
    assert not out.exists() and not list(tmp_path.glob("*.partial"))
