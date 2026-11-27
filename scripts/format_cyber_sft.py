"""Format dataset siber (MITRE/CWE) menjadi pasangan instruksi untuk SFT (v0.1.0, prototipe).

Input : JSONL {"text": ...} keluaran clean_cyber_data.py (*_train.jsonl atau *_eval.jsonl).
Output: JSONL {"prompt": ..., "response": ..., "kind": "mitre"|"cwe"}; nama berkas output
        harus berakhiran split yang sama dengan input.

- Peran ditulis sebagai teks biasa (tanpa token khusus baru; vocab tidak berubah):
      ### User:\n{instruksi}\n### Assistant:\n
  Prompt inferensi HARUS memakai render_prompt() yang sama persis.
- Instruksi dibangun dari template dengan pilihan deterministik (hash nama), bukan acak.
  Ini mengajarkan FORMAT jawaban, bukan pengetahuan baru dan bukan variasi instruksi.
- Dokumen yang tidak cocok pola, kosong, atau memuat penanda peran literal dilewati dan
  dihitung. Tidak mencetak teks dokumen; hanya statistik.
- Split train dan eval diformat terpisah; tidak ada pencampuran antar split.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

__version__ = "0.1.0"
USER_TAG = "### User:\n"
ASSISTANT_TAG = "### Assistant:\n"
ROLE_MARKERS = ("### User:", "### Assistant:")

MITRE_RE = re.compile(r"\ATitle: (?P<name>[^\n]+)\nDescription: (?P<desc>.+)\Z", re.S)
# clean_text() men-strip akhir teks, jadi spasi setelah "Extended Description:" bisa hilang.
CWE_RE = re.compile(
    r"\A(?P<name>CWE-\d+: [^\n]+)\n\nDescription:[ \t]*(?P<desc>.*?)"
    r"\n\nExtended Description:[ \t]*(?P<ext>.*)\Z", re.S)

MITRE_TEMPLATES = ("Describe {name}.",
                   "What is {name} in the MITRE ATT&CK knowledge base?",
                   "Explain {name}.")
CWE_TEMPLATES = ("Explain {name}.",
                 "What is weakness {name}?",
                 "Describe {name}.")


def render_prompt(instruction):
    return f"{USER_TAG}{instruction}\n{ASSISTANT_TAG}"


def _pick(templates, name):
    h = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16)
    return templates[h % len(templates)]


def parse_doc(text):
    """Kembalikan (kind, name, response) atau None bila pola tidak dikenali."""
    m = MITRE_RE.match(text)
    if m:
        return "mitre", m["name"].strip(), m["desc"].strip()
    m = CWE_RE.match(text)
    if m:
        parts = [p.strip() for p in (m["desc"], m["ext"]) if p.strip()]
        return "cwe", m["name"].strip(), " ".join(parts)
    return None


def build_pair(text):
    """Kembalikan (pair, alasan_lewati); tepat salah satunya None."""
    parsed = parse_doc(text)
    if parsed is None:
        return None, "unparsed"
    kind, name, response = parsed
    if not name or not response:
        return None, "empty"
    templates = MITRE_TEMPLATES if kind == "mitre" else CWE_TEMPLATES
    instruction = _pick(templates, name).format(name=name)
    if any(mk in instruction or mk in response for mk in ROLE_MARKERS):
        return None, "role_marker"
    return {"prompt": render_prompt(instruction), "response": response, "kind": kind}, None


def check_paths(inp, out):
    i, o = Path(inp), Path(out)
    for split in ("train", "eval"):
        if i.name.lower().endswith(f"_{split}.jsonl"):
            if not o.name.lower().endswith(f"_{split}.jsonl"):
                raise ValueError(f"nama output harus berakhiran _{split}.jsonl")
            break
    else:
        raise ValueError(f"DITOLAK (nama input harus *_train.jsonl atau *_eval.jsonl): {i.name}")
    if any(w in i.name.lower() for w in ("test", "holdout")):
        raise ValueError(f"DITOLAK (dicadangkan): {i.name}")
    if not i.is_file():
        raise FileNotFoundError(f"tidak ada: {i}")
    return i, o


def format_file(inp, out, overwrite=False):
    i, o = check_paths(inp, out)
    if o.exists() and not overwrite:
        raise FileExistsError(f"sudah ada: {o}; pakai --overwrite")
    o.parent.mkdir(parents=True, exist_ok=True)
    stats = {"read": 0, "written": 0, "mitre": 0, "cwe": 0,
             "skipped_unparsed": 0, "skipped_empty": 0, "skipped_role_marker": 0}
    tmp = o.with_name(o.name + ".partial")
    try:
        with open(i, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
            for n, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{i.name}: baris {n}: JSON tidak valid ({e.msg})") from None
                if not isinstance(rec, dict) or not isinstance(rec.get("text"), str):
                    raise ValueError(f"{i.name}: baris {n}: field 'text' (str) tidak ada")
                stats["read"] += 1
                pair, why = build_pair(rec["text"])
                if pair is None:
                    stats[f"skipped_{why}"] += 1
                    continue
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
                stats["written"] += 1
                stats[pair["kind"]] += 1
        tmp.replace(o)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        st = format_file(a.input, a.output, a.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    print(" ".join(f"{k}={v}" for k, v in st.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
