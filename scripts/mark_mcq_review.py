"""Menandai status tinjauan soal MCQ proyek di kedua berkas berpasangan sekaligus (v0.1.0).

Status: unreviewed (awal) | ok (pemilik sudah memeriksa kunci dan redaksi) | rejected (tidak
pernah dimuat harness). Status satu id selalu sama di kedua berkas, agar versi id dan en tetap
berpasangan. Pemberi tinjauan wajib dicatat (--reviewer). Alat ini tidak mengubah teks soal.

Contoh:  python scripts/mark_mcq_review.py --reviewer zerforge --ok-all --reject sec-044 sec-055
         (--ok-all hanya mengubah soal berstatus unreviewed; soal rejected tidak dihidupkan lagi)
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

__version__ = "0.1.0"
ROOT = Path(__file__).resolve().parents[1]
FILES = ("mcq_id_v0.jsonl", "mcq_en_parallel_v0.jsonl")
STATUS = ("unreviewed", "ok", "rejected")


def load(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def save(path, recs):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def apply_marks(sets, reviewer, ok_all=False, ok=(), reject=(), unreview=()):
    """sets: {nama_berkas: [record]} (diubah di tempat). Mengembalikan Counter status akhir."""
    names = list(sets)
    ids = [r["id"] for r in sets[names[0]]]
    for n in names[1:]:
        if [r["id"] for r in sets[n]] != ids:
            raise ValueError("id/urutan soal berbeda antar berkas: set tidak berpasangan")
    explicit = [list(ok), list(reject), list(unreview)]
    flat = [i for g in explicit for i in g]
    if len(flat) != len(set(flat)):
        raise ValueError("satu id tidak boleh muncul di lebih dari satu daftar (--ok/--reject/--unreview)")
    unknown = sorted(set(flat) - set(ids))
    if unknown:
        raise ValueError(f"id tidak dikenal: {unknown}")
    if (ok_all or ok or reject) and not (reviewer or "").strip():
        raise ValueError("--reviewer wajib diisi untuk menandai ok atau rejected")
    target = {}
    if ok_all:
        for r in sets[names[0]]:
            if r["review"] == "unreviewed":
                target[r["id"]] = "ok"
    target.update({i: "ok" for i in ok})
    target.update({i: "rejected" for i in reject})
    target.update({i: "unreviewed" for i in unreview})
    for n in names:
        for r in sets[n]:
            st = target.get(r["id"])
            if st is None:
                continue
            r["review"] = st
            if st == "unreviewed":
                r.pop("reviewed_by", None)
            else:
                r["reviewed_by"] = reviewer
    final = Counter(r["review"] for r in sets[names[0]])
    for n in names[1:]:
        if [r["review"] for r in sets[n]] != [r["review"] for r in sets[names[0]]]:
            raise ValueError("status berbeda antar berkas setelah penandaan")
    return final


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(ROOT / "evals"))
    ap.add_argument("--files", nargs="+", default=list(FILES))
    ap.add_argument("--reviewer", default="")
    ap.add_argument("--ok-all", action="store_true", help="tandai semua soal unreviewed sebagai ok")
    ap.add_argument("--ok", nargs="*", default=[])
    ap.add_argument("--reject", nargs="*", default=[])
    ap.add_argument("--unreview", nargs="*", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    paths = {n: Path(a.dir) / n for n in a.files}
    sets = {n: load(p) for n, p in paths.items()}
    try:
        final = apply_marks(sets, a.reviewer, a.ok_all, a.ok, a.reject, a.unreview)
    except ValueError as e:
        raise SystemExit(f"GALAT: {e}")
    print({s: final.get(s, 0) for s in STATUS})
    if a.dry_run:
        print("dry-run: tidak ada berkas yang ditulis", file=sys.stderr)
        return
    for n, p in paths.items():
        save(p, sets[n])
    print("ditulis:", ", ".join(str(p) for p in paths.values()), file=sys.stderr)


if __name__ == "__main__":
    main()
