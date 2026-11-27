"""Bundel evaluasi kecil untuk dibawa ke Kaggle (v0.1.0, prototipe, deterministik).

Membaca tiga berkas JSONL {"text": ...} (domain siber, umum Inggris, umum Indonesia) dan menulis
sampel tetap + manifest (sha256 input/keluaran, jumlah dokumen, byte). Urutan sampel ditentukan
oleh sha256 teks, bukan urutan berkas. Hanya data publik yang sudah ada di proyek; unggah sebagai
Kaggle Dataset PRIVATE. Tidak mencetak teks dokumen.
"""
import argparse
import hashlib
import json
from pathlib import Path

__version__ = "0.1.0"
NAMES = ("cyber", "en", "id")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_texts(path, key="text", min_chars=200):
    texts, skipped = [], 0
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict) or not isinstance(obj.get(key), str):
            keys = sorted(obj) if isinstance(obj, dict) else type(obj).__name__
            raise ValueError(f"{path}:{ln}: tidak ada kunci '{key}' bertipe str (ada: {keys})")
        if len(obj[key].strip()) < min_chars:
            skipped += 1
            continue
        texts.append(obj[key])
    return texts, skipped


def pick(texts, n):
    """n dokumen pertama menurut sha256(teks); semuanya bila n <= 0 atau n >= jumlah."""
    ordered = sorted(set(texts), key=lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    return ordered if n <= 0 else ordered[:n]


def build(inputs, out_dir, n_docs, min_chars=200, key="text"):
    out = Path(out_dir)
    out.mkdir(parents=True)  # menolak folder yang sudah ada
    manifest = {"script_version": __version__, "n_docs_requested": n_docs, "min_chars": min_chars,
                "corpora": {}}
    for name in NAMES:
        src = Path(inputs[name])
        texts, skipped = read_texts(src, key, min_chars)
        chosen = pick(texts, 0 if name == "cyber" else n_docs)
        dst = out / f"{name}_eval.jsonl"
        with open(dst, "x", encoding="utf-8") as f:
            for t in chosen:
                f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
        manifest["corpora"][name] = {
            "input": str(src), "input_sha256": sha256_file(src), "docs": len(chosen),
            "skipped_short": skipped, "bytes": sum(len(t.encode("utf-8")) for t in chosen),
            "file": dst.name, "file_sha256": sha256_file(dst)}
    with open(out / "manifest.json", "x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for n in NAMES:
        ap.add_argument(f"--{n}", required=True, help=f"JSONL {n}")
    ap.add_argument("--n-docs", type=int, default=200, help="dokumen per korpus en/id (cyber: semua)")
    ap.add_argument("--out-dir", required=True, help="folder baru (ditolak bila sudah ada)")
    a = ap.parse_args(argv)
    m = build({n: getattr(a, n) for n in NAMES}, a.out_dir, a.n_docs)
    print(json.dumps({k: {"docs": v["docs"], "bytes": v["bytes"], "skipped_short": v["skipped_short"]}
                      for k, v in m["corpora"].items()}, indent=1))


if __name__ == "__main__":
    main()
