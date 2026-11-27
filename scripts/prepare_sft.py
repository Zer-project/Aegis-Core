"""Persiapan token SFT Aegis-Core (v0.1.0, prototipe).

Input : JSONL {"prompt": str, "response": str} (keluaran format_cyber_sft.py),
        nama *_train.jsonl atau *_eval.jsonl (aturan nama sama dengan prepare_tokens.py).
Output: per berkas, <stem>.bin (uint16, sama dengan format pretraining) dan <stem>.mask
        (uint8, panjang sama): 1 = token ikut dihitung loss (respons + EOS), 0 = prompt.
        meta_{split}.json berisi sft=true dan nama/hash mask.

- Prompt dan respons di-encode TERPISAH lalu digabung (+EOS); cocok dengan generate.py
  yang meng-encode prompt secara terpisah. Mask dihitung di sini, bukan dicari ulang di loader.
- Dokumen dengan token khusus literal, prompt kosong, atau respons kosong dilewati dan dihitung.
- Tidak mencetak teks atau ID dokumen.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import format_cyber_sft as fc
import prepare_tokens as pt

__version__ = "0.1.0"


def _write_shard(path, tok, eos, tmp_bin, tmp_mask, vocab):
    hb, hm = hashlib.sha256(), hashlib.sha256()
    docs = skipped_special = skipped_empty = tokens = loss_tokens = 0
    resp_lens = []
    with open(path, "r", encoding="utf-8") as fin, \
            open(tmp_bin, "wb") as fb, open(tmp_mask, "wb") as fm:
        for n, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path.name}: baris {n}: JSON tidak valid ({e.msg})") from None
            if not isinstance(rec, dict) or not isinstance(rec.get("prompt"), str) \
                    or not isinstance(rec.get("response"), str):
                raise ValueError(f"{path.name}: baris {n}: field 'prompt'/'response' (str) tidak ada")
            p, r = rec["prompt"], rec["response"]
            if any(s in p or s in r for s in pt.SPECIALS):
                skipped_special += 1
                continue
            p_ids = tok.encode(p).ids if p else []
            r_ids = tok.encode(r).ids if r else []
            if not p_ids or not r_ids:
                skipped_empty += 1
                continue
            ids = p_ids + r_ids + [eos]
            if max(ids) >= vocab:
                raise ValueError(f"{path.name}: baris {n}: id di luar vocab")
            mask = [0] * len(p_ids) + [1] * (len(r_ids) + 1)
            b = np.asarray(ids, dtype="<u2").tobytes()
            m = np.asarray(mask, dtype=np.uint8).tobytes()
            fb.write(b)
            fm.write(m)
            hb.update(b)
            hm.update(m)
            docs += 1
            tokens += len(ids)
            loss_tokens += len(r_ids) + 1
            resp_lens.append(len(r_ids) + 1)
    q = (lambda v: int(np.percentile(resp_lens, v))) if resp_lens else (lambda v: 0)
    return {"tokens": tokens, "docs": docs, "loss_tokens": loss_tokens,
            "skipped_special": skipped_special, "skipped_empty": skipped_empty,
            "response_tokens_p50": q(50), "response_tokens_p90": q(90),
            "response_tokens_max": max(resp_lens) if resp_lens else 0,
            "sha256": hb.hexdigest(), "mask_sha256": hm.hexdigest(),
            "input": path.name, "input_sha256": pt.sha256_file(path),
            "input_bytes": path.stat().st_size}


def prepare(files, split, tok_dir, out_dir, overwrite=False):
    files = [pt.check_input(f, split) for f in files]
    if not files:
        raise ValueError("tidak ada berkas")
    tok, tok_sha, ids, vocab = pt.load_tokenizer(tok_dir)
    names = [p.stem + ".bin" for p in files]
    if len(set(names)) != len(names):
        raise ValueError("nama berkas duplikat")
    mnames = {n: n[:-len(".bin")] + ".mask" for n in names}
    out = Path(out_dir)
    mp = out / f"meta_{split}.json"
    targets = [out / n for n in names] + [out / m for m in mnames.values()] + [mp]
    if not overwrite and any(t.exists() for t in targets):
        raise FileExistsError(f"sudah ada di {out}; pakai --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    shards, partials = {}, []
    try:
        for p, n in zip(files, names):
            tb, tm = out / (n + ".partial"), out / (mnames[n] + ".partial")
            partials += [tb, tm]
            info = _write_shard(p, tok, ids[pt.EOS], tb, tm, vocab)
            info["mask"] = mnames[n]
            shards[n] = info
        for n in names:
            os.replace(out / (n + ".partial"), out / n)
            os.replace(out / (mnames[n] + ".partial"), out / mnames[n])
    except BaseException:
        for t in partials:
            t.unlink(missing_ok=True)
        raise
    meta = {"script_version": __version__, "split": split, "dtype": "uint16",
            "sft": True, "mask_dtype": "uint8",
            "template": {"user_tag": fc.USER_TAG, "assistant_tag": fc.ASSISTANT_TAG},
            "tokenizer_json_sha256": tok_sha, "vocab_size": vocab, "eos_id": ids[pt.EOS],
            "shards": shards,
            "catatan": "Prototipe SFT; pasangan instruksi dari template, BUKAN data final."}
    mp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--split", required=True, choices=["train", "eval"])
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    try:
        meta = prepare(a.files, a.split, a.tokenizer_dir, a.out_dir, a.overwrite)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"GALAT: {e}", file=sys.stderr)
        return 2
    for n, s in meta["shards"].items():
        print(f"{n}: contoh={s['docs']} token={s['tokens']} token_loss={s['loss_tokens']} "
              f"lewati_khusus={s['skipped_special']} lewati_kosong={s['skipped_empty']} "
              f"respons_token p50/p90/maks={s['response_tokens_p50']}/"
              f"{s['response_tokens_p90']}/{s['response_tokens_max']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
