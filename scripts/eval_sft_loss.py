"""Loss SFT (hanya token respons+EOS) sebuah checkpoint pada data SFT (v0.1.0, prototipe).

Untuk baseline yang adil antar checkpoint pada data SFT yang SAMA. Deterministik:
jendela tidak tumpang-tindih (stride = block_size) pada tiap shard; sisa ekor shard yang
lebih pendek dari satu jendela tidak dihitung; jendela tanpa target respons dilewati.
Loss di-agregasi per token (bukan per jendela). Hanya mencetak angka, bukan teks.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import data_loader as dl  # noqa: E402

__version__ = "0.1.0"


def window_nll(model, shards, masks, T, B=8):
    """Rata-rata NLL per token target (y != -1) pada jendela tak-tumpang-tindih."""
    if T < 1 or B < 1:
        raise ValueError("T dan B harus >= 1")
    total, n_tok, n_win = 0.0, 0, 0
    xs, ys = [], []

    def flush():
        nonlocal total, n_tok, n_win, xs, ys
        if not xs:
            return
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys).astype(np.int64))
        cnt = int((y != -1).sum())
        with torch.no_grad():
            _, loss = model(x, y)
        total += float(loss) * cnt
        n_tok += cnt
        n_win += len(xs)
        xs, ys = [], []

    for s, m in zip(shards, masks):
        for i in range(0, len(s) - T, T):
            keep = np.asarray(m[i + 1:i + T + 1]) != 0
            if not keep.any():
                continue
            xs.append(np.asarray(s[i:i + T], dtype=np.int64))
            ys.append(np.where(keep, np.asarray(s[i + 1:i + T + 1], dtype=np.int64), -1))
            if len(xs) == B:
                flush()
    flush()
    if n_tok == 0:
        raise ValueError("tidak ada token target respons yang dievaluasi")
    nll = total / n_tok
    return {"nll": nll, "ppl": math.exp(nll), "target_tokens": n_tok, "windows": n_win}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--data-dir", required=True, help="keluaran prepare_sft.py")
    ap.add_argument("--split", choices=dl.SPLITS, default="eval")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", help="tulis hasil JSON ke berkas ini (menolak menimpa)")
    a = ap.parse_args(argv)
    from generate import load_model, sha256_file  # impor lambat: torch+tokenizers
    model, _tok, _eos, step = load_model(a.ckpt, a.tokenizer_dir)
    tok_sha = sha256_file(Path(a.tokenizer_dir) / "tokenizer.json")
    shards, masks, _ = dl.load_split_sft(a.data_dir, a.split, tok_sha)
    T = model.cfg.block_size
    res = window_nll(model, shards, masks, T, a.batch_size)
    res.update({"ckpt": str(a.ckpt), "ckpt_sha256": sha256_file(a.ckpt), "ckpt_step": step,
                "split": a.split, "block_size": T, "script_version": __version__})
    text = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "x", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return res


if __name__ == "__main__":
    main()
