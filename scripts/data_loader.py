"""Data loader Aegis-Core (v0.2.0, prototipe).

Daftar shard HANYA dari meta_{split}.json; split selain train/eval ditolak.
x dan y selalu int64, kontigu, dengan y[t] = x[t+1].
get_batch_sft (data SFT dari prepare_sft.py): sama, tetapi y[t] = -1 bila target = prompt.
"""
import json
from pathlib import Path

import numpy as np
import torch

__version__ = "0.2.0"
SPLITS = ("train", "eval")


def load_split(data_dir, split, expected_tokenizer_sha256=None):
    if split not in SPLITS:
        raise ValueError(f"split tidak valid: {split!r}")
    d = Path(data_dir)
    mp = d / f"meta_{split}.json"
    if not mp.is_file():
        raise FileNotFoundError(f"tidak ada: {mp}")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("split") != split:
        raise ValueError(f"meta bukan split {split}")
    if expected_tokenizer_sha256 is not None and \
            meta.get("tokenizer_json_sha256") != expected_tokenizer_sha256:
        raise ValueError("hash tokenizer di meta tidak cocok dengan yang diharapkan")
    shards = []
    for name, info in sorted(meta["shards"].items()):
        if name != Path(name).name or not name.endswith(f"_{split}.bin"):
            raise ValueError(f"nama shard tidak valid untuk split {split}: {name}")
        p = d / name
        if p.stat().st_size != int(info["tokens"]) * 2:
            raise ValueError(f"ukuran shard tidak cocok dengan meta: {name}")
        if info["tokens"] > 0:
            shards.append(np.memmap(p, dtype="<u2", mode="r"))
    return shards, meta


def get_batch(data_dir, split, B, T, seed, step=0, expected_tokenizer_sha256=None):
    shards, _ = load_split(data_dir, split, expected_tokenizer_sha256)
    usable = [s for s in shards if len(s) > T]
    if not usable:
        raise ValueError(f"tidak ada shard dengan panjang > {T} pada split {split}")
    w = np.array([len(s) - T for s in usable], dtype=np.float64)
    rng = np.random.default_rng([int(seed), int(step)])
    which = rng.choice(len(usable), size=B, p=w / w.sum())
    xs, ys = [], []
    for k in which:
        s = usable[int(k)]
        i = int(rng.integers(0, len(s) - T))
        xs.append(np.asarray(s[i:i + T], dtype=np.int64))
        ys.append(np.asarray(s[i + 1:i + T + 1], dtype=np.int64))
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys))
    assert x.is_contiguous() and y.is_contiguous()
    return x, y


def load_split_sft(data_dir, split, expected_tokenizer_sha256=None):
    """Seperti load_split, plus mask uint8 paralel per shard (1 = target ikut loss)."""
    shards, meta = load_split(data_dir, split, expected_tokenizer_sha256)
    if meta.get("sft") is not True:
        raise ValueError("meta bukan data SFT (sft=true tidak ada); buat dengan prepare_sft.py")
    d = Path(data_dir)
    masks = []
    for name, info in sorted(meta["shards"].items()):
        if int(info["tokens"]) <= 0:
            continue  # sama dengan load_split: shard kosong dilewati, urutan tetap sejajar
        mname = info.get("mask")
        if not isinstance(mname, str) or mname != Path(mname).name or not mname.endswith(".mask"):
            raise ValueError(f"nama mask tidak valid untuk shard {name}")
        p = d / mname
        if p.stat().st_size != int(info["tokens"]):
            raise ValueError(f"ukuran mask tidak cocok dengan meta: {mname}")
        masks.append(np.memmap(p, dtype="u1", mode="r"))
    if len(masks) != len(shards):
        raise ValueError("jumlah mask tidak sama dengan jumlah shard")
    return shards, masks, meta


def get_batch_sft(data_dir, split, B, T, seed, step=0, expected_tokenizer_sha256=None,
                  max_tries=100):
    """Seperti get_batch, tetapi y[t] = -1 (ignore_index) bila token target adalah prompt.

    Mask diambil dengan indeks yang sama dengan y, jadi benar untuk jendela yang dimulai
    di mana pun (termasuk di tengah respons). Jendela tanpa satu pun target respons
    ditarik ulang secara deterministik (rng yang sama), agar tiap baris punya sinyal loss;
    loss semua-ignore akan NaN. Gagal jelas bila max_tries habis.
    """
    shards, masks, _ = load_split_sft(data_dir, split, expected_tokenizer_sha256)
    usable = [(s, m) for s, m in zip(shards, masks) if len(s) > T]
    if not usable:
        raise ValueError(f"tidak ada shard dengan panjang > {T} pada split {split}")
    w = np.array([len(s) - T for s, _ in usable], dtype=np.float64)
    p = w / w.sum()
    rng = np.random.default_rng([int(seed), int(step)])
    xs, ys = [], []
    for _ in range(B):
        for _try in range(max_tries):
            s, m = usable[int(rng.choice(len(usable), p=p))]
            i = int(rng.integers(0, len(s) - T))
            keep = np.asarray(m[i + 1:i + T + 1]) != 0
            if keep.any():
                break
        else:
            raise ValueError(f"tidak menemukan jendela dengan target respons "
                             f"dalam {max_tries} percobaan (split {split})")
        xs.append(np.asarray(s[i:i + T], dtype=np.int64))
        ys.append(np.where(keep, np.asarray(s[i + 1:i + T + 1], dtype=np.int64), -1))
    x = torch.from_numpy(np.stack(xs))
    y = torch.from_numpy(np.stack(ys).astype(np.int64))
    assert x.is_contiguous() and y.is_contiguous()
    return x, y
