#!/usr/bin/env python3
"""Uji memorisasi v2 (Teacher-Forced Loss).
Membandingkan loss pada data latih vs eval. Loss yang jauh lebih rendah di data 
latih mengindikasikan memorisasi.
"""
import argparse
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as gen

def sample_chunks(bin_file, num_chunks, seq_len, seed=42):
    rng = random.Random(seed)
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    chunks = []
    for _ in range(num_chunks):
        start = rng.randint(0, total_tokens - seq_len)
        chunks.append(data[start:start+seq_len].astype(np.int64).tolist())
    return chunks

def calculate_loss(model, chunks, seq_len):
    """Menghitung rata-rata teacher-forced cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for chunk in chunks:
            # Input adalah semua token kecuali yang terakhir
            x = torch.tensor([chunk[:-1]], dtype=torch.long).to(device)
            # Target adalah semua token kecuali yang pertama
            y = torch.tensor([chunk[1:]], dtype=torch.long).to(device)
            
            _, loss = model(x, y)
            total_loss += loss.item()
            
    return total_loss / len(chunks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer-dir", default="data/tokenizer_cmp/v16384")
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--eval-bin", required=True)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Memuat model {args.ckpt}...")
    model, tok, eos_id, step = gen.load_model(args.ckpt, args.tokenizer_dir)

    print(f"Mengambil {args.samples} sampel train...")
    train_chunks = sample_chunks(args.train_bin, args.samples, args.seq_len, args.seed)

    print(f"Mengambil {args.samples} sampel eval...")
    eval_chunks = sample_chunks(args.eval_bin, args.samples, args.seq_len, args.seed)

    print("\nMenghitung Teacher-Forced Loss...")
    tr_loss = calculate_loss(model, train_chunks, args.seq_len)
    ev_loss = calculate_loss(model, eval_chunks, args.seq_len)

    print("\n================ HASIL UJI MEMORISASI V2 ================")
    print(f"Total sampel per set: {args.samples} (Panjang seq: {args.seq_len})")
    print("-" * 54)
    print(f"Loss Train (Latih) : {tr_loss:.4f}")
    print(f"Loss Eval (Uji)    : {ev_loss:.4f}")
    print(f"Selisih (Eval - Train): {ev_loss - tr_loss:.4f}")
    print("-" * 54)
    print("CATATAN: Jika Train << Eval, model kemungkinan menghafal.")

if __name__ == "__main__":
    main()
