#!/usr/bin/env python3
"""Uji memorisasi (G1): Membandingkan tingkat kecocokan teks latih vs eval.

Skrip ini murni membaca (read-only) file .bin dan model. Ia mengambil awalan
32 token acak, meminta model menebak 32 token berikutnya, dan membandingkannya.
"""
import argparse
import sys
import random
import numpy as np
import torch
from pathlib import Path

# Impor dari proyek Aegis-Core
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate as gen

def sample_chunks(bin_file, num_chunks, prefix_len, suffix_len, seed=42):
    """Mengambil chunk acak (prefix + suffix) dari file .bin."""
    rng = random.Random(seed)
    data = np.memmap(bin_file, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    chunk_size = prefix_len + suffix_len
    
    chunks = []
    # Pastikan kita tidak mengambil chunk yang melebihi batas file
    for _ in range(num_chunks):
        start = rng.randint(0, total_tokens - chunk_size)
        chunks.append(data[start:start+chunk_size].astype(np.int64).tolist())
    return chunks

def evaluate_memorization(model, tok, eos_id, chunks, prefix_len, suffix_len):
    """Menghitung berapa banyak suffix yang ditebak dengan benar secara greedy."""
    exact_match = 0
    partial_match_16 = 0
    total_tokens_matched = 0

    model.eval()
    for chunk in chunks:
        prefix = chunk[:prefix_len]
        target_suffix = chunk[prefix_len:]
        
        # Siapkan tensor input
        input_ids = torch.tensor([prefix], dtype=torch.long)
        
        # Lakukan generasi greedy (temperature=0.0)
        with torch.no_grad():
            # Kita asumsikan model.generate mendukung temperature=0 untuk greedy
            # (Berdasarkan kode generate.sample di proyek Anda)
            output_ids = model.generate(input_ids, max_new_tokens=suffix_len, temperature=0.0)[0].tolist()
        
        # Ambil hasil tebakan (buang prefix)
        predicted_suffix = output_ids[prefix_len:]
        
        # Hitung kecocokan
        matches = sum(1 for p, t in zip(predicted_suffix, target_suffix) if p == t)
        total_tokens_matched += matches
        
        if predicted_suffix == target_suffix:
            exact_match += 1
        
        # Cek kecocokan parsial minimal 16 token berurutan
        max_consecutive = 0
        current_consecutive = 0
        for p, t in zip(predicted_suffix, target_suffix):
            if p == t:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        if max_consecutive >= 16:
            partial_match_16 += 1

    return exact_match, partial_match_16, total_tokens_matched

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path ke file checkpoint (.pt)")
    ap.add_argument("--tokenizer-dir", default="data/tokenizer_cmp/v16384")
    ap.add_argument("--train-bin", required=True, help="Path ke file .bin data latih")
    ap.add_argument("--eval-bin", required=True, help="Path ke file .bin data eval")
    ap.add_argument("--samples", type=int, default=100, help="Jumlah dokumen yang diuji per set")
    ap.add_argument("--prefix-len", type=int, default=32)
    ap.add_argument("--suffix-len", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Memuat model {args.ckpt}...")
    try:
        model, tok, eos_id, step = gen.load_model(args.ckpt, args.tokenizer_dir)
    except Exception as e:
        print(f"ERROR memuat model: {e}")
        return 1

    print(f"Mengambil sampel data dari {args.train_bin}...")
    train_chunks = sample_chunks(args.train_bin, args.samples, args.prefix_len, args.suffix_len, args.seed)
    
    print(f"Mengambil sampel data dari {args.eval_bin}...")
    eval_chunks = sample_chunks(args.eval_bin, args.samples, args.prefix_len, args.suffix_len, args.seed)

    print("\n--- EVALUASI DATA LATIH (TRAIN) ---")
    tr_exact, tr_part16, tr_tokens = evaluate_memorization(model, tok, eos_id, train_chunks, args.prefix_len, args.suffix_len)
    
    print("\n--- EVALUASI DATA UJI (EVAL) ---")
    ev_exact, ev_part16, ev_tokens = evaluate_memorization(model, tok, eos_id, eval_chunks, args.prefix_len, args.suffix_len)

    print("\n================ HASIL UJI MEMORISASI ================")
    print(f"Total sampel per set: {args.samples}")
    print(f"Prefix len: {args.prefix_len}, Suffix len: {args.suffix_len}")
    print("-" * 54)
    print(f"{'Metrik':<30} | {'Train':<10} | {'Eval':<10}")
    print("-" * 54)
    print(f"{'Exact Match (100%)':<30} | {tr_exact:<10} | {ev_exact:<10}")
    print(f"{'Partial Match >=16 token':<30} | {tr_part16:<10} | {ev_part16:<10}")
    print(f"{'Total token cocok':<30} | {tr_tokens:<10} | {ev_tokens:<10}")
    print("======================================================")
    print("CATATAN: Hasil negatif (Train = Eval) BUKAN jaminan privasi.")
    print("Hasil positif (Train >> Eval) adalah bukti kuat adanya memorisasi.")

if __name__ == "__main__":
    main()
