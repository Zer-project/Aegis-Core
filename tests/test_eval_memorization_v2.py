import pytest
import torch
import sys
import random
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_memorization_v2 import calculate_loss
from model import AegisConfig, AegisNano

def test_positive_control_memorization():
    """Memastikan skrip bisa mendeteksi model yang benar-benar menghafal."""
    vocab_size = 100
    seq_len = 32
    cfg = AegisConfig(vocab_size=vocab_size, block_size=seq_len, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = AegisNano(cfg)

    # Buat 1 sekuens "hafalan"
    memorized_seq = [10, 11, 12, 13] * 8  # 32 token
    memorized_chunk = [memorized_seq]

    # Buat 1 sekuens "acak" (tidak dihafal)
    random_chunk = [[random.randint(0, 99) for _ in range(seq_len)]]

    # 1. Paksa model menghafal dengan melatihnya sebentar pada memorized_chunk
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(50):  # 50 langkah sudah cukup untuk model sekecil ini menghafal 1 baris
        x = torch.tensor([memorized_seq[:-1]], dtype=torch.long)
        y = torch.tensor([memorized_seq[1:]], dtype=torch.long)
        optimizer.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()

    # 2. Uji apakah fungsi calculate_loss mendeteksi perbedaan
    loss_memorized = calculate_loss(model, memorized_chunk, seq_len)
    loss_random = calculate_loss(model, random_chunk, seq_len)

    # Karena model dilatih khusus pada memorized_chunk, loss-nya harus jauh lebih kecil
    assert loss_memorized < loss_random, f"Gagal! Loss hafalan ({loss_memorized}) tidak lebih kecil dari loss acak ({loss_random})"
    assert loss_memorized < 1.0, "Model mainan gagal menghafal (loss tidak turun)"
