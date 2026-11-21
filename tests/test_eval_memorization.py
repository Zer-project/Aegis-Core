import pytest
import numpy as np
import sys
from pathlib import Path

# Tambahkan folder scripts ke sys.path agar bisa di-import
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_memorization import sample_chunks

def test_sample_chunks(tmp_path):
    # Buat file .bin palsu dengan 1000 token
    dummy_bin = tmp_path / "dummy.bin"
    data = np.arange(1000, dtype=np.uint16)
    data.tofile(dummy_bin)
    
    # Uji pengambilan sampel
    chunks = sample_chunks(dummy_bin, num_chunks=5, prefix_len=32, suffix_len=32, seed=42)
    
    assert len(chunks) == 5
    for chunk in chunks:
        assert len(chunk) == 64
        assert isinstance(chunk, list)
        
    # Pastikan seed yang sama menghasilkan hasil yang sama (deterministik)
    chunks2 = sample_chunks(dummy_bin, num_chunks=5, prefix_len=32, suffix_len=32, seed=42)
    assert chunks == chunks2
