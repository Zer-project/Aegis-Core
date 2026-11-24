"""Tes opsi train.py v0.2.0 (anggaran waktu, amp, mirror). Data sintetis; hanya CPU."""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "scripts", HERE):
    sys.path.insert(0, str(p))
import train as tr  # noqa: E402
from test_train import PERIODIC, make_data, targs  # noqa: E402


@pytest.fixture
def data(tmp_path):
    make_data(tmp_path / "data", PERIODIC)
    return tmp_path


def test_anggaran_waktu_berhenti_rapi_lalu_resume_identik(data):
    ra = tr.run(targs(data, "--out-dir", str(data / "run_a"), "--max-steps", "30"))
    rb1 = tr.run(targs(data, "--out-dir", str(data / "run_b"), "--max-steps", "30",
                       "--time-budget-hours", "1e-9"))
    assert rb1["budget_stop"] is True and rb1["end_step"] == 1
    assert (data / "run_b" / "ckpt.pt").is_file()
    rb2 = tr.run(targs(data, "--out-dir", str(data / "run_b"), "--max-steps", "30", "--resume"))
    assert rb2["start_step"] == 1 and rb2["budget_stop"] is False
    assert rb1["losses"] + rb2["losses"] == ra["losses"]
    ca = torch.load(data / "run_a" / "ckpt.pt", weights_only=True)
    cb = torch.load(data / "run_b" / "ckpt.pt", weights_only=True)
    assert ca["step"] == cb["step"] == 30
    assert all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])


def test_checkpoint_format_lama_tanpa_kunci_amp_bisa_resume(data):
    tr.run(targs(data, "--max-steps", "20", "--stop-after", "5"))
    p = data / "run" / "ckpt.pt"
    c = torch.load(p, weights_only=True)
    c.pop("amp")
    c.pop("scaler")
    torch.save(c, p)
    r = tr.run(targs(data, "--max-steps", "20", "--resume"))
    assert r["start_step"] == 5 and r["end_step"] == 20


def test_checkpoint_memuat_kunci_amp_default_off(data):
    tr.run(targs(data, "--max-steps", "3"))
    c = torch.load(data / "run" / "ckpt.pt", weights_only=True)
    assert c["amp"] == "off" and c["scaler"] is None
    assert c["script_version"] == tr.__version__


def test_amp_fp16_di_cpu_ditolak(data):
    with pytest.raises(ValueError, match="cuda"):
        tr.run(targs(data, "--amp", "fp16"))


def test_cuda_tidak_tersedia_ditolak(data, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA tidak tersedia"):
        tr.run(targs(data, "--device", "cuda"))


def test_anggaran_waktu_nonpositif_ditolak(data):
    with pytest.raises(ValueError, match="time-budget"):
        tr.run(targs(data, "--time-budget-hours", "0"))


def test_resume_menolak_mode_amp_berbeda(data):
    tr.run(targs(data, "--max-steps", "20", "--stop-after", "5"))
    p = data / "run" / "ckpt.pt"
    c = torch.load(p, weights_only=True)
    c["amp"] = "fp16"
    torch.save(c, p)
    with pytest.raises(ValueError, match="amp"):
        tr.run(targs(data, "--max-steps", "20", "--resume"))


def test_copy_ckpt_to_menyalin_ckpt_dan_log(data):
    dest = data / "mirror"
    tr.run(targs(data, "--max-steps", "4", "--copy-ckpt-to", str(dest)))
    assert (dest / "ckpt.pt").read_bytes() == (data / "run" / "ckpt.pt").read_bytes()
    assert (dest / "log.jsonl").read_bytes() == (data / "run" / "log.jsonl").read_bytes()


def test_copy_ckpt_to_gagal_hanya_peringatan(data, capsys):
    blocker = data / "berkas"
    blocker.write_text("x")
    r = tr.run(targs(data, "--max-steps", "4", "--copy-ckpt-to", str(blocker / "sub")))
    assert r["end_step"] == 4
    assert "PERINGATAN" in capsys.readouterr().err


def test_resume_memuat_dengan_map_location_cpu(data, monkeypatch):
    tr.run(targs(data, "--max-steps", "20", "--stop-after", "5"))
    seen, orig = [], torch.load

    def spy(*a, **k):
        seen.append(k.get("map_location"))
        return orig(*a, **k)

    monkeypatch.setattr(torch, "load", spy)
    tr.run(targs(data, "--max-steps", "20", "--resume"))
    assert seen == ["cpu"]
