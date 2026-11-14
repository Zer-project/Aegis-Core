"""Tes dedup_against_eval. Data sintetis di tmp_path; tanpa data nyata, tanpa jaringan."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parent / "scripts"):
    sys.path.insert(0, str(p))
import dedup_against_eval as dd  # noqa: E402


def write_jsonl(path, texts, prefix):
    with open(path, "wb") as f:
        for i, t in enumerate(texts):
            row = json.dumps({"id": f"{prefix}{i}", "text": t}, ensure_ascii=False)
            f.write((row + "\n").encode("utf-8"))


def run(monkeypatch, train, evals, out):
    argv = ["dedup_against_eval.py", "--train-files", *map(str, train),
            "--eval-files", *map(str, evals), "--out-dir", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    return dd.main()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_removes_exact_duplicate_and_copies_rest_byte_identical(tmp_path, monkeypatch):
    tr, ev, out = tmp_path / "en_train.jsonl", tmp_path / "en_eval.jsonl", tmp_path / "out"
    write_jsonl(tr, ["alpha", "beta é", "gamma"], "t")
    write_jsonl(ev, ["beta é", "zzz"], "e")  # id berbeda, teks sama
    lines = tr.read_bytes().splitlines(keepends=True)
    assert run(monkeypatch, [tr], [ev], out) == 0
    assert (out / "en_train.jsonl").read_bytes() == lines[0] + lines[2]


def test_input_files_not_modified(tmp_path, monkeypatch):
    tr, ev, out = tmp_path / "a_train.jsonl", tmp_path / "a_eval.jsonl", tmp_path / "out"
    write_jsonl(tr, ["alpha", "beta"], "t")
    write_jsonl(ev, ["beta"], "e")
    before = (sha(tr), sha(ev))
    assert run(monkeypatch, [tr], [ev], out) == 0
    assert (sha(tr), sha(ev)) == before


def test_near_duplicate_is_not_removed(tmp_path, monkeypatch):
    tr, ev, out = tmp_path / "a_train.jsonl", tmp_path / "a_eval.jsonl", tmp_path / "out"
    write_jsonl(tr, ["beta ", "beta"], "t")
    write_jsonl(ev, ["beta"], "e")
    assert run(monkeypatch, [tr], [ev], out) == 0
    kept = [json.loads(l)["text"] for l in (out / "a_train.jsonl").read_bytes().splitlines()]
    assert kept == ["beta "]


def test_refuses_existing_out_dir(tmp_path, monkeypatch):
    tr, ev, out = tmp_path / "a_train.jsonl", tmp_path / "a_eval.jsonl", tmp_path / "out"
    write_jsonl(tr, ["alpha"], "t")
    write_jsonl(ev, ["beta"], "e")
    out.mkdir()
    assert run(monkeypatch, [tr], [ev], out) == 1
    assert list(out.iterdir()) == []


def test_invalid_json_in_eval_fails_before_creating_out_dir(tmp_path, monkeypatch):
    tr, ev, out = tmp_path / "a_train.jsonl", tmp_path / "a_eval.jsonl", tmp_path / "out"
    write_jsonl(tr, ["alpha"], "t")
    ev.write_bytes(b"not json\n")
    assert run(monkeypatch, [tr], [ev], out) == 1
    assert not out.exists()


def test_invalid_json_in_train_fails(tmp_path, monkeypatch):
    tr, ev, out = tmp_path / "a_train.jsonl", tmp_path / "a_eval.jsonl", tmp_path / "out"
    tr.write_bytes(b"not json\n")
    write_jsonl(ev, ["beta"], "e")
    assert run(monkeypatch, [tr], [ev], out) == 1


def test_duplicate_basenames_rejected(tmp_path, monkeypatch):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    t1, t2 = tmp_path / "x" / "en_train.jsonl", tmp_path / "y" / "en_train.jsonl"
    ev, out = tmp_path / "e.jsonl", tmp_path / "out"
    write_jsonl(t1, ["a"], "t")
    write_jsonl(t2, ["b"], "t")
    write_jsonl(ev, ["c"], "e")
    assert run(monkeypatch, [t1, t2], [ev], out) == 1
    assert not out.exists()


def test_prints_counts_only_no_text_or_ids(tmp_path, monkeypatch, capsys):
    tr, ev, out = tmp_path / "a_train.jsonl", tmp_path / "a_eval.jsonl", tmp_path / "out"
    write_jsonl(tr, ["SECRET_TEXT_XYZ", "other"], "t")
    write_jsonl(ev, ["SECRET_TEXT_XYZ"], "e")
    assert run(monkeypatch, [tr], [ev], out) == 0
    printed = capsys.readouterr().out
    assert "kept=1 removed=1" in printed
    assert "SECRET_TEXT_XYZ" not in printed and "t0" not in printed and "e0" not in printed
