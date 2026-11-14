#!/usr/bin/env python3
"""Aegis-Core: remove training documents whose text is byte-identical to an eval document.

v0.1.0. Exact match only (SHA-256 of UTF-8 text). Near-duplicates are NOT detected.
Kept lines are copied byte-for-byte. Inputs are never modified.
Refuses to write into an existing output directory. Prints counts only,
never document text or ids.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

VERSION = "0.1.0"


def text_hash(line: bytes) -> str:
    text = json.loads(line).get("text", "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--eval-files", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print(f"ERROR: {out_dir} already exists; refusing to write.", file=sys.stderr)
        return 1
    names = [Path(p).name for p in args.train_files]
    if len(set(names)) != len(names):
        print("ERROR: train files must have distinct base names.", file=sys.stderr)
        return 1
    if any(Path(p).resolve() == (out_dir / Path(p).name).resolve() for p in args.train_files):
        print("ERROR: output would overwrite an input.", file=sys.stderr)
        return 1

    eval_hashes = set()
    for p in args.eval_files:
        with open(p, "rb") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    try:
                        eval_hashes.add(text_hash(line))
                    except json.JSONDecodeError:
                        print(f"ERROR: invalid JSON at {p} line {n}.", file=sys.stderr)
                        return 1
    print(f"dedup_against_eval v{VERSION} | eval hashes: {len(eval_hashes)}")

    out_dir.mkdir(parents=True, exist_ok=False)
    for p in args.train_files:
        kept = removed = 0
        with open(p, "rb") as fi, open(out_dir / Path(p).name, "xb") as fo:
            for n, line in enumerate(fi, 1):
                if not line.strip():
                    continue
                try:
                    dup = text_hash(line) in eval_hashes
                except json.JSONDecodeError:
                    print(f"ERROR: invalid JSON at {p} line {n}.", file=sys.stderr)
                    return 1
                if dup:
                    removed += 1
                else:
                    fo.write(line)
                    kept += 1
        print(f"{Path(p).name}: kept={kept} removed={removed}")
    print("Exact-text matches only; near-duplicates were not checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
