import json
import re
from pathlib import Path

# Kata kunci INTI JUDI (sesuai kesepakatan F1 dengan Claude)
CORE_GAMBLING = re.compile(r"\b(slot|judi|togel|gacor|maxwin|situs slot|bandar togel)\b", re.IGNORECASE)

def is_clean(text):
    # Buang teks yang terlalu pendek
    if len(text.split()) < 50:
        return False
    # Buang jika ada kata inti judi
    if CORE_GAMBLING.search(text):
        return False
    return True

def process_file(input_path, output_path):
    kept, dropped = 0, 0
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                if is_clean(data.get('text', '')):
                    f_out.write(line)
                    kept += 1
                else:
                    dropped += 1
            except:
                continue
    print(f"{input_path} -> Disimpan: {kept} | Dibuang (Judi/Pendek): {dropped}")

if __name__ == "__main__":
    print("=== MEMULAI FILTER F1 (ANTI-JUDI) ===")
    process_file("data/sample_9x/en_train.jsonl", "data/sample_9x_clean/en_train.jsonl")
    process_file("data/sample_9x/id_train.jsonl", "data/sample_9x_clean/id_train.jsonl")
    # Eval tidak perlu difilter karena kita ingin perbandingan adil dengan v4/B1
    print("=== SELESAI ===")
