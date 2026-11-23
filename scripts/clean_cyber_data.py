import json
import re
import random
from pathlib import Path

# Pengaturan
INPUT_MITRE = Path("data/cyber_dataset/mitre_attack.json")
INPUT_CWE = Path("data/cyber_dataset/cwe_train.jsonl")
OUT_TRAIN = Path("data/cyber_dataset/aegis_sec_v2_train.jsonl")
OUT_EVAL = Path("data/cyber_dataset/aegis_sec_v2_eval.jsonl")
SEED = 42

def clean_text(text):
    """Menghapus penanda sitasi seperti (Citation: ...) dari teks."""
    # Regex untuk mencari teks dalam kurung yang diawali (Citation:
    cleaned = re.sub(r'\(Citation:[^)]+\)', '', text)
    return cleaned.strip()

def process_mitre():
    docs = []
    if not INPUT_MITRE.exists():
        print(f"File {INPUT_MITRE} tidak ditemukan.")
        return docs
        
    with open(INPUT_MITRE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    valid_types = {
        "attack-pattern", "intrusion-set", "malware", "tool", "course-of-action", "campaign"
    }
    
    for obj in data.get("objects", []):
        if obj.get("type") not in valid_types:
            continue
        # Filter data yang sudah dicabut atau usang
        if obj.get("revoked") is True or obj.get("x_mitre_deprecated") is True:
            continue
            
        name = obj.get("name", "")
        desc = obj.get("description", "")
        if not desc:
            continue
            
        text = clean_text(f"Title: {name}\nDescription: {desc}")
        if len(text) > 100:
            docs.append({"text": text})
            
    return docs

def process_cwe():
    docs = []
    if not INPUT_CWE.exists():
        print(f"File {INPUT_CWE} tidak ditemukan.")
        return docs
        
    with open(INPUT_CWE, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            # Bersihkan juga data CWE
            obj["text"] = clean_text(obj["text"])
            docs.append(obj)
    return docs

def main():
    print("Memproses data MITRE ATT&CK...")
    mitre_docs = process_mitre()
    print(f"  -> {len(mitre_docs)} dokumen bersih (tanpa revoked/deprecated).")
    
    print("Memproses data CWE...")
    cwe_docs = process_cwe()
    print(f"  -> {len(cwe_docs)} dokumen bersih.")
    
    all_docs = mitre_docs + cwe_docs
    print(f"Total dataset gabungan: {len(all_docs)} dokumen.")
    
    # Shuffle dan Split 90/10
    random.seed(SEED)
    random.shuffle(all_docs)
    
    split_idx = int(len(all_docs) * 0.9)
    train_docs = all_docs[:split_idx]
    eval_docs = all_docs[split_idx:]
    
    # Tulis ke JSONL
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for d in train_docs:
            f.write(json.dumps(d) + "\n")
            
    with open(OUT_EVAL, "w", encoding="utf-8") as f:
        for d in eval_docs:
            f.write(json.dumps(d) + "\n")
            
    print(f"Berhasil menyimpan {len(train_docs)} data latih ke {OUT_TRAIN}")
    print(f"Berhasil menyimpan {len(eval_docs)} data uji ke {OUT_EVAL}")

if __name__ == "__main__":
    main()
