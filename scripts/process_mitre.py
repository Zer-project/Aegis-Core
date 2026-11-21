import json
import os

input_file = "data/cyber_dataset/mitre_attack.json"
output_file = "data/cyber_dataset/mitre_attack_processed.jsonl"

print(f"Membaca {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

objects = data.get("objects", [])
valid_types = {
    "attack-pattern",    # Teknik serangan (TTPs)
    "intrusion-set",     # Aktor ancaman (APT)
    "malware",           # Perangkat lunak berbahaya
    "tool",              # Alat yang digunakan penyerang
    "course-of-action",  # Mitigasi/Pertahanan
    "campaign"           # Kampanye serangan
}

processed_count = 0
with open(output_file, 'w', encoding='utf-8') as f_out:
    for obj in objects:
        if obj.get("type") in valid_types:
            name = obj.get("name", "")
            desc = obj.get("description", "")
            
            # Abaikan jika tidak ada deskripsi
            if not desc:
                continue
                
            # Gabungkan nama dan deskripsi menjadi satu teks paragraf
            text = f"Title: {name}\nDescription: {desc}"
            
            # Tulis dalam format JSONL (sesuai dengan format dataset Aegis)
            f_out.write(json.dumps({"text": text}) + "\n")
            processed_count += 1

print(f"Selesai! Berhasil memproses {processed_count} dokumen siber.")
print(f"File tersimpan di: {output_file}")

# Cek ukuran file hasil
size_mb = os.path.getsize(output_file) / (1024 * 1024)
print(f"Ukuran file: {size_mb:.2f} MB")
