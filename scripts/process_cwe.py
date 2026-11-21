import xml.etree.ElementTree as ET
import json
import glob

input_files = glob.glob("data/cyber_dataset/*.xml")
if not input_files:
    print("ERROR: Tidak ada file .xml di data/cyber_dataset/")
    exit(1)

input_file = input_files[0]
output_file = "data/cyber_dataset/cwe_train.jsonl"

print(f"Membaca {input_file}...")
tree = ET.parse(input_file)
root = tree.getroot()

# Menggunakan wildcard {*} agar tidak peduli versi namespace CWE
weaknesses = root.findall('.//{*}Weakness')
processed_count = 0

with open(output_file, 'w', encoding='utf-8') as f_out:
    for w in weaknesses:
        cwe_id = w.get('ID')
        name = w.get('Name')
        
        desc_elem = w.find('{*}Description')
        desc = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""
        
        ext_desc_elem = w.find('{*}Extended_Description')
        ext_desc = ""
        if ext_desc_elem is not None:
            ext_desc = " ".join([t.strip() for t in ext_desc_elem.itertext() if t.strip()])

        text = f"CWE-{cwe_id}: {name}\n\nDescription: {desc}\n\nExtended Description: {ext_desc}"
        
        if len(text) < 100:
            continue

        f_out.write(json.dumps({"text": text}) + "\n")
        processed_count += 1

print(f"Selesai! Berhasil memproses {processed_count} kelemahan (CWE).")
