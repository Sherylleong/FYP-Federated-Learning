import os
import json
from pathlib import Path

import re

def extract_id(filename):
    match = re.match(r"^(\d{3})", filename)
    return match.group(1) if match else None

base_dir = Path(r"D:\FF\crops/original_sequences")
output_base = Path(r"D:\FF\crops/combined_original")
os.makedirs(output_base, exist_ok=True)

# Ensure subfolders exist
for split in ["train", "val", "test"]:
    (output_base / split).mkdir(parents=True, exist_ok=True)

# Load json files as set of ids
def load_ids(filename):
    with open(filename, "r") as f:
        data = json.load(f)
        return set(id for sublist in data for id in sublist)

id_map = {
    "train": load_ids(r"D:\FF\train.json"),
    "val": load_ids(r"D:\FF\val.json"),
    "test": load_ids(r"D:\FF\test.json"),
}

# for all files in original_sequences
for img_file in base_dir.iterdir():

    if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue

    filename = img_file.name
    prefix = extract_id(filename)


    # determine which split the file belongs to
    for split, ids in id_map.items():
        if prefix in ids:
            dest = output_base / split / filename
            dest.write_bytes(img_file.read_bytes()) 
            break
