import os
import shutil
import random
from pathlib import Path

def split_images(source_dir, dest_dir, seed=42):
    random.seed(seed)

    # Get all image files
    image_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    print(f"\nProcessing {source_dir} ({len(image_files)} images found)")

    if not image_files:
        print("⚠️ No images found — check your source path.")
        return

    random.shuffle(image_files)
    total = len(image_files)
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    splits = {
        'train': image_files[:train_end],
        'val': image_files[train_end:val_end],
        'test': image_files[val_end:]
    }

    for split_name, file_list in splits.items():
        split_path = Path(dest_dir) / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        print(f"📂 Copying {len(file_list)} images to {split_path}")
        for file_name in file_list:
            src_file = Path(source_dir) / file_name
            dst_file = split_path / file_name
            try:
                shutil.copy2(src_file, dst_file)
                print(f"{file_name}")
            except Exception as e:
                print(f"Error copying {file_name}: {e}")

# Paths
original_src = r"D:\FF\crops_no_split\original_sequences"
manipulated_src = r"D:\FF\crops_no_split\manipulated_sequences"

original_dst = r"D:\FF\crops\combined_original_701515"
manipulated_dst = r"D:\FF\crops\combined_deepfake_701515"

# Split and copy images
split_images(original_src, original_dst)
split_images(manipulated_src, manipulated_dst)

print("Image splitting and copying complete.")
