'''
combine deepfake types' train/val/test into a combined train/val/test in "combined_deepfakes" directory.
crops folder was created from base fyp preprocessing code
'''

import os
import shutil


base_dir = r'D:\FF\crops'
deepfake_dirs = ['Deepfakes', 'Face2Face', 'FaceSwap', 'FaceShifter', 'NeuralTextures']
splits = ['train', 'val', 'test']
output_base = os.path.join(base_dir, 'combined_deepfakes')


for split in splits:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

for dfolder in deepfake_dirs:
    for split in splits:
        input_dir = os.path.join(base_dir, dfolder, split, 'manipulated_sequences')
        output_dir = os.path.join(output_base, split)

        if not os.path.exists(input_dir):
            continue

        for fname in os.listdir(input_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                src = os.path.join(input_dir, fname)
                # prevent filename collision by prefixing with folder name
                dst_fname = f"{dfolder}_{fname}"
                dst = os.path.join(output_dir, dst_fname)
                shutil.copy2(src, dst)
