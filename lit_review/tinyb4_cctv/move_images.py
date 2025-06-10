'''
make it easier to use with pytorch imagefolder
creates 
'''
import os
import shutil

def move_images(source_root=r'D:\FF\crops\combined_701515', target_root=r'D:\FF\crops\combined_imagefolder_701515'):
    splits = ['train', 'val', 'test']
    classes = ['combined_original_701515', 'combined_deepfake_701515']

    for split in splits:
        for cls in classes:
            print(split, cls)
            source_dir = os.path.join(source_root, cls, split)
            target_dir = os.path.join(target_root, split, cls)

            os.makedirs(target_dir, exist_ok=True)

            for fname in os.listdir(source_dir):
                fpath = os.path.join(source_dir, fname)
                if os.path.isfile(fpath):
                    shutil.copy(fpath, os.path.join(target_dir, fname))

    print("Done")

# Run it
move_images()
