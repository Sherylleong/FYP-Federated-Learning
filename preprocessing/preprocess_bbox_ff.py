import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Suppress all warnings

# from tqdm import tqdm_notebook
# from google.colab.patches import cv2_imshow
from IPython.display import HTML #imports to play videos
from base64 import b64encode 
#from skimage.measure import compare_ssim
import glob
import time
from PIL import Image
from tqdm import tqdm
from functools import partial
from PIL import Image
from multiprocessing import Pool
import cv2
import skimage.measure
#import albumentations as A
#from albumentations.pytorch import ToTensor

from facenet_pytorch import MTCNN



import cv2
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from torchvision import models, transforms

import matplotlib.pyplot as plt

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


device = 'cuda'

SOURCE_FOLDER= r"D:\FF"

DEEPFAKE_TYPE = 'FaceSwap'
ORI_VIDEOS_FOLDER = fr'original_sequences\youtube\c23\videos'
MANIP_VIDEOS_FOLDER = fr'manipulated_sequences\{DEEPFAKE_TYPE}\c23\videos'
ORI_BOXES_FOLDER = fr'original_sequences'
MANIP_BOXES_FOLDER = fr'manipulated_sequences\{DEEPFAKE_TYPE}'


DATA_FOLDER = os.path.join(SOURCE_FOLDER, ORI_VIDEOS_FOLDER)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10


frames_per_video = 32

img_size = 380
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)



def extract_frames_from_video(path, n_frames=32):
    # Create video reader and find length
    v_cap = cv2.VideoCapture(path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pick 'n_frames' evenly spaced frames to sample
    if n_frames is None:
        sample = np.arange(0, v_len)
    else:
        sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    # Loop through frames
    frames = []
    for j in range(v_len):
        success = v_cap.grab()
        if j in sample:
            # Load frame
            success, frame = v_cap.retrieve()
            if not success:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    return frames


def get_faces_from_video(video_path):
    detector = MTCNN(margin=300, select_largest=False, factor=0.5, device=device, post_process=False) # post_process=False if want human readable image
    faces = []
    # Movie to Image
    try:
        frames = extract_frames_from_video(video_path)
    except:
        frames = []
    if len(frames) == 0:
        return []
    # Detect Faces
    _frame = np.array(frames)
    boxes, probs = detector.detect(_frame, landmarks=False)

    return [b.tolist()[0] if b is not None else None for b in boxes]
import json

def save_face_bbox(boxes, out_dir, video_name):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "{}.json".format(video_name)), "w") as f:
        json.dump(boxes, f) # raw bounding box data for each video for frame


        

def get_video_names_from_folder(folder_path):
    # List all files in the specified folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if 'metadata.json' in files:
        files.remove('metadata.json')
    return files

import concurrent.futures


def preprocess_bbox(source_folder, to_folder, video_name):
    try:
        video_path = os.path.join(SOURCE_FOLDER, source_folder, video_name)
        video_save_path = os.path.join(SOURCE_FOLDER, to_folder)
        os.makedirs(video_save_path, exist_ok=True)
        faces = get_faces_from_video(video_path)
        save_face_bbox(faces, video_save_path, video_name[:-4])
    except Exception as e: 
        print(e)
import json
def get_train_val_test_splits(SOURCE_FOLDER):
    # Assuming the JSON data is stored in a file called 'data.json'
    with open(os.path.join(SOURCE_FOLDER, 'train.json'), 'r') as file:
        data = json.load(file)
        train_set_files = [item for sublist in data for item in sublist]
    with open(os.path.join(SOURCE_FOLDER, 'val.json'), 'r') as file:
        data = json.load(file)
        val_set_files = [item for sublist in data for item in sublist]
    with open(os.path.join(SOURCE_FOLDER, 'test.json'), 'r') as file:
        data = json.load(file)
        test_set_files = [item for sublist in data for item in sublist]
    return train_set_files, val_set_files, test_set_files
if __name__ == '__main__':
    print(get_train_val_test_splits(SOURCE_FOLDER))
    

    source_folder = ORI_VIDEOS_FOLDER
    to_folder = ORI_BOXES_FOLDER

    out_dir = os.path.join(SOURCE_FOLDER, 'extracted_boxes', to_folder)

    os.makedirs(out_dir, exist_ok=True)
    video_names = get_video_names_from_folder(os.path.join(SOURCE_FOLDER, source_folder))

    for video_name in tqdm(video_names):
        preprocess_bbox(source_folder, out_dir, video_name)


