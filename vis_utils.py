import os
import json
import subprocess
import cv2
import re
from collections import defaultdict
from tqdm import tqdm

# Constants
KEY_FILE = 'colab-upload-bot-key.json'
BUCKET_NAME = 'vis-data-2025'
GCS_TRAIN_DIR = f'gs://{BUCKET_NAME}/trainsm'
GCS_JSON_URL = f'gs://{BUCKET_NAME}/train.json'
LOCAL_BASE_DIR = './data_local'
LOCAL_TRAIN_DIR = os.path.join(LOCAL_BASE_DIR, 'trainsm')
LOCAL_JSON_PATH = './train.json'

def check_and_download_data():
    """Checks for data and downloads if missing."""
    print("\n🚀 CHECKING DATA...")

    # A. Check Annotations
    if not os.path.exists(LOCAL_JSON_PATH):
        print(f"⬇️ 'train.json' not found. Downloading...")
        if os.path.exists(KEY_FILE):
            os.system(f'gcloud auth activate-service-account --key-file="{KEY_FILE}"')
        os.system(f'gsutil cp {GCS_JSON_URL} {LOCAL_JSON_PATH}')
    else:
        print("✅ Annotations found locally.")

    # B. Check Video Data
    if os.path.exists(LOCAL_TRAIN_DIR) and len(os.listdir(LOCAL_TRAIN_DIR)) > 0:
        print(f"✅ Training data found in '{LOCAL_TRAIN_DIR}'. Skipping download.")
    else:
        print(f"⬇️ Data not found. Downloading from GCS...")
        if os.path.exists(KEY_FILE):
            os.system(f'gcloud auth activate-service-account --key-file="{KEY_FILE}"')
        
        os.makedirs(LOCAL_BASE_DIR, exist_ok=True)
        # Download logic
        parent_dir = os.path.dirname(LOCAL_TRAIN_DIR)
        cmd = f'gsutil -m cp -r {GCS_TRAIN_DIR} {parent_dir}'
        print(f"Running: {cmd}")
        os.system(cmd)

def load_json_ground_truth(json_path):
    """Loads Ground Truth annotations from JSON."""
    if not os.path.exists(json_path):
        print(f"❌ Error: Annotation file not found at {json_path}")
        return {}

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ JSON Parse Error: {e}")
        return {}

    id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    img_id_to_boxes = defaultdict(list)
    if 'annotations' in data:
        for ann in data['annotations']:
            img_id_to_boxes[ann['image_id']].append(ann['bbox'])

    filename_to_gt = {}
    for img_id, filename in id_to_filename.items():
        key = filename
        if key.startswith('train/'):
            key = key.replace('train/', '', 1)
        filename_to_gt[key] = img_id_to_boxes.get(img_id, [])

    return filename_to_gt

def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) between two boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def get_next_version_path(path):
    """
    Returns a new file path with an incremented version number if the file already exists.
    """
    if not os.path.exists(path):
        return path

    directory, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    pattern = re.compile(rf"^{re.escape(name)}_(\d+){re.escape(ext)}$")
    max_version = 0
    
    # Check existing files
    if os.path.exists(directory if directory else '.'):
        for f in os.listdir(directory if directory else '.'):
            match = pattern.match(f)
            if match:
                version = int(match.group(1))
                if version > max_version:
                    max_version = version

    new_filename = f"{name}_{max_version + 1}{ext}"
    return os.path.join(directory, new_filename)
