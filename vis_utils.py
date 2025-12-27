"""
Utility functions for VIS pipeline execution.

Handles:
- Logging setup (local + Cloud Logging)
- GCS authentication using VM default credentials
- Data download from GCS
- Ground truth loading
- Metrics calculation
- Object tracking
"""

import os
import json
import logging
import subprocess
import zipfile
import shutil
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

# Suppress Torch/YOLO internal deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")

from config import Config

logger = logging.getLogger(__name__)


def setup_logging(log_name: Optional[str] = None):
    """
    Configure logging with file and console handlers, plus optional Cloud Logging.
    
    Args:
        log_name: Optional basename for the log file (e.g. 'baseline.log').
                 If None, uses Config.LOG_FILE.
    """
    logger = logging.getLogger()
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Set GOOGLE_APPLICATION_CREDENTIALS if key file exists
    if Config.SERVICE_ACCOUNT_KEY and os.path.exists(Config.SERVICE_ACCOUNT_KEY):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(Config.SERVICE_ACCOUNT_KEY)
    
    # Ensure logs directory exists
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Determine log path
    filename = log_name if log_name else Config.LOG_FILE
    log_path = os.path.join(Config.LOG_DIR, filename)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(name)20s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    

    
    # Google Cloud Logging (if enabled and available)
    if Config.ENABLE_CLOUD_LOGGING:
        try:
            import google.cloud.logging
            client = google.cloud.logging.Client()
            cloud_handler = client.get_default_handler()
            cloud_handler.setLevel(logging.INFO)
            logger.addHandler(cloud_handler)
            logger.info(f"✅ Google Cloud Logging enabled (Log: {filename})")
        except ImportError:
            logger.warning("⚠️  google-cloud-logging not installed. Cloud Logging disabled.")
        except Exception as e:
            logger.warning(f"⚠️  Could not setup Google Cloud Logging: {e}")
    
    return logger


def authenticate_gcs(key_file: Optional[str] = None) -> bool:
    """
    Verify GCS access using provided key file or VM default credentials.
    
    Args:
        key_file: Path to service account JSON key file (optional)
        
    Returns:
        True if authentication successful
        
    Raises:
        RuntimeError: If GCS access verification fails
    """
    if key_file:
        logger.info(f"🔐 Verifying GCS access using key file: {key_file}...")
    else:
        logger.info("🔐 Verifying GCS access using VM default credentials...")
    
    try:
        from google.cloud import storage
        
        # Ensure environment variable is set
        if key_file and os.path.exists(key_file):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_file)
            logger.info(f"✅ Using service account key: {key_file}")
        
        client = storage.Client()
        if not (key_file and os.path.exists(key_file)):
            logger.info("✅ Using VM default credentials")
        
        # Test access by listing bucket
        bucket = client.bucket(Config.BUCKET_NAME)
        
        # Try to list objects (this verifies permissions)
        list(bucket.list_blobs(max_results=1))
        
        logger.info(f"✅ GCS bucket access verified: {Config.BUCKET_NAME}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ GCS bucket access FAILED: {error_msg}")
        logger.error("CRITICAL: Cannot access GCS bucket!")
        logger.error(f"Bucket: {Config.BUCKET_NAME}")
        logger.error("Required IAM role: Storage Object Admin (or Storage Object Viewer)")
        logger.error("Possible causes:")
        logger.error("1. VM service account lacks required permissions")
        logger.error("2. Bucket name is incorrect")
        logger.error("3. Bucket does not exist")
        logger.error("To fix:")
        logger.error("  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\")
        logger.error("    --member='serviceAccount:YOUR_VM_SERVICE_ACCOUNT' \\")
        logger.error("    --role='roles/storage.objectAdmin'")
        
        raise RuntimeError(f"GCS authentication failed: {error_msg}")


def check_and_download_data():
    """
    Check if training data exists locally, download from GCS and unzip if not.
    Uses google-cloud-storage Python library with VM default credentials.
    """
    logger.info("📥 Checking training data...")
    
    # Check if JSON exists
    json_exists = os.path.exists(Config.LOCAL_JSON_PATH)
    # Check if train dir exists
    dir_exists = os.path.exists(Config.LOCAL_TRAIN_DIR)
    
    num_videos = 0
    if dir_exists:
        num_videos = len([d for d in os.listdir(Config.LOCAL_TRAIN_DIR) 
                         if os.path.isdir(os.path.join(Config.LOCAL_TRAIN_DIR, d))])
    
    if json_exists and dir_exists and num_videos > 0:
        logger.info(f"✅ Training data validated: {num_videos} videos found in {Config.LOCAL_TRAIN_DIR}")
        return
    
    # Log reason for download
    if not json_exists:
        logger.info(f"Missing annotations: {Config.LOCAL_JSON_PATH}")
    if not dir_exists:
        logger.info(f"Missing data directory: {Config.LOCAL_TRAIN_DIR}")
    elif num_videos == 0:
        logger.info(f"Data directory is empty: {Config.LOCAL_TRAIN_DIR}")
    
    logger.info("🚀 Triggering download from GCS...")
    
    logger.info("Downloading data from GCS...")
    os.makedirs(Config.LOCAL_BASE_DIR, exist_ok=True)
    
    try:
        from google.cloud import storage
        
        client = storage.Client()
        bucket = client.bucket(Config.BUCKET_NAME)
        
        # 1. Download annotation JSON
        logger.info("Downloading annotations...")
        blob_json = bucket.blob(Config.GCS_JSON_URL)
        blob_json.download_to_filename(Config.LOCAL_JSON_PATH)
        logger.info("✅ Annotations downloaded")
        
        # 2. Download training ZIP
        logger.info(f"Downloading training ZIP: {Config.GCS_TRAIN_ZIP}")
        blob_zip = bucket.blob(Config.GCS_TRAIN_ZIP)
        
        # Check if zip exists in GCS
        if not blob_zip.exists():
            raise RuntimeError(f"Zip file '{Config.GCS_TRAIN_ZIP}' not found in bucket '{Config.BUCKET_NAME}'")
            
        blob_zip.download_to_filename(Config.LOCAL_ZIP_PATH)
        logger.info("✅ Zip file downloaded")
        
        # 3. Extract training ZIP
        logger.info(f"Extracting {Config.LOCAL_ZIP_PATH}...")
        with zipfile.ZipFile(Config.LOCAL_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(Config.LOCAL_BASE_DIR)
        logger.info("✅ Extraction complete")
        
        # 4. Clean up zip file to save space
        if os.path.exists(Config.LOCAL_ZIP_PATH):
            os.remove(Config.LOCAL_ZIP_PATH)
            logger.info("🧹 Cleaned up zip file")
            
        # Verify download
        dir_exists_now = os.path.exists(Config.LOCAL_TRAIN_DIR)
        if dir_exists_now:
            num_videos = len([d for d in os.listdir(Config.LOCAL_TRAIN_DIR) 
                             if os.path.isdir(os.path.join(Config.LOCAL_TRAIN_DIR, d))])
            logger.info(f"✅ Training data ready: {num_videos} videos found in {Config.LOCAL_TRAIN_DIR}")
        else:
            logger.error(f"❌ Error: Expected data directory {Config.LOCAL_TRAIN_DIR} not found after extraction.")
            # List what was actually extracted to help debug zip structure
            actual_contents = os.listdir(Config.LOCAL_BASE_DIR)
            logger.error(f"Contents of {Config.LOCAL_BASE_DIR}: {actual_contents[:10]}...")
            raise RuntimeError(f"Zip extraction did not create expected directory: {Config.LOCAL_TRAIN_DIR}")
            
    except Exception as e:
        logger.error(f"❌ Data download/extraction failed: {e}")
        # Clean up partial downloads
        if os.path.exists(Config.LOCAL_ZIP_PATH):
            os.remove(Config.LOCAL_ZIP_PATH)
        raise RuntimeError(f"Failed to process data from GCS: {e}")


def load_json_ground_truth(json_path: str) -> Dict:
    """Load ground truth annotations from JSON file."""
    logger.info(f"📖 Loading ground truth from {json_path}")
    
    if not os.path.exists(json_path):
        logger.error(f"❌ Ground truth file not found: {json_path}")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        num_images = len(data)
        logger.info(f"✅ Loaded {num_images} annotated images")
        return data
        
    except Exception as e:
        logger.error(f"❌ Failed to load ground truth: {e}")
        return {}


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """Calculate Euclidean distance between box centers (xywh format)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    cx1 = x1 + w1 / 2
    cy1 = y1 + h1 / 2
    cx2 = x2 + w2 / 2
    cy2 = y2 + h2 / 2
    
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def box_iou_xywh(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in xywh format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to xyxy
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def nms_xywh(boxes: List[List[float]], scores: List[float], iou_thr: float = 0.5) -> List[int]:
    """Non-maximum suppression for boxes in xywh format."""
    if len(boxes) == 0:
        return []
    
    order = np.argsort(scores)[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        rest = order[1:]
        new_rest = []
        for j in rest:
            if box_iou_xywh(boxes[i], boxes[j]) < iou_thr:
                new_rest.append(j)
        order = np.array(new_rest, dtype=np.int64)
    
    return keep


def align_frames(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
    """Align previous frame to current frame using ECC."""
    try:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)
        
        _, warp_matrix = cv2.findTransformECC(
            prev_gray, curr_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria
        )
        
        h, w = curr_gray.shape
        aligned = cv2.warpAffine(prev_gray, warp_matrix, (w, h), flags=cv2.INTER_LINEAR)
        
        return aligned
    except Exception:
        return None


def get_next_version_path(base_path: str) -> str:
    """Get next available version of a file path."""
    if not os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    version = 1
    
    while True:
        new_path = f"{base}_v{version}{ext}"
        if not os.path.exists(new_path):
            return new_path
        version += 1


class ObjectTracker:
    """Simple object tracker using center distance."""
    
    def __init__(self, dist_thresh: float = 50, max_frames_to_skip: int = 5, min_hits: int = 3):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0
    
    def update(self, detections: List[List[float]]) -> List[List[float]]:
        """Update tracker with new detections."""
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append({
                    'id': self.next_id,
                    'box': det,
                    'hits': 1,
                    'skipped': 0
                })
                self.next_id += 1
        else:
            # Match detections to tracks
            matched = set()
            for track in self.tracks:
                best_dist = float('inf')
                best_idx = -1
                
                for i, det in enumerate(detections):
                    if i in matched:
                        continue
                    dist = calculate_center_distance(track['box'], det)
                    if dist < best_dist and dist < self.dist_thresh:
                        best_dist = dist
                        best_idx = i
                
                if best_idx >= 0:
                    track['box'] = detections[best_idx]
                    track['hits'] += 1
                    track['skipped'] = 0
                    matched.add(best_idx)
                else:
                    track['skipped'] += 1
            
            # Add new tracks
            for i, det in enumerate(detections):
                if i not in matched:
                    self.tracks.append({
                        'id': self.next_id,
                        'box': det,
                        'hits': 1,
                        'skipped': 0
                    })
                    self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t['skipped'] <= self.max_frames_to_skip]
        
        # Return confirmed tracks
        return [t['box'] for t in self.tracks if t['hits'] >= self.min_hits]
