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
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from config import Config

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging with file and console handlers, plus optional Cloud Logging."""
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(Config.LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Google Cloud Logging (if enabled and available)
    if Config.ENABLE_CLOUD_LOGGING:
        try:
            import google.cloud.logging
            client = google.cloud.logging.Client()
            cloud_handler = client.get_default_handler()
            cloud_handler.setLevel(logging.INFO)
            logger.addHandler(cloud_handler)
            logger.info("✅ Google Cloud Logging enabled")
        except ImportError:
            logger.warning("⚠️  google-cloud-logging not installed. Cloud Logging disabled.")
        except Exception as e:
            logger.warning(f"⚠️  Could not setup Google Cloud Logging: {e}")
    
    return logger


def authenticate_gcs() -> bool:
    """
    Verify GCS access using VM default credentials.
    
    Uses google-cloud-storage Python library which automatically uses:
    - VM service account credentials (when running on GCP)
    - Application default credentials (local development)
    
    Returns:
        True if authentication successful
        
    Raises:
        RuntimeError: If GCS access verification fails
    """
    logger.info("🔐 Verifying GCS access using VM credentials...")
    
    try:
        from google.cloud import storage
        
        # Create storage client (automatically uses VM credentials)
        client = storage.Client()
        
        # Test access by listing bucket
        bucket = client.bucket(Config.BUCKET_NAME)
        
        # Try to list objects (this verifies permissions)
        blobs = list(bucket.list_blobs(max_results=1))
        
        logger.info(f"✅ GCS bucket access verified: {Config.BUCKET_NAME}")
        logger.info("   Using VM default credentials")
        
        if Config.ENABLE_CLOUD_LOGGING:
            try:
                import google.cloud.logging
                cloud_client = google.cloud.logging.Client()
                cloud_logger = cloud_client.logger("vis-pipeline")
                cloud_logger.log_text(
                    f"✅ GCS authentication successful for bucket: {Config.BUCKET_NAME}",
                    severity="INFO"
                )
            except Exception:
                pass
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ GCS bucket access FAILED: {error_msg}")
        logger.error("\n" + "=" * 70)
        logger.error("CRITICAL: Cannot access GCS bucket!")
        logger.error("=" * 70)
        logger.error(f"Bucket: {Config.BUCKET_NAME}")
        logger.error("Required IAM role: Storage Object Admin (or Storage Object Viewer)")
        logger.error("\nPossible causes:")
        logger.error("1. VM service account lacks required permissions")
        logger.error("2. Bucket name is incorrect")
        logger.error("3. Bucket does not exist")
        logger.error("\nTo fix:")
        logger.error("  gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \\")
        logger.error("    --member='serviceAccount:YOUR_VM_SERVICE_ACCOUNT' \\")
        logger.error("    --role='roles/storage.objectAdmin'")
        logger.error("=" * 70)
        
        raise RuntimeError(f"GCS authentication failed: {error_msg}")


def check_and_download_data():
    """
    Check if training data exists locally, download from GCS if not.
    Uses google-cloud-storage Python library with VM default credentials.
    """
    logger.info("📥 Checking training data...")
    
    # Check if data already exists
    if os.path.exists(Config.LOCAL_TRAIN_DIR) and os.path.exists(Config.LOCAL_JSON_PATH):
        num_videos = len([d for d in os.listdir(Config.LOCAL_TRAIN_DIR) 
                         if os.path.isdir(os.path.join(Config.LOCAL_TRAIN_DIR, d))])
        if num_videos > 0:
            logger.info(f"   ✅ Training data already exists: {num_videos} videos found")
            return
    
    logger.info("   Downloading data from GCS...")
    os.makedirs(Config.LOCAL_TRAIN_DIR, exist_ok=True)
    
    try:
        from google.cloud import storage
        
        client = storage.Client()
        bucket = client.bucket(Config.BUCKET_NAME)
        
        # Download annotation JSON
        logger.info(f"   Downloading annotations...")
        blob = bucket.blob("train.json")
        blob.download_to_filename(Config.LOCAL_JSON_PATH)
        logger.info("   ✅ Annotations downloaded")
        
        # Download training data
        logger.info(f"   Downloading training data (this may take a while)...")
        
        # List all blobs in trainxs/ directory
        blobs = bucket.list_blobs(prefix="trainxs/")
        
        downloaded_count = 0
        for blob in blobs:
            # Skip directory markers
            if blob.name.endswith('/'):
                continue
            
            # Create local path
            local_path = os.path.join(Config.LOCAL_BASE_DIR, blob.name)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download file
            blob.download_to_filename(local_path)
            downloaded_count += 1
            
            if downloaded_count % 100 == 0:
                logger.info(f"   Downloaded {downloaded_count} files...")
        
        # Verify download
        num_videos = len([d for d in os.listdir(Config.LOCAL_TRAIN_DIR) 
                         if os.path.isdir(os.path.join(Config.LOCAL_TRAIN_DIR, d))])
        
        logger.info(f"   ✅ Training data downloaded: {num_videos} videos, {downloaded_count} total files")
        
        if num_videos == 0:
            raise RuntimeError("No video folders found after download")
            
    except Exception as e:
        logger.error(f"❌ Data download failed: {e}")
        raise RuntimeError(f"Failed to download data from GCS: {e}")


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
        logger.info(f"   ✅ Loaded {num_images} annotated images")
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
