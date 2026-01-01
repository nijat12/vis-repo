"""
Utility functions for VIS pipeline execution.

Handles:
- Logging setup (local + Cloud Logging)
- GCS authentication and data download error handling
- VM shutdown logic
- Ground truth loading
- Metrics calculation
- Object tracking
"""

import os
import sys
import json
import logging
import subprocess
import zipfile
import shutil
import warnings
import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2
import psutil
from scipy.optimize import linear_sum_assignment


try:
    from sahi.predict import get_sliced_prediction
    from sahi import AutoDetectionModel
except ImportError:
    get_sliced_prediction = None
    AutoDetectionModel = None


# Suppress Torch/YOLO internal deprecation warnings
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="torch.cuda.amp.autocast"
)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")

from config import Config

logger = logging.getLogger(__name__)


def setup_logging(log_name: Optional[str] = None):
    """
    Configure logging with file and console handlers, plus optional Cloud Logging.
    This function is now safe to call multiple times (e.g., in multiprocessing).

    Args:
        log_name: Optional basename for the log file (e.g. 'baseline.log').
                 If None, uses Config.LOG_FILE.
    """
    root_logger = logging.getLogger()

    # Remove all existing handlers to ensure a clean setup in each process
    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()

    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    # Set GOOGLE_APPLICATION_CREDENTIALS if key file exists
    if Config.SERVICE_ACCOUNT_KEY and os.path.exists(Config.SERVICE_ACCOUNT_KEY):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
            Config.SERVICE_ACCOUNT_KEY
        )

    # Ensure logs directory exists
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # Determine log path
    filename = log_name if log_name else Config.LOG_FILE
    log_path = os.path.join(Config.LOG_DIR, filename)

    # File handler
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_formatter = logging.Formatter("%(name)45s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(name)45s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Google Cloud Logging (if enabled and available)
    if Config.ENABLE_CLOUD_LOGGING:
        try:
            import google.cloud.logging
            import atexit

            client = google.cloud.logging.Client()
            atexit.register(client.close)

            cloud_handler = client.get_default_handler(name=filename)
            cloud_handler.setLevel(logging.INFO)
            root_logger.addHandler(cloud_handler)
        except ImportError:
            root_logger.warning(
                "⚠️  google-cloud-logging not installed. Cloud Logging disabled."
            )
        except Exception as e:
            root_logger.warning(f"⚠️  Could not setup Google Cloud Logging: {e}")

    return root_logger


def setup_worker_logging(log_queue: Optional[Any] = None):
    """
    Configures logging for a worker process.
    If log_queue is provided, uses QueueHandler to send logs to main process.
    Otherwise, configures stdout logging.

    Args:
        log_queue: Multiprocessing Queue to send logs to.
    """
    root_logger = logging.getLogger()

    # Avoid adding multiple handlers if the worker is reused
    if root_logger.hasHandlers():
        return root_logger

    root_logger.setLevel(getattr(logging, Config.LOG_LEVEL))

    if log_queue:
        try:
            from logging.handlers import QueueHandler

            queue_handler = QueueHandler(log_queue)
            queue_handler.setLevel(logging.DEBUG)  # Send everything to main
            root_logger.addHandler(queue_handler)
        except Exception as e:
            print(f"Failed to setup QueueHandler: {e}")
    else:
        # Fallback to console logging if no queue is provided
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(name)45s - %(message)s")
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

    return root_logger


def authenticate_gcs(key_file: Optional[str] = None) -> bool:
    """
    Verify GCS access using provided key file or VM default credentials.
    """
    if key_file and os.path.exists(key_file):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(key_file)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(Config.BUCKET_NAME)

    # Test access by trying to list a single object
    list(bucket.list_blobs(max_results=1))
    return True


def authenticate_with_gcs_and_handle_errors():
    """Wrapper to handle GCS authentication and subsequent errors."""
    try:
        logger.info("🔐 Authenticating with Google Cloud Storage...")
        authenticate_gcs(key_file=Config.SERVICE_ACCOUNT_KEY)
        logger.info("✅ GCS authentication successful")
    except Exception as e:
        logger.critical(f"❌ GCS authentication FAILED: {e}")
        logger.critical("   Cannot proceed without GCS access. Aborting.")
        trigger_vm_shutdown_if_enabled(force=True)
        sys.exit(1)


def check_and_download_data_with_error_handling():
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
        num_videos = len(
            [
                d
                for d in os.listdir(Config.LOCAL_TRAIN_DIR)
                if os.path.isdir(os.path.join(Config.LOCAL_TRAIN_DIR, d))
            ]
        )

    if json_exists and dir_exists and num_videos > 0:
        logger.info(
            f"✅ Training data validated: {num_videos} videos found in {Config.LOCAL_TRAIN_DIR}"
        )
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
            raise RuntimeError(
                f"Zip file '{Config.GCS_TRAIN_ZIP}' not found in bucket '{Config.BUCKET_NAME}'"
            )

        blob_zip.download_to_filename(Config.LOCAL_ZIP_PATH)
        logger.info("✅ Zip file downloaded")

        # 3. Extract training ZIP
        logger.info(f"Extracting {Config.LOCAL_ZIP_PATH}...")
        with zipfile.ZipFile(Config.LOCAL_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(Config.LOCAL_BASE_DIR)
        logger.info("✅ Extraction complete")

        # 4. Clean up zip file to save space
        if os.path.exists(Config.LOCAL_ZIP_PATH):
            os.remove(Config.LOCAL_ZIP_PATH)
            logger.info("🧹 Cleaned up zip file")

        # Verify download
        dir_exists_now = os.path.exists(Config.LOCAL_TRAIN_DIR)
        if dir_exists_now:
            num_videos = len(
                [
                    d
                    for d in os.listdir(Config.LOCAL_TRAIN_DIR)
                    if os.path.isdir(os.path.join(Config.LOCAL_TRAIN_DIR, d))
                ]
            )
            logger.info(
                f"✅ Training data ready: {num_videos} videos found in {Config.LOCAL_TRAIN_DIR}"
            )
        else:
            logger.error(
                f"❌ Error: Expected data directory {Config.LOCAL_TRAIN_DIR} not found after extraction."
            )
            # List what was actually extracted to help debug zip structure
            actual_contents = os.listdir(Config.LOCAL_BASE_DIR)
            logger.error(
                f"Contents of {Config.LOCAL_BASE_DIR}: {actual_contents[:10]}..."
            )
            raise RuntimeError(
                f"Zip extraction did not create expected directory: {Config.LOCAL_TRAIN_DIR}"
            )

    except Exception as e:
        logger.critical(f"❌ Data download FAILED: {e}", exc_info=True)
        # Clean up partial downloads
        if os.path.exists(Config.LOCAL_ZIP_PATH):
            os.remove(Config.LOCAL_ZIP_PATH)
        trigger_vm_shutdown_if_enabled(force=True)
        sys.exit(1)


def trigger_vm_shutdown_if_enabled(force: bool = False):
    """
    Initiates VM shutdown if the killswitch is enabled.

    Args:
        force (bool): If True, shutdown even if the killswitch is disabled in config.
                      Used for critical errors.
    """
    if force or Config.get_runtime_killswitch():
        delay = 10 if force else Config.KILLSWITCH_DELAY_SECONDS
        level = logging.CRITICAL if force else logging.INFO

        logger.log(
            level, f"🔴 Killswitch triggered. VM will shutdown in {delay} seconds."
        )

        try:
            from vm_utils import shutdown_vm

            shutdown_vm(delay_seconds=delay)
        except ImportError:
            logger.error("Could not import vm_utils. Shutdown will not occur.")
        except Exception as e:
            logger.error(f"Failed to trigger shutdown: {e}")
    else:
        logger.info("💡 Killswitch disabled. VM will remain running.")
        logger.info("   Remember to stop the VM manually to avoid charges!")


def load_json_ground_truth(json_path: str) -> Dict:
    """
    Load ground truth annotations from COCO-style JSON file.
    Returns a dictionary mapping 'video_name/frame_name' to a list of bounding boxes [x, y, w, h].
    """
    logger.info(f"📖 Loading ground truth from {json_path}")

    if not os.path.exists(json_path):
        logger.error(f"❌ Ground truth file not found: {json_path}")
        return {}

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        # Index images by ID
        images = {img["id"]: img for img in data.get("images", [])}

        # Index annotations by image ID
        processed_data = {}
        for ann in data.get("annotations", []):
            img_id = ann.get("image_id")
            if img_id not in images:
                continue

            file_name = images[img_id].get("file_name")
            if file_name not in processed_data:
                processed_data[file_name] = []

            # Bbox is already in [x, y, w, h] format in COCO
            processed_data[file_name].append(ann.get("bbox"))

        logger.info(f"✅ Loaded ground truth for {len(processed_data)} images")
        return processed_data

    except Exception as e:
        logger.error(f"❌ Failed to load ground truth: {e}")
        return {}


def calculate_center_distance(box1: List[float], box2: List[float]) -> float:
    """Calculate Euclidean distance between box centers (xywh format)."""
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    cx1 = x1 + w1 / 2
    cy1 = y1 + h1 / 2
    cx2 = x2 + w2 / 2
    cy2 = y2 + h2 / 2

    return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def calculate_avg_dotd(matches: List[float]) -> float:
    """Calculates the average Dot Distance from a list of distances."""
    if not matches:
        return 0.0
    return float(np.mean(matches))


def calculate_video_map(
    detections: List[List[Any]],
    ground_truths: List[List[List[float]]],
    iou_thresh: float = 0.5,
) -> float:
    """
    Calculates mAP for a single video/class (assumes single class 'bird').

    Args:
        detections: List of detections for each frame. Each detection is [x, y, w, h, score].
        ground_truths: List of ground truths for each frame. Each GT is [x, y, w, h].
        iou_thresh: IoU threshold for a positive match.

    Returns:
        Average Precision (AP) for this video.
    """
    all_detections = []  # List of (score, frame_idx, box)
    total_gt = 0

    for frame_idx, (frame_dets, frame_gts) in enumerate(zip(detections, ground_truths)):
        total_gt += len(frame_gts)
        for det in frame_dets:
            # Check if detection has score
            if len(det) >= 5:
                score = det[4]
                box = det[:4]
                all_detections.append((score, frame_idx, box))
            else:
                # Fallback if no score provided (should not happen after refactor)
                all_detections.append((0.0, frame_idx, det[:4]))

    if total_gt == 0:
        return 0.0 if not all_detections else 0.0

    # Sort by confidence descending
    all_detections.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(all_detections))
    fp = np.zeros(len(all_detections))

    # Keep track of matched GTs to avoid double counting: set((frame_idx, gt_idx))
    matched_gts = set()

    for i, (score, frame_idx, pred_box) in enumerate(all_detections):
        gts = ground_truths[frame_idx]
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gts):
            iou = box_iou_xywh(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh:
            if (frame_idx, best_gt_idx) not in matched_gts:
                tp[i] = 1
                matched_gts.add((frame_idx, best_gt_idx))
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Compute precision and recall
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / total_gt
    precisions = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-6)

    # Compute AP (Area Under Curve) using 11-point interpolation or VOC style
    # Here using standard integration
    ap = 0.0
    # Concatenate 0 and 1 endpoints
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute projection of max precision
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return float(ap)


def get_memory_usage() -> float:
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def box_iou_xywh(box1: List[float], box2: List[float]) -> float:
    """Calculate IoU between two boxes in xywh format."""
    x1, y1, w1, h1 = box1[:4]
    x2, y2, w2, h2 = box2[:4]

    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    inter_x_min = max(x1, x2)
    inter_y_min = max(y1, y2)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def nms_xywh(
    boxes: List[List[float]], scores: List[float], iou_thr: float = 0.5
) -> List[int]:
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


def get_sahi_predictions(
    model, image_bgr: np.ndarray, config: Dict[str, Any]
) -> List[List[float]]:
    """
    Performs sliced inference using the SAHI library.

    Args:
        model: The loaded YOLO model object.
        image_bgr: The input image in BGR format.
        config: A dictionary containing SAHI and model parameters.

    Returns:
        A list of detections in [x, y, w, h] format.
    """
    if get_sliced_prediction is None:
        logger.error("❌ SAHI library not found. Please run: pip install sahi")
        return []

    # SAHI expects images in RGB format
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    detection_model = model
    # Auto-wrap Ultralytics YOLO model if it hasn't been wrapped yet
    if AutoDetectionModel is not None:
        # Check if it has perform_inference (SAHI specific)
        if not hasattr(model, "perform_inference"):
            # Use a cached wrapper if we made one previously to avoid re-init overhead
            if not hasattr(model, "_sahi_wrapper"):
                try:
                    # 'yolov8' model_type works for Ultralytics YOLOv8/v11/v12
                    model._sahi_wrapper = AutoDetectionModel.from_pretrained(
                        model_type="yolov8",
                        model=model,
                        confidence_threshold=config.get("conf_thresh", 0.25),
                        device="cpu",  # Assume CPU for this environment
                    )
                except Exception as e:
                    logger.error(f"❌ Failed to wrap YOLO model for SAHI: {e}")
                    return []
            detection_model = model._sahi_wrapper

    result = get_sliced_prediction(
        image_rgb,
        detection_model,
        slice_height=config.get("sahi_slice_height", 640),
        slice_width=config.get("sahi_slice_width", 640),
        overlap_height_ratio=config.get("sahi_overlap_height_ratio", 0.2),
        overlap_width_ratio=config.get("sahi_overlap_width_ratio", 0.2),
        postprocess_type="NMS",
        postprocess_match_metric="IOU",
        postprocess_match_threshold=config.get("iou_thresh", 0.45),
        verbose=0,
    )

    final_preds = []
    for pred in result.object_prediction_list:
        if pred.category.id in [config["bird_class_id"]]:
            box = pred.bbox.to_xywh()
            score = pred.score.value
            final_preds.append(box + [score])

    return final_preds


def linear_interpolate_box(
    box1: List[float], box2: List[float], t: float
) -> List[float]:
    """Linearly interpolate between two bounding boxes [x, y, w, h]."""
    return [b1 * (1 - t) + b2 * t for b1, b2 in zip(box1[:4], box2[:4])]


def generate_interpolated_boxes(
    last_preds: List[List[float]],
    current_preds: List[List[float]],
    last_frame_idx: int,
    current_frame_idx: int,
    config: Dict[str, Any],
) -> Dict[int, List[List[float]]]:
    """
    Generates interpolated bounding boxes for frames between two keyframes.
    """
    interpolated_results = defaultdict(list)
    if not last_preds or not current_preds or last_frame_idx >= current_frame_idx:
        return interpolated_results

    # Create a cost matrix based on center distance
    cost_matrix = np.full((len(last_preds), len(current_preds)), float("inf"))
    for r, box1 in enumerate(last_preds):
        for c, box2 in enumerate(current_preds):
            dist = calculate_center_distance(box1, box2)
            cost_matrix[r, c] = dist

    # Match boxes using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    max_dist = config.get("interpolation_max_dist", 200)
    frame_interval = current_frame_idx - last_frame_idx

    for r, c in zip(row_ind, col_ind):
        # Only interpolate if the match is within the distance threshold
        if cost_matrix[r, c] < max_dist:
            start_box = last_preds[r]
            end_box = current_preds[c]

            # Generate boxes for all intermediate frames
            for j in range(1, frame_interval):
                inter_frame_idx = last_frame_idx + j
                t = j / float(frame_interval)
                inter_box = linear_interpolate_box(start_box, end_box, t)
                interpolated_results[inter_frame_idx].append(inter_box)

    return interpolated_results


class ObjectTracker:
    """Simple object tracker using center distance."""

    def __init__(
        self, dist_thresh: float = 50, max_frames_to_skip: int = 5, min_hits: int = 3
    ):
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 0

    def update(self, detections: List[List[float]]) -> List[List[float]]:
        """Update tracker with new detections."""
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(
                    {
                        "id": self.next_id,
                        "box": det[:4],
                        "score": det[4] if len(det) > 4 else 1.0,
                        "hits": 1,
                        "skipped": 0,
                    }
                )
                self.next_id += 1
        else:
            matched = set()
            for track in self.tracks:
                best_dist, best_idx = float("inf"), -1
                for i, det in enumerate(detections):
                    if i in matched:
                        continue
                    dist = calculate_center_distance(track["box"], det[:4])
                    if dist < best_dist and dist < self.dist_thresh:
                        best_dist, best_idx = dist, i

                if best_idx != -1:
                    track.update(
                        {
                            "box": detections[best_idx][:4],
                            "score": (
                                detections[best_idx][4]
                                if len(detections[best_idx]) > 4
                                else track.get("score", 1.0)
                            ),
                            "hits": track["hits"] + 1,
                            "skipped": 0,
                        }
                    )
                    matched.add(best_idx)
                else:
                    track["skipped"] += 1

            for i, det in enumerate(detections):
                if i not in matched:
                    self.tracks.append(
                        {
                            "id": self.next_id,
                            "box": det[:4],
                            "score": det[4] if len(det) > 4 else 1.0,
                            "hits": 1,
                            "skipped": 0,
                        }
                    )
                    self.next_id += 1

        self.tracks = [
            t for t in self.tracks if t["skipped"] <= self.max_frames_to_skip
        ]
        return [
            t["box"] + ([t["score"]] if "score" in t else [])
            for t in self.tracks
            if t["hits"] >= self.min_hits
        ]


def log_video_metrics(logger: logging.Logger, video_name: str, metrics: Dict[str, Any]):
    """Logs metrics for a single video in a column-based list."""
    logger.info("-" * 40)
    logger.info(f"📊 VIDEO RESULTS: {video_name}")
    logger.info("-" * 40)

    display_mapping = [
        ("n_frames", "Frames"),
        ("fps", "FPS"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("tp", "TP"),
        ("fp", "FP"),
        ("fn", "FN"),
        ("iou", "IoU"),
        ("mAP", "mAP"),
        ("iou", "IoU"),
        ("mAP", "mAP"),
        ("dotd", "DotD"),
        ("memory_usage_mb", "Memory (MB)"),
        ("vid_time", "Time"),
    ]

    for key, label in display_mapping:
        val = metrics.get(key)
        if val is not None:
            if "time" in key.lower():
                time_str = str(datetime.timedelta(seconds=int(val)))
                logger.info(f"{label:<20}: {time_str} ({val:.1f}s)")
            elif isinstance(val, (float, np.floating)):
                logger.info(f"{label:<20}: {val:.4f}")
            else:
                logger.info(f"{label:<20}: {val}")
    logger.info("-" * 40 + "\n")


def log_pipeline_summary(
    logger: logging.Logger, pipeline_name: str, metrics: Dict[str, Any]
):
    """Logs the final summary of the pipeline in a column-based list."""
    logger.info("=" * 50)
    logger.info(f"🚀 FINAL SUMMARY: {pipeline_name.upper()}")
    logger.info("=" * 50)

    for key, val in metrics.items():
        if key in ["total_frames", "processing_time_sec"]:
            continue

        label = key.replace("_", " ").title()
        if "time" in key.lower():
            time_str = str(datetime.timedelta(seconds=int(val)))
            logger.info(f"{label:<25}: {time_str} ({val:.1f}s)")
        elif isinstance(val, (float, np.floating)):
            logger.info(f"{label:<25}: {val:.4f}")
        else:
            logger.info(f"{label:<25}: {val}")

    # Log computed averages (Step 5.3)
    total_frames = metrics.get("total_frames", 0)
    proc_time = metrics.get("processing_time_sec", 0)

    if total_frames > 0:
        avg_time_per_img = proc_time / total_frames
        logger.info(f"{'Avg Time Per Image':<25}: {avg_time_per_img:.4f}s")

    # Assume 1 second per video is too trivial, but good to have
    # We don't have num_videos passed directly here usually, but if execution_time is total wall clock
    # we can just rely on the specific metrics passed.

    logger.info("=" * 50 + "\n")
