"""
Strategy 9 Pipeline: SAHI Slicing + YOLO + Kalman/Hungarian (DotD)

Implements the architecture defined in the class diagram:
1. Slicing: 3840x2160 -> 640x640 slices (with overlap)
2. Detection: YOLO (pretrained) on slices
3. Merger: NMS to combine slice detections
4. Tracking: Kalman Filter + Hungarian Algorithm (DotD cost)
"""

import os
import glob
import time
import datetime
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
from scipy.optimize import linear_sum_assignment

# Attempt to import ultralytics for YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

logger = logging.getLogger(__name__)

# =========================================================================
#  Helper: Kalman Filter + Hungarian Tracker (The "Strategy 9" Logic)
# =========================================================================


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    Uses a Constant Velocity model standard in multi-object tracking.
    """

    count = 0

    def __init__(self, bbox):
        # Initialize state vector [u, v, s, r, u_dot, v_dot, s_dot]
        try:
            from filterpy.kalman import KalmanFilter
        except ImportError:
            raise ImportError("filterpy library missing. Please pip install filterpy")

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State Transition Matrix F
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement Function H
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        # Covariance Matrices (tuned for small objects)
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.score = bbox[4] if len(bbox) > 4 else 0.0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        if len(bbox) > 4:
            self.score = bbox[4]

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score == None:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
            ).reshape((1, 4))
        else:
            return np.array(
                [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
            ).reshape((1, 5))


class SortDotDTracker:
    """
    SORT Tracker modified to use 'DotD' (Distance based cost) for the Hungarian Algorithm.
    Recommended for small objects where IoU is unreliable.
    """

    def __init__(self, max_age=15, min_hits=3, dist_threshold=100.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0

    def get_dotd_cost_matrix(self, trackers, detections):
        """
        Compute cost matrix using Euclidean distance between centers.
        """
        if len(trackers) == 0 or len(detections) == 0:
            return np.zeros((len(trackers), len(detections)))

        cost_matrix = np.zeros((len(trackers), len(detections)))

        for t, trk in enumerate(trackers):
            # Tracker state: [x1, y1, x2, y2]
            t_box = trk.get_state()[0]
            t_cx = (t_box[0] + t_box[2]) / 2
            t_cy = (t_box[1] + t_box[3]) / 2

            for d, det in enumerate(detections):
                # Detection: [x1, y1, x2, y2]
                d_cx = (det[0] + det[2]) / 2
                d_cy = (det[1] + det[3]) / 2

                # Euclidean distance
                dist = np.sqrt((t_cx - d_cx) ** 2 + (t_cy - d_cy) ** 2)
                cost_matrix[t, d] = dist

        return cost_matrix

    def update(self, dets):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        """
        self.frame_count += 1

        # 1. Predict existing tracks using Kalman Filter
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. Association (Hungarian Algorithm with DotD cost)
        matched, unmatched_dets, unmatched_trks = [], [], []

        if len(dets) > 0 and len(self.trackers) > 0:
            cost_matrix = self.get_dotd_cost_matrix(self.trackers, dets)

            # Hungarian Algorithm (linear_sum_assignment)
            row_inds, col_inds = linear_sum_assignment(cost_matrix)

            # Filter by threshold
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] > self.dist_threshold:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    matched.append([r, c])

            # Handle unmatched
            for t in range(len(self.trackers)):
                if t not in row_inds:
                    unmatched_trks.append(t)
            for d in range(len(dets)):
                if d not in col_inds:
                    unmatched_dets.append(d)
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(self.trackers)))

        # 3. Update matched trackers
        for t, d in matched:
            self.trackers[t].update(dets[d, :])

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 5. Output management
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # Standard SORT logic: return if hit streak is good or age is young enough
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # Return format: [x, y, w, h, id, score]
                ret.append(
                    np.concatenate((d, [trk.id + 1], [trk.score])).reshape(1, -1)
                )
            i -= 1
            # Remove dead tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            # Convert x1,y1,x2,y2 to x,y,w,h for output
            final_res = []
            for r in ret:
                r = r[0]
                # r is [x1,y1,x2,y2, id, score]
                x1, y1, x2, y2 = r[0], r[1], r[2], r[3]
                score = r[5] if len(r) > 5 else 1.0

                final_res.append([x1, y1, x2 - x1, y2 - y1, score])
            return final_res

        return []


# =========================================================================
#  Helper: Slicing Logic (SAHI style)
# =========================================================================


def slice_image(image, slice_wh=(640, 640), overlap_ratio=0.2):
    """
    Divides image into overlapping slices.
    Returns: list of (slice_img, x_offset, y_offset)
    """
    img_h, img_w, _ = image.shape
    slice_w, slice_h = slice_wh

    X_points = list(range(0, img_w - slice_w, int(slice_w * (1 - overlap_ratio))))
    Y_points = list(range(0, img_h - slice_h, int(slice_h * (1 - overlap_ratio))))

    # Add final points to ensure full coverage
    X_points.append(img_w - slice_w)
    Y_points.append(img_h - slice_h)

    # Remove duplicates
    X_points = sorted(list(set(X_points)))
    Y_points = sorted(list(set(Y_points)))

    slices = []
    for y in Y_points:
        for x in X_points:
            s_img = image[y : y + slice_h, x : x + slice_w]
            slices.append((s_img, x, y))

    return slices


def nms_global(detections, iou_thresh=0.5):
    """
    Merges overlapping detections from different slices.
    detections: list of [x1, y1, x2, y2, conf]
    """
    if len(detections) == 0:
        return []

    boxes = torch.tensor([d[:4] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in detections], dtype=torch.float32)

    keep_indices = torchvision.ops.nms(boxes, scores, iou_thresh)

    return [detections[i] for i in keep_indices]


# =========================================================================
#  Parallel Worker & Pipeline Execution
# =========================================================================

_WORKER_MODEL = None


def load_worker_model(model_path):
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = YOLO(model_path)
    return _WORKER_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 9.
    """
    video_path, config, gt_data = args
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

    if YOLO is None:
        raise ImportError("ultralytics library missing")

    model = load_worker_model(config["model_path"])

    video_name = os.path.basename(video_path)
    images = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    if not images:
        return None

    vid_tp = vid_fp = vid_fn = 0
    vid_dotd_list = []
    vid_all_preds = []
    vid_all_gts = []
    image_results = []

    vid_start = time.time()
    n_frames = len(images)

    # Initialize Strategy 9 Specific Tracker (Kalman + Hungarian)
    kf_tracker = SortDotDTracker(
        max_age=config["max_age"],
        min_hits=config["min_hits"],
        dist_threshold=config["tracker_dist_thresh"],
    )

    for i, img_path in enumerate(images):
        img_start_time = time.time()

        if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
            percent = ((i + 1) / n_frames) * 100
            logger.info(
                f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
            )

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # --- STEP 1: SAHI SLICING ---
        slices = slice_image(
            frame,
            slice_wh=(
                config.get("sahi_slice_width", 640),
                config.get("sahi_slice_height", 640),
            ),
            overlap_ratio=config.get("sahi_overlap_height_ratio", 0.2),
        )

        # --- STEP 2: BATCH INFERENCE ---
        batch_imgs = [s[0] for s in slices]

        # Run Inference
        results = model(
            batch_imgs,
            imgsz=config["img_size"],
            verbose=False,
            conf=config["conf_thresh"],
            classes=[config["bird_class_id"]],
        )

        # --- STEP 3: MERGE DETECTIONS ---
        raw_detections = []

        for idx, res in enumerate(results):
            off_x, off_y = slices[idx][1], slices[idx][2]
            boxes = res.boxes
            for box in boxes:
                # Get local coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # Transform to global coordinates
                g_x1 = x1 + off_x
                g_y1 = y1 + off_y
                g_x2 = x2 + off_x
                g_y2 = y2 + off_y

                raw_detections.append([g_x1, g_y1, g_x2, g_y2, conf])

        # NMS Merger
        merged_detections = nms_global(raw_detections, iou_thresh=0.3)

        # Prepare for tracker
        if merged_detections:
            tracker_input = np.array(merged_detections)
        else:
            tracker_input = np.empty((0, 5))

        # --- STEP 4: TRACKING (Kalman + Hungarian) ---
        final_preds = kf_tracker.update(tracker_input)

        # --- EVALUATION ---
        key = f"{video_name}/{os.path.basename(img_path)}"
        gts = gt_data.get(key, [])

        # Store for mAP calc
        vid_all_preds.append(final_preds)
        vid_all_gts.append(gts)

        matched_gt = set()
        img_tp = img_fp = 0

        for p_box in final_preds:
            best_dist = 10000
            best_idx = -1
            for idx, g_box in enumerate(gts):
                if idx in matched_gt:
                    continue
                d = vis_utils.calculate_center_distance(p_box, g_box)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx

            if best_dist <= 30:
                vid_tp += 1
                img_tp += 1
                vid_dotd_list.append(best_dist)
                matched_gt.add(best_idx)
            else:
                vid_fp += 1
                img_fp += 1

        img_fn = len(gts) - len(matched_gt)
        vid_fn += img_fn

        # Calculate IoU for matched pairs
        img_ious = []
        matched_gt_indices = set()
        for p_box in final_preds:
            best_iou = 0
            best_idx = -1
            for g_idx, g_box in enumerate(gts):
                if g_idx in matched_gt_indices:
                    continue
                iou = vis_utils.box_iou_xywh(p_box[:4], g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx
            if best_idx != -1 and best_iou > 0:
                img_ious.append(best_iou)
                matched_gt_indices.add(best_idx)

        img_avg_iou = np.mean(img_ious) if img_ious else 0.0

        # Calculate processing time and memory for this image
        img_processing_time = time.time() - img_start_time
        img_mem = vis_utils.get_memory_usage()

        image_result = csv_utils.create_image_result(
            video_name=video_name,
            frame_name=os.path.basename(img_path),
            image_path=img_path,
            predictions=final_preds,
            ground_truths=gts,
            tp=img_tp,
            fp=img_fp,
            fn=img_fn,
            processing_time_sec=img_processing_time,
            iou=img_avg_iou,
            memory_usage_mb=img_mem,
        )
        image_results.append(image_result)

    vid_time = time.time() - vid_start
    fps = n_frames / vid_time if vid_time > 0 else 0
    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate mAP and DotD for video
    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    return {
        "video_name": video_name,
        "n_frames": n_frames,
        "fps": fps,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tp": vid_tp,
        "fp": vid_fp,
        "fn": vid_fn,
        "mAP": vid_map,
        "dotd": vid_dotd,
        "vid_time": vid_time,
        "image_results": image_results,
    }


@register_pipeline("strategy_9")
def run_strategy_9_pipeline(config: Dict[str, Any]):
    """Execute Strategy 9 Pipeline (Parallel): SAHI + YOLO + Tracking."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 9 (PARALLEL): {pipeline_name} ---")

    if YOLO is None:
        raise ImportError("ultralytics library missing")

    logger.info(f"⏳ Loading YOLO Model: {config['model_path']}...")
    try:
        # Check model in main process
        _ = YOLO(config["model_path"])
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()

    video_folders = sorted(glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, "*")))
    video_folders = [f for f in video_folders if os.path.isdir(f)]

    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [video_folders[i] for i in Config.VIDEO_INDEXES]
        else:
            video_folders = video_folders[
                : min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)
            ]

    if not video_folders:
        raise RuntimeError(f"No video folders found in {Config.LOCAL_TRAIN_DIR}")

    logger.info(
        f"📂 Found {len(video_folders)} videos. Starting parallel processing with {Config.MAX_WORKERS} workers..."
    )

    # Initialize results tracker
    tracker = csv_utils.get_results_tracker()

    total_tp = total_fp = total_fn = total_time = total_frames = 0
    total_map_sum = 0.0
    total_dotd_sum = 0.0
    total_videos_processed = 0

    worker_args = [(vf, config, gt_data) for vf in video_folders]

    import concurrent.futures

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=Config.MAX_WORKERS
    ) as executor:
        future_to_video = {
            executor.submit(process_video_worker, args): args[0] for args in worker_args
        }

        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            video_name = os.path.basename(video_path)
            try:
                result = future.result()
                if result is None:
                    continue

                vis_utils.log_video_metrics(
                    logger,
                    result["video_name"],
                    {
                        "n_frames": result["n_frames"],
                        "fps": result["fps"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1_score": result["f1_score"],
                        "tp": result["tp"],
                        "fp": result["fp"],
                        "fn": result["fn"],
                        "mAP": result["mAP"],
                        "dotd": result["dotd"],
                        "vid_time": result["vid_time"],
                        "iou": (
                            np.mean([r["iou"] for r in result["image_results"]])
                            if result["image_results"]
                            else 0.0
                        ),
                        "memory_usage_mb": (
                            np.mean(
                                [r["memory_usage_mb"] for r in result["image_results"]]
                            )
                            if result["image_results"]
                            else 0.0
                        ),
                    },
                )

                total_frames += result["n_frames"]
                total_time += result["vid_time"]
                total_tp += result["tp"]
                total_fp += result["fp"]
                total_fn += result["fn"]
                total_map_sum += result["mAP"]
                total_dotd_sum += result["dotd"]
                total_videos_processed += 1

                for img_res in result["image_results"]:
                    tracker.add_image_result(pipeline_name, img_res)
                tracker.save_batch(pipeline_name, batch_size=1)

            except Exception as e:
                logger.error(f"❌ Error processing {video_name}: {e}", exc_info=True)

    # Calculate overall metrics
    avg_fps = total_frames / total_time if total_time > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0
    )

    # Aggregate additional metrics from detailed data
    p_data = tracker.detailed_data.get(pipeline_name, [])
    overall_iou = np.mean([d["iou"] for d in p_data]) if p_data else 0.0
    overall_mem = np.mean([d["memory_usage_mb"] for d in p_data]) if p_data else 0.0

    overall_map = (
        total_map_sum / total_videos_processed if total_videos_processed > 0 else 0.0
    )
    overall_dotd = (
        total_dotd_sum / total_videos_processed if total_videos_processed > 0 else 0.0
    )

    summary_metrics = {
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "iou": overall_iou,
        "mAP": overall_map,
        "dotd": overall_dotd,
        "memory_usage_mb": overall_mem,
        "processing_time_sec": total_time,
        "execution_time_sec": time.time() - start_time,
    }

    # Log summary using standard utility
    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)

    # Update results tracker
    tracker.update_summary(pipeline_name, summary_metrics, config=config)

    return {
        "pipeline": pipeline_name,
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time,
    }
