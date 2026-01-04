from collections import defaultdict
from config import Config
from pipelines import register_pipeline
from typing import Dict, Any
from typing import Dict, Any, List
import concurrent.futures
import csv_utils
import cv2
import datetime
import glob
import logging
import math
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import torchvision
import vis_utils

"""
Baseline Pipeline: YOLO with 4x3 Tiled Inference

This pipeline implements the baseline strategy using:
- YOLO pretrained model (Upgraded from YOLO)
- 4x3 grid tiling with overlap for better small object detection
- Batch inference optimization
- Center distance matching for evaluation
"""
# Attempt to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def get_base_predictions(model, img, img_size, conf_thresh, classes):
    """
    Runs simple full-image inference using YOLO.

    Args:
        model: YOLO model (ultralytics)
        img: Input image (BGR format)
        img_size: Target size for inference
        conf_thresh: Confidence threshold
        classes: List of class IDs to filter

    Returns:
        List of predictions in [x, y, w, h] format
    """
    # YOLO Inference
    results = model(
        img, imgsz=img_size, verbose=False, conf=conf_thresh, classes=classes
    )

    final_preds = []
    if len(results) > 0:
        boxes = results[0].boxes
        if len(boxes) > 0:
            # Convert to xywh format [x, y, w, h]
            # Use cpu() and numpy() for consistent format
            xyxy_boxes = boxes.xyxy.cpu().numpy()
            for box in xyxy_boxes:
                x1, y1, x2, y2 = box[:4]
                score = box[4] if len(box) > 4 else 0.0
                final_preds.append(
                    [float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score)]
                )

    return final_preds


def get_tiled_predictions(model, img, img_size, conf_thresh, classes, use_nms=True):
    """
    Splits image into a 6x4 Grid (24 tiles) and runs inference using YOLO.
    Optimization: Sends all tiles in BATCHES to maximize throughput.

    Args:
        model: YOLO model (ultralytics)
        img: Input image (BGR format)
        img_size: Target size for inference
        conf_thresh: Confidence threshold
        classes: List of class IDs to filter (e.g. [14])
        use_nms: Whether to apply global Non-Maximum Suppression

    Returns:
        List of predictions in [x, y, w, h] format
    """
    h, w, _ = img.shape

    # Grid Configuration: 6 Cols x 4 Rows = 24 Tiles
    # For 3840x2160: 3840/6 = 640, 2160/4 = 540. Matches YOLO12 native resolution.
    N_COLS = 6
    N_ROWS = 4

    h_step = h // N_ROWS
    w_step = w // N_COLS
    h_over = int(h_step * 0.20)  # 20% overlap
    w_over = int(w_step * 0.20)

    crops = []
    offsets = []

    for r in range(N_ROWS):
        for c in range(N_COLS):
            y1 = max(0, r * h_step - h_over)
            x1 = max(0, c * w_step - w_over)
            y2 = min(h, (r + 1) * h_step + h_over)
            x2 = min(w, (c + 1) * w_step + w_over)

            crops.append(img[y1:y2, x1:x2])
            offsets.append((x1, y1))

    # Batch Inference
    all_boxes = []
    all_scores = []

    # Process ALL 12 tiles in one batch
    CHUNK_SIZE = 12

    for i in range(0, len(crops), CHUNK_SIZE):
        sub_crops = crops[i : i + CHUNK_SIZE]
        sub_offsets = offsets[i : i + CHUNK_SIZE]

        # YOLO Inference
        # verbose=False reduces log spam
        results = model(
            sub_crops, imgsz=img_size, verbose=False, conf=conf_thresh, classes=classes
        )

        for j, res in enumerate(results):
            # Ultralytics results object
            boxes = res.boxes
            if len(boxes) > 0:
                # boxes.xyxy is (N, 4), boxes.conf is (N,)
                # We need to move these to CPU numpy or tensor
                local_boxes = boxes.xyxy.cpu()
                local_scores = boxes.conf.cpu()

                x_off, y_off = sub_offsets[j]

                # Shift crop coordinates back to full-frame
                # Clone to avoid modifying the original if cached
                shifted_boxes = local_boxes.clone()
                shifted_boxes[:, 0] += x_off
                shifted_boxes[:, 1] += y_off
                shifted_boxes[:, 2] += x_off
                shifted_boxes[:, 3] += y_off

                all_boxes.append(shifted_boxes)
                all_scores.append(local_scores)

    if not all_boxes:
        return []

    # Merge all predictions
    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)

    if use_nms:
        keep_indices = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
        final_boxes = pred_boxes[keep_indices]
        final_scores = pred_scores[keep_indices]
    else:
        final_boxes = pred_boxes
        final_scores = pred_scores

    final_preds = []
    # Convert to standard list format [x, y, w, h, score] for downstream use
    final_boxes_np = final_boxes.numpy()
    final_scores_np = final_scores.numpy()

    for i, box in enumerate(final_boxes_np):
        x1, y1, x2, y2 = box
        score = final_scores_np[i]
        final_preds.append(
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(score)]
        )

    return final_preds


@register_pipeline("baseline_base")
@register_pipeline("baseline_w_tiling")
@register_pipeline("baseline_w_tiling_and_nms")
def run_baseline(config: Dict[str, Any]):
    """
    Core logic for running all baseline variants.
    Behavior is controlled by `use_tiling` and `use_nms` in the config.
    PARALLELIZED VERSION
    """
    pipeline_name = config["run_name"]
    logger = logging.getLogger(f"{pipeline_name}")
    logger.info(f"--- STARTING VARIANT: {pipeline_name} ---")

    # We check model existence in main process but load in workers
    if YOLO is None:
        logger.error("❌ ultralytics library missing")
        raise ImportError("ultralytics library missing")

    # Load ground truth
    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()

    # Select videos to process
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

    # Prepare Args
    # The config dict is now self-contained, so we can pass it directly.
    worker_args = [(vf, config, gt_data) for vf in video_folders]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=Config.MAX_WORKERS
    ) as executor:
        # Submit all jobs
        future_to_video = {
            executor.submit(process_video_worker, args): args[0] for args in worker_args
        }

        for future in concurrent.futures.as_completed(future_to_video):
            video_path = future_to_video[future]
            video_name = os.path.basename(video_path)
            try:
                result = future.result()
                if result is None:
                    logger.warning(f"⚠️ No result for {video_name}")
                    continue

                # Unpack results
                # Metric Logging
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

                # Update totals
                total_frames += result["n_frames"]
                total_time += result["vid_time"]
                total_tp += result["tp"]
                total_fp += result["fp"]
                total_fn += result["fn"]
                total_map_sum += result["mAP"]
                total_dotd_sum += result["dotd"]
                total_videos_processed += 1

                # Add to tracker
                for img_res in result["image_results"]:
                    tracker.add_image_result(pipeline_name, img_res)

                # Save batch occasionally (here we save after every video to be safe)
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

    # Prepare summary metrics
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

    # Update results tracker with summary metrics
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


# Global model cache for worker processes
_WORKER_MODEL = None


def load_worker_model(model_name):
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = YOLO(model_name)
    return _WORKER_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video.
    Args:
        args: Tuple containing (video_path, config, gt_data_subset)
    Returns:
        Dict: Video metrics and list of image results
    """
    video_path, config, gt_data = args

    # Setup logging for the worker
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config.get("run_name"))
    if YOLO is None:
        raise ImportError("ultralytics library missing")

    model = load_worker_model(config["model_name"])

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

    # These flags are now read directly from the self-contained config
    use_sahi = config.get("use_sahi", False)
    use_tiling = config.get("use_tiling", False)
    use_nms = config.get("use_nms", False)

    for i, img_path in enumerate(images):
        img_start_time = time.time()

        if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
            percent = ((i + 1) / n_frames) * 100
            logger.info(
                f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
            )

        img = cv2.imread(img_path)
        if img is None:
            continue

        if use_sahi:
            preds = vis_utils.get_sahi_predictions(model, img, config)
        elif use_tiling:
            preds = get_tiled_predictions(
                model,
                img,
                config["img_size"],
                config["conf_thresh"],
                config["model_classes"],
                use_nms=use_nms,
            )
        else:
            preds = get_base_predictions(
                model,
                img,
                config["img_size"],
                config["conf_thresh"],
                config["model_classes"],
            )

        # Persistence omitted in baseline? Original code didn't have tracker in baseline.py
        final_preds = preds

        # --- EVALUATION ---
        key = f"{video_name}/{os.path.basename(img_path)}"
        gts = gt_data.get(key, [])

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
                img_tp += 1
                vid_tp += 1
                vid_dotd_list.append(best_dist)
                matched_gt.add(best_idx)
            else:
                img_fp += 1
                vid_fp += 1

        img_fn = len(gts) - len(matched_gt)
        vid_fn += img_fn

        # IoU
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
        img_processing_time = time.time() - img_start_time
        img_mem = vis_utils.get_memory_usage()

        # Collect Result
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

    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    # We return the aggregated metrics for this video and the detailed image results
    # The main process will handle logging and saving to CSV to avoid IPC cost of big objects/locking

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


"""
Strategy 10 Pipeline: Motion-Gated Native Tiling + YOLO

This pipeline implements a high-precision hybrid approach:
1. Global Motion Compensation (GMC) to stabilize the background.
2. Motion Gating: Divide the frame into 640x640 native tiles.
3. Active Tile Selection: Only process tiles with significant detected motion.
4. Native Inference: Run YOLO on active tiles without ANY resizing to maintain pixel accuracy.
"""


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


logger = logging.getLogger(__name__)


def get_native_slices(img_h, img_w, slice_wh=(640, 640), overlap_ratio=0.2):
    """
    Generates coordinates for overlapping tiles of a fixed size.
    Ensures that for a 4K image, every slice is exactly 640x640.
    """
    slice_w, slice_h = slice_wh

    # Step size based on overlap
    step_x = int(slice_w * (1 - overlap_ratio))
    step_y = int(slice_h * (1 - overlap_ratio))

    # Calculate starting points
    x_points = list(range(0, img_w - slice_w, step_x))
    if img_w > slice_w:
        x_points.append(img_w - slice_w)

    y_points = list(range(0, img_h - slice_h, step_y))
    if img_h > slice_h:
        y_points.append(img_h - slice_h)

    # Unique sorted points
    x_points = sorted(list(set(x_points)))
    y_points = sorted(list(set(y_points)))

    coords = []
    for y in y_points:
        for x in x_points:
            coords.append((x, y, x + slice_w, y + slice_h))
    return coords


_WORKER_MODEL = None


def load_worker_model(model_name):
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = YOLO(model_name)
    return _WORKER_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 10.
    """
    video_path, config, gt_data = args
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

    if YOLO is None:
        raise ImportError("ultralytics library missing")

    model = load_worker_model(config["model_name"])

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
    prev_gray = None
    use_sahi = config.get("use_sahi", False)

    # Determine tile grid once
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        return None

    h_img, w_img = first_frame.shape[:2]
    tile_coords = get_native_slices(
        h_img,
        w_img,
        slice_wh=(config["img_size"], config["img_size"]),
        overlap_ratio=0.2,
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

        raw_detections = []
        if use_sahi:
            raw_detections = vis_utils.get_sahi_predictions(model, frame, config)
        else:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 1. GMC + Motion Mask + Keyframe Logic
            active_tiles = []
            active_offsets = []

            is_keyframe = (
                config["full_scan_interval"] > 0
                and i % config["full_scan_interval"] == 0
            )

            if is_keyframe:
                for x1, y1, x2, y2 in tile_coords:
                    active_tiles.append(frame[y1:y2, x1:x2])
                    active_offsets.append((x1, y1))

            elif prev_gray is not None:
                warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                if warped_prev is not None:
                    diff = cv2.absdiff(curr_gray, warped_prev)
                    mean, std = cv2.meanStdDev(diff)
                    final_thresh = max(
                        20,
                        min(
                            80,
                            mean[0][0] + config["motion_thresh_scale"] * std[0][0],
                        ),
                    )
                    _, thresh = cv2.threshold(
                        diff, final_thresh, 255, cv2.THRESH_BINARY
                    )

                    if config.get("use_morphological_dilation", False):
                        kernel = np.ones((3, 3), np.uint8)
                        thresh = cv2.dilate(thresh, kernel, iterations=1)

                    for x1, y1, x2, y2 in tile_coords:
                        tile_mask = thresh[y1:y2, x1:x2]
                        if cv2.countNonZero(tile_mask) > config.get(
                            "motion_pixel_threshold", 20
                        ):
                            active_tiles.append(frame[y1:y2, x1:x2])
                            active_offsets.append((x1, y1))

            if active_tiles:
                results = model(
                    active_tiles,
                    imgsz=config["img_size"],
                    verbose=False,
                    conf=config["conf_thresh"],
                    classes=config["model_classes"],
                )

                all_boxes = []
                all_scores = []
                for j, res in enumerate(results):
                    boxes = res.boxes
                    if len(boxes) > 0:
                        local_boxes = boxes.xyxy.cpu()
                        local_scores = boxes.conf.cpu()
                        x_off, y_off = active_offsets[j]

                        shifted_boxes = local_boxes.clone()
                        shifted_boxes[:, 0] += x_off
                        shifted_boxes[:, 1] += y_off
                        shifted_boxes[:, 2] += x_off
                        shifted_boxes[:, 3] += y_off

                        all_boxes.append(shifted_boxes)
                        all_scores.append(local_scores)

                if all_boxes:
                    pred_boxes = torch.cat(all_boxes, dim=0)
                    pred_scores = torch.cat(all_scores, dim=0)
                    keep_indices = torchvision.ops.nms(
                        pred_boxes, pred_scores, iou_threshold=0.45
                    )
                    final_boxes = pred_boxes[keep_indices]
                    final_scores = pred_scores[keep_indices]

                    raw_detections = []
                    for i, box in enumerate(final_boxes):
                        x1, y1, x2, y2 = box.tolist()
                        score = float(final_scores[i])
                        raw_detections.append(
                            [
                                float(x1),
                                float(y1),
                                float(x2 - x1),
                                float(y2 - y1),
                                score,
                            ]
                        )

            prev_gray = curr_gray

        # --- EVALUATION ---
        img_filename = os.path.basename(img_path)
        key = f"{video_name}/{img_filename}"
        gts = gt_data.get(key, [])

        # Store for mAP calc
        vid_all_preds.append(raw_detections)
        vid_all_gts.append(gts)

        matched_gt = set()
        img_tp = img_fp = 0

        for p_box in raw_detections:
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
        for p_box in raw_detections:
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
            frame_name=img_filename,
            image_path=img_path,
            predictions=raw_detections,
            ground_truths=gts,
            tp=img_tp,
            fp=img_fp,
            fn=img_fn,
            processing_time_sec=img_processing_time,
            iou=img_avg_iou,
            memory_usage_mb=img_mem,
        )
        image_results.append(image_result)

    # Video Stats
    vid_time = time.time() - vid_start
    vid_fps = n_frames / vid_time if vid_time > 0 else 0

    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate mAP and DotD for video
    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    return {
        "video_name": video_name,
        "n_frames": n_frames,
        "fps": vid_fps,
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


@register_pipeline("strategy_10")
def run_strategy_10_pipeline(config: Dict[str, Any]):
    """Execute Strategy 10: Motion-Gated Native Tiling."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 10 (PARALLEL): {pipeline_name} ---")

    if YOLO is None:
        logger.error("❌ ultralytics library not found.")
        raise ImportError("ultralytics library missing")

    logger.info(f"⏳ Loading YOLO: {config['model_name']}...")
    try:
        # Check model in main process
        _ = YOLO(config["model_name"])
        logger.info(f"✅ Model Loaded.")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time_global = time.time()

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

    total_tp = total_fp = total_fn = total_time_sec = total_frames = 0
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
                total_time_sec += result["vid_time"]
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

    # Final Summary
    # Calculate overall metrics
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
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
        "processing_time_sec": total_time_sec,
        "execution_time_sec": time.time() - start_time_global,
    }

    # Log summary using standard utility
    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)

    tracker.update_summary(pipeline_name, summary_metrics, config=config)

    return {
        "pipeline": pipeline_name,
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time_global,
    }


"""
Strategy 11 Pipeline: ROI Selection + YOLO Classification Filter + YOLO Detection

Implements efficient detection by:
1. Generating motion-based ROI proposals (from Strategy 8).
2. Filtering these ROIs through a lightweight YOLO classifier.
3. Running full YOLO detection only on ROIs that passed the classifier.
"""


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


logger = logging.getLogger(__name__)


def get_tiled_coords(h_img, w_img, tile_size, overlap_ratio=0.2):
    """Generates coordinates for overlapping tiles."""
    step = int(tile_size * (1 - overlap_ratio))
    x_points = list(range(0, w_img - tile_size, step))
    x_points.append(w_img - tile_size)
    y_points = list(range(0, h_img - tile_size, step))
    y_points.append(h_img - tile_size)

    x_points = sorted(list(set(x_points)))
    y_points = sorted(list(set(y_points)))

    coords = []
    for y in y_points:
        for x in x_points:
            coords.append((x, y, x + tile_size, y + tile_size))
    return coords


def get_filtered_roi_predictions(
    det_model, cls_model, img_bgr, config: Dict[str, Any], frame_idx, motion_mask=None
):
    """
    1. Grid-based multi-scale classification (224px and 448px).
    2. Overlapping tiles to prevent split objects.
    3. Motion-gating (optional) to skip empty skies.
    4. Hit verification with 640px detector.
    """
    logger = logging.getLogger("pipelines.strategy_11.predictions")
    if det_model is None or cls_model is None:
        return []

    h_img, w_img, _ = img_bgr.shape

    # 1. Generate Grids
    grid_small = get_tiled_coords(
        h_img, w_img, config["cls_img_size"], config["cls_overlap"]
    )
    grid_large = get_tiled_coords(
        h_img, w_img, config["cls_scale2_size"], config["cls_overlap"]
    )

    active_crops = []
    active_info = []  # (x0, y0, scale_size)

    # 2. Extract & Filter by Motion (if mask provided)
    for x0, y0, x1, y1 in grid_small:
        if motion_mask is not None:
            m_pixels = cv2.countNonZero(motion_mask[y0:y1, x0:x1])
            if m_pixels < 5:  # Slightly lower threshold for tiny birds
                continue
        active_crops.append(img_bgr[y0:y1, x0:x1])
        active_info.append((x0, y0, config["cls_img_size"]))

    for x0, y0, x1, y1 in grid_large:
        if motion_mask is not None:
            m_pixels = cv2.countNonZero(motion_mask[y0:y1, x0:x1])
            if m_pixels < 10:
                continue
        active_crops.append(img_bgr[y0:y1, x0:x1])
        active_info.append((x0, y0, config["cls_scale2_size"]))

    if not active_crops:
        if frame_idx % 100 == 0:
            logger.debug(f"Frame {frame_idx}: No active crops after motion gating.")
        return []

    # 3. Stage 1: Batch Classification (imgsz=224)
    cls_results = cls_model(active_crops, imgsz=config["cls_img_size"], verbose=False)

    verification_centers = []

    # Common bird-related keywords in ImageNet for better filtering
    bird_keywords = [
        "bird",
        "finch",
        "bunting",
        "indigo",
        "robin",
        "bulbul",
        "jay",
        "magpie",
        "chickadee",
        "water ouzel",
        "dipper",
        "kite",
        "eagle",
        "vulture",
        "falcon",
    ]

    for idx, res in enumerate(cls_results):
        top1_idx = res.probs.top1
        top1_conf = float(res.probs.top1conf)
        top1_name = res.names[top1_idx].lower()

        # Robust bird check
        is_bird = any(kw in top1_name for kw in bird_keywords)

        if is_bird and top1_conf >= config["cls_conf_thresh"]:
            x0, y0, sz = active_info[idx]
            cx, cy = x0 + sz / 2, y0 + sz / 2
            verification_centers.append((cx, cy))
            logger.debug(
                f"Frame {frame_idx}: Hit! Tile at ({x0}, {y0}) classified as '{top1_name}' ({top1_conf:.2f})"
            )
        elif top1_conf > 0.3:  # Log interesting near-misses for debug
            logger.debug(
                f"Frame {frame_idx}: Candidate at ({active_info[idx][0]}, {active_info[idx][1]}) was '{top1_name}' ({top1_conf:.2f})"
            )

    if not verification_centers:
        return []

    logger.info(
        f"Frame {frame_idx}: {len(verification_centers)} potential bird regions found by classifier."
    )

    # 4. Stage 2: Verification with 640px Detector
    final_verification_crops = []
    final_verification_offsets = []

    merged_centers = []
    temp_centers = verification_centers.copy()
    while temp_centers:
        curr = temp_centers.pop(0)
        merged_centers.append(curr)
        temp_centers = [
            c
            for c in temp_centers
            if np.sqrt((c[0] - curr[0]) ** 2 + (c[1] - curr[1]) ** 2) > 200
        ]

    for cx, cy in merged_centers:
        x0 = int(max(0, cx - config["img_size"] / 2))
        y0 = int(max(0, cy - config["img_size"] / 2))
        x1 = int(min(w_img, x0 + config["img_size"]))
        y1 = int(min(h_img, y0 + config["img_size"]))

        if x1 - x0 < config["img_size"]:
            x0 = max(0, x1 - config["img_size"])
        if y1 - y0 < config["img_size"]:
            y0 = max(0, y1 - config["img_size"])

        crop = img_bgr[y0:y1, x0:x1]
        if crop.size > 0:
            final_verification_crops.append(crop)
            final_verification_offsets.append((x0, y0))

    if not final_verification_crops:
        return []

    # Final Detection Pass
    det_results = det_model(
        final_verification_crops,
        imgsz=config["img_size"],
        verbose=False,
        conf=config["conf_thresh"],
        classes=config["model_classes"],
    )

    all_boxes = []
    all_scores = []

    for j, res in enumerate(det_results):
        boxes = res.boxes
        if len(boxes) > 0:
            local_boxes = boxes.xyxy.cpu()
            local_scores = boxes.conf.cpu()
            x_off, y_off = final_verification_offsets[j]

            shifted_boxes = local_boxes.clone()
            shifted_boxes[:, 0] += x_off
            shifted_boxes[:, 1] += y_off
            shifted_boxes[:, 2] += x_off
            shifted_boxes[:, 3] += y_off

            all_boxes.append(shifted_boxes)
            all_scores.append(local_scores)
            logger.info(f"Frame {frame_idx}: Detector CONFIRMED {len(boxes)} bird(s).")

    if not all_boxes:
        logger.debug(
            f"Frame {frame_idx}: Detector rejected all {len(final_verification_crops)} classifier proposals."
        )
        return []

    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)
    keep_indices = torchvision.ops.nms(
        pred_boxes, pred_scores, iou_threshold=config["iou_thresh"]
    )
    final_boxes = pred_boxes[keep_indices]
    final_scores = pred_scores[keep_indices]

    final_results = []
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box.tolist()
        score = float(final_scores[i])
        final_results.append(
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1), score]
        )
    return final_results


_WORKER_DET_MODEL = None
_WORKER_CLS_MODEL = None


def load_worker_models(det_model_name, cls_model_name):
    global _WORKER_DET_MODEL, _WORKER_CLS_MODEL
    if _WORKER_DET_MODEL is None:
        _WORKER_DET_MODEL = YOLO(det_model_name)
    if _WORKER_CLS_MODEL is None:
        _WORKER_CLS_MODEL = YOLO(cls_model_name)
    return _WORKER_DET_MODEL, _WORKER_CLS_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 11.
    """
    video_path, config, gt_data = args
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

    if YOLO is None:
        raise ImportError("ultralytics library missing")

    det_model, cls_model = load_worker_models(
        config["model_name"], config["classifier_model_name"]
    )

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
    prev_gray = None
    use_sahi = config.get("use_sahi", False)

    # Increase skip threshold to bridge the gap between detection frames (every 5 frames)
    # min_hits=1 ensures we don't drop discoveries immediately.
    obj_tracker = vis_utils.ObjectTracker(
        dist_thresh=100, max_frames_to_skip=config["detect_every"], min_hits=1
    )

    last_final_preds = []  # Persistent results across skipped frames

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

        raw_detections = []
        if use_sahi:
            # When using SAHI, we use the main detection model directly
            raw_detections = vis_utils.get_sahi_predictions(det_model, frame, config)
        else:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Stage 1: Classifier-Gated Detection
            if i % config["detect_every"] == 0:
                motion_mask = None
                if prev_gray is not None:
                    warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                    if warped_prev is not None:
                        diff = cv2.absdiff(curr_gray, warped_prev)
                        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                        # Initial filtering to reduce noise
                        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, k3)
                        motion_mask = cv2.dilate(motion_mask, k3, iterations=2)

                # Run Tiled multi-scale classification guided by motion
                raw_detections = get_filtered_roi_predictions(
                    det_model, cls_model, frame, config, i, motion_mask=motion_mask
                )

                # Update tracker only on discovery frames
                last_final_preds = obj_tracker.update(raw_detections)

            prev_gray = curr_gray
        # Use persistent predictions for all frames
        final_preds = last_final_preds

        # Evaluation
        img_filename = os.path.basename(img_path)
        key = f"{video_name}/{img_filename}"
        gts = gt_data.get(key, [])

        # Store for mAP calc
        vid_all_preds.append(final_preds)
        vid_all_gts.append(gts)

        matched_gt = set()
        img_tp = img_fp = 0

        for p_idx, p_box in enumerate(final_preds):
            best_dist = 10000
            best_idx = -1
            for g_idx, g_box in enumerate(gts):
                if g_idx in matched_gt:
                    continue
                d = vis_utils.calculate_center_distance(p_box, g_box)
                if d < best_dist:
                    best_dist = d
                    best_idx = g_idx

            # Distance threshold for TP match (increased to 100 for 4K)
            if best_dist <= 100:
                vid_tp += 1
                img_tp += 1
                vid_dotd_list.append(best_dist)
                matched_gt.add(best_idx)
            else:
                vid_fp += 1
                img_fp += 1

        img_fn = len(gts) - len(matched_gt)
        vid_fn += img_fn

        # IoU
        img_ious = []
        matched_gt_indices = set()
        for p_box in final_preds:
            best_iou = 0
            best_idx = -1
            for g_idx, g_box in enumerate(gts):
                if g_idx in matched_gt_indices:
                    continue
                # p_box is [x,y,w,h,score]
                iou = vis_utils.box_iou_xywh(p_box[:4], g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = g_idx
            if best_idx != -1 and best_iou > 0:
                img_ious.append(best_iou)
                matched_gt_indices.add(best_idx)

        img_avg_iou = np.mean(img_ious) if img_ious else 0.0
        img_processing_time = time.time() - img_start_time
        img_mem = vis_utils.get_memory_usage()

        image_result = csv_utils.create_image_result(
            video_name=video_name,
            frame_name=img_filename,
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

    # Video Stats
    vid_time = time.time() - vid_start
    vid_fps = n_frames / vid_time if vid_time > 0 else 0
    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate mAP and DotD for video
    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    return {
        "video_name": video_name,
        "n_frames": n_frames,
        "fps": vid_fps,
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


@register_pipeline("strategy_11")
def run_strategy_11_pipeline(config: Dict[str, Any]):
    """Execute Strategy 11: ROI + Classifier + Detector."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 11 (PARALLEL): {pipeline_name} ---")

    if YOLO is None:
        logger.error("❌ ultralytics library not found.")
        raise ImportError("ultralytics library missing")

    # Load models
    logger.info(f"⏳ Loading Detector: {config['model_name']}...")
    logger.info(f"⏳ Loading Classifier: {config['classifier_model_name']}...")
    try:
        # Check models in main process
        _ = YOLO(config["model_name"])
        _ = YOLO(config["classifier_model_name"])
        logger.info(f"✅ Models Loaded.")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time_global = time.time()

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

    tracker = csv_utils.get_results_tracker()
    total_tp = total_fp = total_fn = total_time_sec = total_frames = 0
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
                total_time_sec += result["vid_time"]
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

    # Final Summary
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0
    )

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
        "processing_time_sec": total_time_sec,
        "execution_time_sec": time.time() - start_time_global,
    }

    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)
    tracker.update_summary(pipeline_name, summary_metrics, config=config)

    return {
        "pipeline": pipeline_name,
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time_global,
    }


"""
Strategy 12 Pipeline: GMC + Interpolation

This pipeline builds on Strategy 2 by adding frame skipping and interpolation.
1. Global Motion Compensation (GMC) to align frames.
2. Runs detection only every N frames (`detect_every`).
3. For intermediate frames, it interpolates bounding boxes linearly between keyframes.
"""


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


logger = logging.getLogger(__name__)


def _expand_roi_xywh(box, w_img, h_img, scale=2.0, min_size=192):
    """Expand ROI with minimum size constraint."""
    x, y, w, h = box
    cx = x + w * 0.5
    cy = y + h * 0.5
    rw = max(w * scale, min_size)
    rh = max(h * scale, min_size)
    x0 = int(max(0, cx - rw * 0.5))
    y0 = int(max(0, cy - rh * 0.5))
    x1 = int(min(w_img, cx + rw * 0.5))
    y1 = int(min(h_img, cy + rh * 0.5))
    return x0, y0, x1, y1


def get_roi_predictions(model, img_bgr, proposals_xywh, config: Dict[str, Any]):
    """Run YOLO only on ROI crops around proposals."""
    if model is None or not proposals_xywh:
        return []

    h, w, _ = img_bgr.shape
    crops = []
    offsets = []

    use_props = proposals_xywh[: min(len(proposals_xywh), config["max_rois"])]

    for b in use_props:
        x0, y0, x1, y1 = _expand_roi_xywh(
            b, w, h, scale=config["roi_scale"], min_size=config["min_roi_size"]
        )
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((x0, y0))

    if len(crops) == 0:
        return []

    results = model(
        crops,
        imgsz=config["img_size"],
        verbose=False,
        conf=config["conf_thresh"],
        classes=config["model_classes"],
    )

    all_boxes = []
    all_scores = []
    for j, res in enumerate(results):
        if len(res.boxes) > 0:
            local_boxes = res.boxes.xyxy.cpu()
            x_off, y_off = offsets[j]
            shifted_boxes = local_boxes.clone()
            shifted_boxes[:, 0] += x_off
            shifted_boxes[:, 1] += y_off
            shifted_boxes[:, 2] += x_off
            shifted_boxes[:, 3] += y_off
            all_boxes.append(shifted_boxes)
            all_scores.append(res.boxes.conf.cpu())

    if not all_boxes:
        return []

    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)

    # Recover scores
    final_preds = []

    # We need to map indices back to scores.
    # Just iterate carefully.
    keep_indices = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
    final_boxes = pred_boxes[keep_indices]
    final_scores = pred_scores[keep_indices]

    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box.tolist()
        score = float(final_scores[i])
        final_preds.append(
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1), score]
        )

    return final_preds


_WORKER_MODEL = None


def load_worker_model(model_name):
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = YOLO(model_name)
    return _WORKER_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 12.
    """
    video_path, config, gt_data = args
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

    if YOLO is None:
        raise ImportError("ultralytics library missing")

    model = load_worker_model(config["model_name"])

    video_name = os.path.basename(video_path)
    images = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    if not images:
        return None

    vid_start_time = time.time()
    n_frames = len(images)

    vid_dotd_list = []
    vid_all_preds = []
    vid_all_gts = []
    image_results = []

    vid_tp = vid_fp = vid_fn = 0

    # --- PASS 1: Generate Predictions (including Interpolation) ---
    all_predictions = defaultdict(list)
    last_keyframe_preds: List[List[float]] = []
    last_keyframe_idx = -1
    prev_gray = None
    detect_every = config.get("detect_every", 5)
    use_sahi = config.get("use_sahi", False)

    images_w_metadata = [
        {
            "img_start_time": time.time(),
            "image": image,
        }
        for image in images
    ]

    for i, img_metadata in enumerate(images_w_metadata):
        img_metadata["img_start_time"] = time.time()
        if i % detect_every == 0:

            if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(
                    f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
                )

            frame = cv2.imread(img_metadata["image"])
            if frame is None:
                continue

            current_preds = []
            if use_sahi:
                current_preds = vis_utils.get_sahi_predictions(model, frame, config)
            else:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                proposals = []
                if prev_gray is not None:
                    warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                    if warped_prev is not None:
                        diff = cv2.absdiff(curr_gray, warped_prev)
                        mean, std = cv2.meanStdDev(diff)
                        dynamic_thresh = (
                            mean[0][0] + config["dynamic_multiplier"] * std[0][0]
                        )
                        final_thresh = max(
                            config["min_threshold"],
                            min(config["max_threshold"], dynamic_thresh),
                        )
                        _, thresh = cv2.threshold(
                            diff, final_thresh, 255, cv2.THRESH_BINARY
                        )
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                        thresh = cv2.dilate(thresh, kernel, iterations=1)
                        contours, _ = cv2.findContours(
                            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        h_img, w_img = curr_gray.shape
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if 50 < area < 5000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                if 0.2 < (w / h if h > 0 else 0) < 4.0:
                                    proposals.append([x, y, w, h])
                if proposals:
                    current_preds = get_roi_predictions(model, frame, proposals, config)
                prev_gray = curr_gray

            all_predictions[i] = current_preds

            if last_keyframe_idx != -1:
                interpolated = vis_utils.generate_interpolated_boxes(
                    last_keyframe_preds, current_preds, last_keyframe_idx, i, config
                )
                for frame_idx, boxes in interpolated.items():
                    all_predictions[frame_idx].extend(boxes)

            last_keyframe_preds = current_preds
            last_keyframe_idx = i

    # --- PASS 2: Evaluation ---
    for i, img_metadata in enumerate(images_w_metadata):
        final_preds = all_predictions[i]
        key = f"{video_name}/{os.path.basename(img_metadata['image'])}"
        gts = gt_data.get(key, [])

        # Store for mAP calc
        vid_all_preds.append(final_preds)
        vid_all_gts.append(gts)

        img_tp = img_fp = 0
        matched_gt = set()
        img_ious = []

        for p_box in final_preds:
            best_dist, best_idx = float("inf"), -1
            best_iou = 0
            for idx, g_box in enumerate(gts):
                if idx in matched_gt:
                    continue
                dist = vis_utils.calculate_center_distance(p_box, g_box)
                iou = vis_utils.box_iou_xywh(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                if dist < best_dist:
                    best_dist, best_idx = dist, idx

            if best_dist <= 30:
                img_tp += 1
                matched_gt.add(best_idx)
                img_ious.append(best_iou)
                vid_dotd_list.append(best_dist)
            else:
                img_fp += 1

        img_fn = len(gts) - len(matched_gt)
        vid_tp += img_tp
        vid_fp += img_fp
        vid_fn += img_fn

        img_avg_iou = np.mean(img_ious) if img_ious else 0.0

        img_processing_time = time.time() - img_metadata["img_start_time"]
        img_mem = vis_utils.get_memory_usage()

        # Log per-image results
        image_result = csv_utils.create_image_result(
            video_name=video_name,
            frame_name=os.path.basename(img_metadata["image"]),
            image_path=img_metadata["image"],
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

    # Video-level stats
    vid_time = time.time() - vid_start_time
    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate mAP and DotD for video
    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    return {
        "video_name": video_name,
        "n_frames": n_frames,
        "fps": n_frames / vid_time if vid_time > 0 else 0,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "tp": vid_tp,
        "fp": vid_fp,
        "fn": vid_fn,
        "iou": np.mean([r["iou"] for r in image_results]) if image_results else 0.0,
        "mAP": vid_map,
        "dotd": vid_dotd,
        "vid_time": vid_time,
        "image_results": image_results,
    }


@register_pipeline("strategy_12")
def run_strategy_12_pipeline(config: Dict[str, Any]):
    """Execute Strategy 12: GMC + Frame Skipping + Interpolation."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 12 (PARALLEL): {pipeline_name} ---")

    if YOLO is None:
        raise ImportError("❌ ultralytics library not found.")

    logger.info(f"⏳ Loading YOLO Model: {config['model_name']}...")
    try:
        # Check model in main process
        _ = YOLO(config["model_name"])
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}", exc_info=True)
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

    results_tracker = csv_utils.get_results_tracker()
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
                        "iou": result["iou"],
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
                    results_tracker.add_image_result(pipeline_name, img_res)
                results_tracker.save_batch(pipeline_name, batch_size=1)

            except Exception as e:
                logger.error(f"❌ Error processing {video_name}: {e}", exc_info=True)

    # Final summary
    avg_fps = total_frames / total_time if total_time > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0
    )
    overall_map = (
        total_map_sum / total_videos_processed if total_videos_processed > 0 else 0.0
    )
    overall_dotd = (
        total_dotd_sum / total_videos_processed if total_videos_processed > 0 else 0.0
    )

    # Aggregate additional metrics from detailed data for summary
    p_data = results_tracker.detailed_data.get(pipeline_name, [])
    overall_iou = np.mean([d["iou"] for d in p_data]) if p_data else 0.0
    overall_mem = np.mean([d["memory_usage_mb"] for d in p_data]) if p_data else 0.0

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

    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)
    results_tracker.update_summary(pipeline_name, summary_metrics, config=config)
    return {"pipeline": pipeline_name, "status": "completed", **summary_metrics}


"""
Strategy 13 Pipeline: Motion-Gated Classifier Funnel

This advanced pipeline combines multiple strategies for maximum efficiency:
1. It uses GMC for stabilization and divides the frame into tiles (legacy or SAHI).
2. For each tile, it first performs a fast motion check (from Strategy 10).
3. If motion is found, the tile is passed to the main YOLO detector.
4. If NO motion is found, it performs a fast classification check (from Strategy 11).
5. If the classifier finds a potential bird, the tile is passed to the main YOLO detector.
6. If both checks are negative, the tile is skipped, saving significant computation.
7. Includes an interpolation variant for frame skipping.
"""


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


logger = logging.getLogger(__name__)


def get_native_slices(img_h, img_w, slice_wh=(640, 640), overlap_ratio=0.2):
    """Generates coordinates for overlapping tiles."""
    slice_w, slice_h = slice_wh
    step_x = int(slice_w * (1 - overlap_ratio))
    step_y = int(slice_h * (1 - overlap_ratio))
    x_points = sorted(list(set(range(0, img_w - slice_w, step_x)) | {img_w - slice_w}))
    y_points = sorted(list(set(range(0, img_h - slice_h, step_y)) | {img_h - slice_h}))
    return [(x, y, x + slice_w, y + slice_h) for y in y_points for x in x_points]


# Global model cache for worker processes
_WORKER_DET_MODEL = None
_WORKER_CLS_MODEL = None


def load_worker_models(det_model_name, cls_model_name):
    global _WORKER_DET_MODEL, _WORKER_CLS_MODEL
    if _WORKER_DET_MODEL is None:
        _WORKER_DET_MODEL = YOLO(det_model_name)
    if _WORKER_CLS_MODEL is None:
        _WORKER_CLS_MODEL = YOLO(cls_model_name)
    return _WORKER_DET_MODEL, _WORKER_CLS_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 13.
    Contains the full Motion-Gated Classifier Funnel logic.
    """
    video_path, config, gt_data = args
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

    if YOLO is None:
        raise ImportError("ultralytics library missing")

    det_model, cls_model = load_worker_models(
        config["model_name"], config["classifier_model_name"]
    )

    video_name = os.path.basename(video_path)
    images = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    if not images:
        return None

    n_frames = len(images)
    vid_start_time = time.time()

    # Metrics accumulation for this video
    vid_map = 0.0
    vid_dotd_list = []
    vid_all_preds = []
    vid_all_gts = []

    all_predictions = defaultdict(list)
    last_keyframe_preds: List[List[float]] = []
    last_keyframe_idx = -1
    prev_gray = None

    use_interpolation = config.get("use_interpolation", False)
    detect_every = config.get("detect_every", 1) if use_interpolation else 1
    use_sahi = config.get("use_sahi", False)

    # Read first frame to determine image size and generate tile coordinates
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        return None

    h_img, w_img = first_frame.shape[:2]
    # We use the global get_native_slices helper
    tile_coords = get_native_slices(
        h_img, w_img, (config["img_size"], config["img_size"]), 0.2
    )

    images_w_metadata = [{"img_start_time": 0, "image": image} for image in images]

    # --- Pass 1: Generate Detections ---
    for i, img_metadata in enumerate(images_w_metadata):
        img_metadata["img_start_time"] = time.time()
        if i % detect_every == 0:

            if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(
                    f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
                )

            frame = cv2.imread(img_metadata["image"])
            if frame is None:
                continue

            current_preds = []
            if use_sahi:
                current_preds = vis_utils.get_sahi_predictions(det_model, frame, config)
            else:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                active_tiles, active_offsets = [], []
                if prev_gray is not None:
                    # 1. GMC: Align frames
                    warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                    if warped_prev is not None:
                        # 2. Motion Check
                        diff = cv2.absdiff(curr_gray, warped_prev)
                        mean, std = cv2.meanStdDev(diff)
                        motion_thresh = max(
                            20,
                            min(
                                80,
                                mean[0][0] + config["motion_thresh_scale"] * std[0][0],
                            ),
                        )
                        _, motion_mask = cv2.threshold(
                            diff, motion_thresh, 255, cv2.THRESH_BINARY
                        )

                        cls_cand, cls_offs = [], []
                        for x1, y1, x2, y2 in tile_coords:
                            # 3. Check Motion in Tiles
                            if cv2.countNonZero(motion_mask[y1:y2, x1:x2]) > config.get(
                                "motion_pixel_threshold", 20
                            ):
                                active_tiles.append(frame[y1:y2, x1:x2])
                                active_offsets.append((x1, y1))
                            else:
                                cls_cand.append(frame[y1:y2, x1:x2])
                                cls_offs.append((x1, y1))

                        # 4. Classifier Check on Static Tiles
                        if cls_cand:
                            cls_res = cls_model(
                                cls_cand,
                                imgsz=config["cls_img_size"],
                                verbose=False,
                            )
                            bird_kw = ["bird", "finch", "jay", "eagle", "kite"]
                            for j, r in enumerate(cls_res):
                                is_bird = any(
                                    kw in r.names[r.probs.top1].lower()
                                    for kw in bird_kw
                                )
                                if (
                                    is_bird
                                    and r.probs.top1conf >= config["cls_conf_thresh"]
                                ):
                                    active_tiles.append(cls_cand[j])
                                    active_offsets.append(cls_offs[j])
                else:
                    # First frame logic: normally we just set prev_gray and skip motion/cls checks
                    # or assume everything could be active. Original logic implicitly skipped.
                    pass

                prev_gray = curr_gray

                # 5. Run Detector on Active Tiles
                if active_tiles:
                    det_res = det_model(
                        active_tiles,
                        imgsz=config["img_size"],
                        conf=config["conf_thresh"],
                        classes=config["model_classes"],
                        verbose=False,
                    )
                    all_boxes, all_scores = [], []
                    for j, r in enumerate(det_res):
                        if len(r.boxes) > 0:
                            b, s, x_off, y_off = (
                                r.boxes.xyxy.cpu(),
                                r.boxes.conf.cpu(),
                                *active_offsets[j],
                            )
                            shifted = b.clone()
                            shifted[:, 0::2] += x_off
                            shifted[:, 1::2] += y_off
                            all_boxes.append(shifted)
                            all_scores.append(s)
                    if all_boxes:
                        pred_boxes, pred_scores = torch.cat(all_boxes), torch.cat(
                            all_scores
                        )
                        keep = torchvision.ops.nms(
                            pred_boxes, pred_scores, config.get("iou_thresh", 0.45)
                        )
                        # Retrieve selected boxes and scores
                        kept_boxes = pred_boxes[keep]
                        kept_scores = pred_scores[keep]

                        # Convert to list [x, y, w, h, score]
                        for k_idx, box in enumerate(kept_boxes):
                            x1, y1, x2, y2 = box.tolist()
                            score = float(kept_scores[k_idx])
                            current_preds.append(
                                [
                                    float(x1),
                                    float(y1),
                                    float(x2 - x1),
                                    float(y2 - y1),
                                    float(score),
                                ]
                            )

            all_predictions[i] = current_preds
            if use_interpolation and last_keyframe_idx != -1:
                interpolated = vis_utils.generate_interpolated_boxes(
                    last_keyframe_preds, current_preds, last_keyframe_idx, i, config
                )
                for frame_idx, boxes in interpolated.items():
                    all_predictions[frame_idx].extend(boxes)

            last_keyframe_preds = current_preds
            last_keyframe_idx = i

    # --- Pass 2: Evaluation ---
    image_results = []

    # Pre-calculate video counts for metrics
    vid_tp, vid_fp, vid_fn = 0, 0, 0

    for i, img_metadata in enumerate(images_w_metadata):
        final_preds = all_predictions[i]
        key = f"{video_name}/{os.path.basename(img_metadata['image'])}"
        gts = gt_data.get(key, [])

        # Store for mAP calc
        vid_all_preds.append(final_preds)
        vid_all_gts.append(gts)

        img_tp, img_fp, matched_gt = 0, 0, set()
        img_ious = []
        for p_box in final_preds:
            best_dist, best_idx = float("inf"), -1
            best_iou = 0
            for idx, g_box in enumerate(gts):
                if idx in matched_gt:
                    continue
                # p_box is [x, y, w, h, score] or [x, y, w, h]
                dist = vis_utils.calculate_center_distance(p_box[:4], g_box)
                iou = vis_utils.box_iou_xywh(p_box[:4], g_box)
                if iou > best_iou:
                    best_iou = iou
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_dist <= 30:
                img_tp += 1
                img_ious.append(best_iou)
                vid_dotd_list.append(best_dist)
                matched_gt.add(best_idx)
            else:
                img_fp += 1

        img_fn = len(gts) - len(matched_gt)
        vid_tp += img_tp
        vid_fp += img_fp
        vid_fn += img_fn

        # Calculate IoU for matched pairs
        img_avg_iou = np.mean(img_ious) if img_ious else 0.0

        # Calculate processing time and memory for this image
        img_processing_time = time.time() - img_metadata["img_start_time"]
        img_mem = vis_utils.get_memory_usage()

        image_result = csv_utils.create_image_result(
            video_name=video_name,
            frame_name=os.path.basename(img_metadata["image"]),
            image_path=img_metadata["image"],
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

    vid_time = time.time() - vid_start_time
    fps = len(images) / vid_time if vid_time > 0 else 0
    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate mAP and DotD for video
    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    return {
        "video_name": video_name,
        "n_frames": len(images),
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


@register_pipeline("strategy_13")
def run_strategy_13_pipeline(config: Dict[str, Any]):
    """Execute Strategy 13: Motion-Gated Classifier Funnel."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(f"{pipeline_name}")
    logger.info(f"--- STARTING STRATEGY 13 (PARALLEL): {pipeline_name} ---")

    if YOLO is None:
        raise ImportError("❌ ultralytics library not found.")

    logger.info(f"⏳ Loading Detector: {config['model_name']}...")
    logger.info(f"⏳ Loading Classifier: {config['classifier_model_name']}...")
    try:
        # Check models in main process
        _ = YOLO(config["model_name"])
        _ = YOLO(config["classifier_model_name"])
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}", exc_info=True)
        raise

    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()
    video_folders = sorted(
        [
            f
            for f in glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, "*"))
            if os.path.isdir(f)
        ]
    )

    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [
                video_folders[i] for i in Config.VIDEO_INDEXES if i < len(video_folders)
            ]
        else:
            video_folders = video_folders[
                : min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)
            ]

    results_tracker = csv_utils.get_results_tracker()
    total_tp, total_fp, total_fn, total_time, total_frames = 0, 0, 0, 0, 0
    total_map_sum = 0.0
    total_dotd_sum = 0.0
    total_videos_processed = 0

    worker_args = [(vf, config, gt_data) for vf in video_folders]

    import concurrent.futures

    # Using ProcessPoolExecutor for parallel execution
    # Adjust max_workers as needed, likely defined in Config
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
                    results_tracker.add_image_result(pipeline_name, img_res)
                results_tracker.save_batch(pipeline_name, batch_size=1)

            except Exception as e:
                logger.error(f"❌ Error processing {video_name}: {e}", exc_info=True)

    # Final Summary
    avg_fps = total_frames / total_time if total_time > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0
    )

    overall_map = (
        total_map_sum / total_videos_processed if total_videos_processed > 0 else 0.0
    )
    overall_dotd = (
        total_dotd_sum / total_videos_processed if total_videos_processed > 0 else 0.0
    )

    # Calculate overall IoU/Mem from tracker detailed data
    p_data = results_tracker.detailed_data.get(pipeline_name, [])
    overall_iou = np.mean([d["iou"] for d in p_data]) if p_data else 0.0
    overall_mem = np.mean([d["memory_usage_mb"] for d in p_data]) if p_data else 0.0

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
    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)
    results_tracker.update_summary(pipeline_name, summary_metrics, config=config)
    return {"pipeline": pipeline_name, "status": "completed", **summary_metrics}


"""
Strategy 2 Pipeline: GMC + Dynamic Thresholding + YOLO Refiner

This pipeline migrates the logic from Strategy 2 Colab:
1. Global Motion Compensation (GMC) to align frames.
2. Dynamic Statistical Thresholding (mean + 4*std) to isolate motion.
3. Morphological opening (5x5 kernel) to remove noise.
4. Contour filtering (area, aspect ratio, border) to generate ROIs.
5. YOLO inference on these ROIs for final bird detection.
6. Persistence tracking for temporal consistency.
"""


try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


logger = logging.getLogger(__name__)


def _expand_roi_xywh(box, w_img, h_img, scale=2.0, min_size=192):
    """Expand ROI with minimum size constraint."""
    x, y, w, h = box
    cx = x + w * 0.5
    cy = y + h * 0.5
    rw = max(w * scale, min_size)
    rh = max(h * scale, min_size)
    x0 = int(max(0, cx - rw * 0.5))
    y0 = int(max(0, cy - rh * 0.5))
    x1 = int(min(w_img, cx + rw * 0.5))
    y1 = int(min(h_img, cy + rh * 0.5))
    return x0, y0, x1, y1


def get_roi_predictions(model, img_bgr, proposals_xywh, config: Dict[str, Any]):
    """Run YOLO only on ROI crops around proposals."""
    if model is None or not proposals_xywh:
        return []

    h, w, _ = img_bgr.shape
    crops = []
    offsets = []

    # Take top N proposals
    use_props = proposals_xywh[: min(len(proposals_xywh), config["max_rois"])]

    for b in use_props:
        x0, y0, x1, y1 = _expand_roi_xywh(
            b, w, h, scale=config["roi_scale"], min_size=config["min_roi_size"]
        )
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((x0, y0))

    if len(crops) == 0:
        return []

    # Batch Inference
    results = model(
        crops,
        imgsz=config["img_size"],
        verbose=False,
        conf=config["conf_thresh"],
        classes=config["model_classes"],
    )

    all_boxes = []
    all_scores = []

    for j, res in enumerate(results):
        boxes = res.boxes
        if len(boxes) > 0:
            local_boxes = boxes.xyxy.cpu()
            local_scores = boxes.conf.cpu()
            x_off, y_off = offsets[j]

            shifted_boxes = local_boxes.clone()
            shifted_boxes[:, 0] += x_off
            shifted_boxes[:, 1] += y_off
            shifted_boxes[:, 2] += x_off
            shifted_boxes[:, 3] += y_off

            all_boxes.append(shifted_boxes)
            all_scores.append(local_scores)

    if not all_boxes:
        return []

    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)

    keep_indices = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
    final_boxes = pred_boxes[keep_indices]
    final_scores = pred_scores[keep_indices]

    final_preds = []
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box.tolist()
        score = float(final_scores[i])
        final_preds.append(
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1), score]
        )

    return final_preds


# Global model cache for worker processes
_WORKER_MODEL = None


def load_worker_model(model_name):
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = YOLO(model_name)
    return _WORKER_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 2.
    """
    video_path, config, gt_data = args
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])
    if YOLO is None:
        raise ImportError("ultralytics library missing")

    model = load_worker_model(config["model_name"])

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
    prev_gray = None
    obj_tracker = vis_utils.ObjectTracker(
        dist_thresh=50, max_frames_to_skip=5, min_hits=config["min_hits"]
    )
    use_sahi = config.get("use_sahi", False)

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

        raw_detections = []
        if use_sahi:
            raw_detections = vis_utils.get_sahi_predictions(model, frame, config)
        else:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_img, w_img = curr_gray.shape

            proposals = []
            if prev_gray is not None:
                # 1. GMC: Align frames
                warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                if warped_prev is not None:
                    # 2. Dynamic Thresholding (Notebook Logic)
                    diff = cv2.absdiff(curr_gray, warped_prev)
                    mean, std = cv2.meanStdDev(diff)
                    dynamic_thresh = (
                        mean[0][0] + config["dynamic_multiplier"] * std[0][0]
                    )
                    final_thresh = max(
                        config["min_threshold"],
                        min(config["max_threshold"], dynamic_thresh),
                    )
                    _, thresh = cv2.threshold(
                        diff, final_thresh, 255, cv2.THRESH_BINARY
                    )

                    # 3. Morphological Opening (Notebook Logic)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.dilate(thresh, kernel, iterations=1)

                    # 4. Contour Filtering (Notebook Logic)
                    contours, _ = cv2.findContours(
                        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if 50 < area < 5000:
                            x, y, w, h = cv2.boundingRect(cnt)
                            aspect_ratio = float(w) / h
                            if 0.2 < aspect_ratio < 4.0:
                                border = 15
                                if (
                                    x > border
                                    and y > border
                                    and (x + w) < (w_img - border)
                                    and (y + h) < (h_img - border)
                                ):
                                    proposals.append([x, y, w, h])

            # 5. YOLO ROI Refiner
            if proposals:
                raw_detections = get_roi_predictions(model, frame, proposals, config)

            # Update for next frame
            prev_gray = curr_gray

        # 6. Persistence Tracking
        final_preds = obj_tracker.update(raw_detections)

        # --- EVALUATION ---
        key = f"{video_name}/{os.path.basename(img_path)}"
        gts = gt_data.get(key, [])

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
                img_tp += 1
                vid_tp += 1
                vid_dotd_list.append(best_dist)
                matched_gt.add(best_idx)
            else:
                img_fp += 1
                vid_fp += 1

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
                # p_box is [x,y,w,h,score]
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

        # Collect Result
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
    fps = len(images) / vid_time if vid_time > 0 else 0
    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
    vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

    return {
        "video_name": video_name,
        "n_frames": len(images),
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


@register_pipeline("strategy_2")
def run_strategy_2_pipeline(config: Dict[str, Any]):
    """Execute Strategy 2 pipeline: GMC + Dynamic Threshold + YOLO Refiner."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 2 (PARALLEL): {pipeline_name} ---")

    if YOLO is None:
        logger.error(
            "❌ ultralytics library not found. Please run: pip install ultralytics"
        )
        raise ImportError("ultralytics library missing")

    logger.info(f"⏳ Loading YOLO Model: {config['model_name']}...")
    try:
        # Model loading is now handled by the worker function, but we still check here
        # to ensure the model name is valid and ultralytics is installed.
        # We don't actually load it into the main process.
        _ = YOLO(config["model_name"])
        logger.info(f"✅ Model {config['model_name']} check passed.")
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


"""
Strategy 8 Pipeline: YOLO on ROIs (Region of Interest)

Implements efficient detection using:
- Motion compensation for proposal generation
- YOLO inference only on ROI crops
- Configurable detection frequency
- Optional full-frame processing at intervals
"""


# Attempt to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


logger = logging.getLogger(__name__)


def _expand_roi_xywh(box, w_img, h_img, scale=2.0, min_size=256):
    """Expand ROI with minimum size constraint."""
    x, y, w, h = box
    cx = x + w * 0.5
    cy = y + h * 0.5
    rw = max(w * scale, min_size)
    rh = max(h * scale, min_size)
    x0 = int(max(0, cx - rw * 0.5))
    y0 = int(max(0, cy - rh * 0.5))
    x1 = int(min(w_img, cx + rw * 0.5))
    y1 = int(min(h_img, cy + rh * 0.5))
    return x0, y0, x1, y1


def get_roi_predictions(
    model, img_bgr, proposals_xywh, config: Dict[str, Any], frame_idx: int
):
    """Run YOLO only on ROI crops around proposals."""
    if model is None:
        return []

    h, w, _ = img_bgr.shape
    crops = []
    offsets = []

    use_props = proposals_xywh[: min(len(proposals_xywh), config["max_rois"])]

    for b in use_props:
        x0, y0, x1, y1 = _expand_roi_xywh(
            b, w, h, scale=config["roi_scale"], min_size=config["min_roi_size"]
        )
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((x0, y0))

    # Optional full-frame pass
    if config["fullframe_every"] and (frame_idx % config["fullframe_every"] == 0):
        crops.append(img_bgr)
        offsets.append((0, 0))

    if len(crops) == 0:
        return []

    # Run Inference on List of Crops
    results = model(
        crops,
        imgsz=config["img_size"],
        verbose=False,
        conf=config["conf_thresh"],
        classes=config["model_classes"],
    )

    all_boxes = []
    all_scores = []

    for j, res in enumerate(results):
        boxes = res.boxes
        if len(boxes) > 0:
            # Transfer to CPU
            local_boxes = boxes.xyxy.cpu()
            local_scores = boxes.conf.cpu()

            x_off, y_off = offsets[j]

            # Apply offset to get back to full frame coordinates
            shifted_boxes = local_boxes.clone()
            shifted_boxes[:, 0] += x_off
            shifted_boxes[:, 1] += y_off
            shifted_boxes[:, 2] += x_off
            shifted_boxes[:, 3] += y_off

            all_boxes.append(shifted_boxes)
            all_scores.append(local_scores)

    if not all_boxes:
        return []

    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)

    # Standard NMS to merge overlapping ROI detections
    keep_indices = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
    final_boxes = pred_boxes[keep_indices]
    final_scores = pred_scores[keep_indices]

    final_preds = []
    for i, box in enumerate(final_boxes):
        x1, y1, x2, y2 = box.tolist()
        score = float(final_scores[i])
        final_preds.append(
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1), score]
        )

    return final_preds


_WORKER_MODEL = None


def load_worker_model(model_name):
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _WORKER_MODEL = YOLO(model_name)
    return _WORKER_MODEL


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 8.
    """
    video_path, config, gt_data = args
    # Configure logging for worker process
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

    model = load_worker_model(config["model_name"])

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
    prev_gray = None
    obj_tracker = vis_utils.ObjectTracker(
        dist_thresh=50, max_frames_to_skip=4, min_hits=2
    )
    use_sahi = config.get("use_sahi", False)

    for i, img_path in enumerate(images):
        img_start_time = time.time()

        if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
            percent = ((i + 1) / n_frames) * 100
            logger.info(
                f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
            )

        frame = cv2.imread(img_path)
        if frame is None:
            # Create an empty image_result for a missing frame
            image_results.append(
                csv_utils.create_image_result(
                    video_name=video_name,
                    frame_name=os.path.basename(img_path),
                    image_path=img_path,
                    predictions=[],
                    ground_truths=[],
                    tp=0,
                    fp=0,
                    fn=0,
                    processing_time_sec=(time.time() - img_start_time),
                    iou=0.0,
                    memory_usage_mb=vis_utils.get_memory_usage(),
                )
            )
            continue  # skip to next image

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_detections = []
        frame_was_detected = (
            False  # Flag to track if detection was attempted for this frame
        )

        if (
            i % config["detect_every"] == 0
        ):  # Only run detection logic on "detect_every" frames
            frame_was_detected = True
            if use_sahi:
                raw_detections = vis_utils.get_sahi_predictions(model, frame, config)
            else:
                if prev_gray is not None:
                    warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                    if warped_prev is not None:
                        # Simplified motion detection for proposals
                        diff = cv2.absdiff(curr_gray, warped_prev)
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3)
                        thresh = cv2.dilate(thresh, k3, iterations=2)

                        contours, _ = cv2.findContours(
                            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )

                        proposals = []
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if 50 < area < 5000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                proposals.append([x, y, w, h])

                        # Run YOLO on ROIs
                        if len(proposals) > 0 or (
                            config["fullframe_every"]
                            and i % config["fullframe_every"] == 0
                        ):
                            raw_detections = get_roi_predictions(
                                model, frame, proposals, config, frame_idx=i
                            )
        prev_gray = curr_gray  # Always update prev_gray for motion compensation

        # Tracking
        # The tracker is updated regardless, so it can maintain internal state.
        # However, for metric calculation for skipped frames, we set final_preds explicitly to []
        # to ensure no FNs are counted if no detection was intended.
        tracker_output_for_frame = obj_tracker.update(raw_detections)

        # Evaluation
        key = f"{video_name}/{os.path.basename(img_path)}"
        gts = gt_data.get(key, [])  # Get GTs for the current frame

        final_preds = []
        img_tp = 0
        img_fp = 0
        img_fn = 0
        img_avg_iou = 0.0

        if frame_was_detected:  # This frame was chosen for detection
            final_preds = tracker_output_for_frame  # Use predictions from tracker

            # Store for mAP calc (only if detection was active)
            vid_all_preds.append(final_preds)
            vid_all_gts.append(gts)

            matched_gt = set()

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
        else:  # This is a skipped frame, no detection was performed or intended for evaluation
            # For skipped frames, we explicitly ensure metrics are zero.
            # No predictions are "made" from the perspective of this pipeline.
            final_preds = []  # No predictions for skipped frames

            # Store for mAP calc. Even if no detections, we need a corresponding entry for GTs.
            # If final_preds is empty, mAP calculation will naturally handle this.
            vid_all_preds.append(final_preds)  # Empty predictions for skipped frame
            vid_all_gts.append(
                gts
            )  # GTs exist, but won't contribute to TP/FP/FN for this frame.

            # Crucially, for skipped frames, no FNs are counted.
            img_tp = 0
            img_fp = 0
            img_fn = 0
            # Note: vid_tp, vid_fp, vid_fn are not incremented for skipped frames.
            # This aligns with the "don't count FNs for skipped frames" requirement.

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
    # n_frames here still means total frames, which is correct.
    # fps calculation uses n_frames and vid_time, which is good.
    # Overall prec/rec/f1 uses accumulated vid_tp, vid_fp, vid_fn which is now correctly handled.
    fps = n_frames / vid_time if vid_time > 0 else 0
    prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
    rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # Calculate mAP and DotD for video
    # vid_all_preds and vid_all_gts are populated for all frames, with predictions being empty for skipped ones.
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


@register_pipeline("strategy_8")
def run_strategy_8_pipeline(config: Dict[str, Any]):
    """Execute Strategy 8 pipeline with YOLO on ROIs."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 8 (PARALLEL): {pipeline_name} ---")

    # Check dependencies
    if YOLO is None:
        logger.error(
            "❌ ultralytics library not found. Please run: pip install ultralytics"
        )
        raise ImportError("ultralytics library missing")

    # Load model (check in main process)
    logger.info(f"⏳ Loading Model: {config['model_name']}...")
    try:
        _ = YOLO(config["model_name"])
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
