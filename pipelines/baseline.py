"""
Baseline Pipeline: YOLO with 4x3 Tiled Inference

This pipeline implements the baseline strategy using:
- YOLO pretrained model (Upgraded from YOLO)
- 4x3 grid tiling with overlap for better small object detection
- Batch inference optimization
- Center distance matching for evaluation
"""

import os
import glob
import time
import datetime
import sys
import logging
import concurrent.futures
import math
from collections import defaultdict
from typing import Dict, Any

import cv2
import torch
import torchvision
import pandas as pd
import numpy as np

# Attempt to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline


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
def run_baseline_base(config: Dict[str, Any]):
    """Runs the baseline variant with no tiling."""
    return _run_baseline_variant(config, use_tiling=False, use_nms=False)


@register_pipeline("baseline_w_tiling")
def run_baseline_w_tiling(config: Dict[str, Any]):
    """Runs the baseline variant with tiling but no NMS."""
    return _run_baseline_variant(config, use_tiling=True, use_nms=False)


@register_pipeline("baseline_w_tiling_and_nms")
def run_baseline_w_tiling_and_nms(config: Dict[str, Any]):
    """Runs the baseline variant with tiling and NMS."""
    return _run_baseline_variant(config, use_tiling=True, use_nms=True)


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

    use_sahi = config.get("use_sahi", False)
    # Check if tiling is needed (passed via config or inferred from pipeline variant)
    # The variant functions (run_baseline_w_tiling etc) pass this config
    # We need to reconstruct 'use_tiling' and 'use_nms' or pass them in config.
    # The original _run_baseline_variant had these as args.
    # We should add them to config in the caller or infer here.
    # For simplicity, let's assume config has them or we infer from pipeline logic if needed.
    # Use config keys if available, else standard baseline defaults

    # Actually, the original code passed use_tiling/use_nms to _run_baseline_variant.
    # We need to make sure these are in the config dict passed to this worker.
    use_tiling = config.get("use_tiling", False)
    use_nms = config.get("use_nms", False)

    for i, img_path in enumerate(images):
        img_start_time = time.time()

        if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
            percent = ((i + 1) / n_frames) * 100
            logger.info(
                f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
            )

        # We can't log easily to the main logger from here without setup,
        # so we skip per-frame logging or use print/custom log queue if needed.
        # For now, silence per-frame logs or print only on error.

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


def _run_baseline_variant(config: Dict[str, Any], use_tiling: bool, use_nms: bool):
    """
    Core logic for running a specific baseline variant.
    PARALLELIZED VERSION
    """
    pipeline_name = config["run_name"]
    logger = logging.getLogger(f"{pipeline_name}")
    logger.info(f"--- STARTING VARIANT: {pipeline_name} ---")

    # Add variant flags to config for worker
    config["use_tiling"] = use_tiling
    config["use_nms"] = use_nms

    MODEL_NAME = config["model_name"]
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
    # We can pass the whole gt_data or subset. Passing whole dict is cleaner if it's not massive (40k entries is fine for copy-on-write usually)
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
                        # We calculate IoU/Mem from detailed results if needed or approximate
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
