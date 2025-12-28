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
import math
from collections import defaultdict

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
    results = model(img, imgsz=img_size, verbose=False, conf=conf_thresh, classes=classes)
    
    final_preds = []
    if len(results) > 0:
        boxes = results[0].boxes
        if len(boxes) > 0:
            # Convert to xywh format [x, y, w, h]
            # Use cpu() and numpy() for consistent format
            xyxy_boxes = boxes.xyxy.cpu().numpy()
            for box in xyxy_boxes:
                x1, y1, x2, y2 = box
                final_preds.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])
                
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
        results = model(sub_crops, imgsz=img_size, verbose=False, conf=conf_thresh, classes=classes)

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
        # Apply Global NMS (Necessary because we stitched tiles)
        # We use a strict IoU threshold here to merge duplicates at tile boundaries
        keep_indices = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
        final_tensor = pred_boxes[keep_indices]
    else:
        # Just use all boxes without NMS
        final_tensor = pred_boxes

    # Convert to xywh format [x, y, w, h]
    final_preds = []
    final_tensor = final_tensor.numpy() # Convert to numpy for list building
    for box in final_tensor:
        x1, y1, x2, y2 = box
        final_preds.append([float(x1), float(y1), float(x2-x1), float(y2-y1)])

    return final_preds


@register_pipeline("baseline")
def run_all_baseline_variants():
    """
    Master orchestrator for all baseline variants.
    Runs the base case, tiling only, and tiling+nms sequentially.
    """
    logger = logging.getLogger(__name__)
    logger.info("🚀 STARTING ALL BASELINE VARIANTS")
    
    # 1. Base YOLO Inference
    _run_baseline_variant("baseline_base", use_tiling=False, use_nms=False)
    
    # 2. Tiling only (no NMS)
    _run_baseline_variant("baseline_w_tiling", use_tiling=True, use_nms=False)
    
    # 3. Full Baseline (Tiling + NMS)
    results = _run_baseline_variant("baseline_w_tiling_and_nms", use_tiling=True, use_nms=True)
    
    return results


def _run_baseline_variant(pipeline_name: str, use_tiling: bool, use_nms: bool):
    """
    Core logic for running a specific baseline variant.
    """
    logger = logging.getLogger(f"pipelines.{pipeline_name}")
    logger.info(f"--- STARTING VARIANT: {pipeline_name} ---")
    
    # Load configuration
    cfg = Config.get_pipeline_config(pipeline_name)
    MODEL_NAME = cfg["model_name"]
    IMG_SIZE = cfg["img_size"]
    CONF_THRESH = cfg["conf_thresh"]
    model_classes = cfg["model_classes"]
    
    # Check dependencies
    if YOLO is None:
        logger.error("❌ ultralytics library not found. Please run: pip install ultralytics")
        raise ImportError("ultralytics library missing")

    # Load model
    logger.info(f"⏳ Loading Model: {MODEL_NAME} for {pipeline_name}...")
    try:
        model = YOLO(MODEL_NAME)
        logger.info(f"✅ Model {MODEL_NAME} Loaded.")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    # Load ground truth
    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()

    # Select videos to process
    video_folders = sorted(glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, '*')))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [video_folders[i] for i in Config.VIDEO_INDEXES]
        else:
            video_folders = video_folders[:min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)]

    if not video_folders:
        raise RuntimeError(f"No video folders found in {Config.LOCAL_TRAIN_DIR}")

    logger.info(f"📂 Found {len(video_folders)} videos. Starting variant {pipeline_name}...")

    # Initialize results tracker
    tracker = csv_utils.get_results_tracker()

    total_tp = total_fp = total_fn = total_time_sec = total_frames = 0
    results_data = []

    for v_idx, video_path in enumerate(video_folders):
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        if not images:
            continue

        vid_tp = vid_fp = vid_fn = 0
        vid_start = time.time()
        n_frames = len(images)

        for i, img_path in enumerate(images):
            img_start_time = time.time()  # Track per-image time
            
            if i % 20 == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")

            img = cv2.imread(img_path)
            if img is None:
                continue

            # Select prediction method
            if use_tiling:
                preds = get_tiled_predictions(model, img, IMG_SIZE, CONF_THRESH, model_classes, use_nms=use_nms)
            else:
                preds = get_base_predictions(model, img, IMG_SIZE, CONF_THRESH, model_classes)

            img_filename = os.path.basename(img_path)
            key = f"{video_name}/{img_filename}"
            gts = gt_data.get(key, [])

            # Track per-image results
            img_tp = img_fp = 0
            matched_gt = set()
            for p_box in preds:
                best_dist = 10000
                best_idx = -1
                for g_idx, g_box in enumerate(gts):
                    if g_idx in matched_gt:
                        continue
                    d = vis_utils.calculate_center_distance(p_box, g_box)
                    if d < best_dist:
                        best_dist = d
                        best_idx = g_idx

                if best_dist <= 30:
                    vid_tp += 1
                    img_tp += 1
                    matched_gt.add(best_idx)
                else:
                    vid_fp += 1
                    img_fp += 1
            
            img_fn = len(gts) - len(matched_gt)
            vid_fn += img_fn
            
            # Calculate IoU for matched pairs
            img_ious = []
            matched_gt_indices = set()
            for p_box in preds:
                best_iou = 0
                best_idx = -1
                for g_idx, g_box in enumerate(gts):
                    if g_idx in matched_gt_indices:
                        continue
                    iou = vis_utils.box_iou_xywh(p_box, g_box)
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
            
            # Save per-image result
            image_result = csv_utils.create_image_result(
                video_name=video_name,
                frame_name=img_filename,
                image_path=img_path,
                predictions=preds,
                ground_truths=gts,
                tp=img_tp,
                fp=img_fp,
                fn=img_fn,
                processing_time_sec=img_processing_time,
                iou=img_avg_iou, mAP=0.0, memory_usage_mb=img_mem
            )
            tracker.add_image_result(pipeline_name, image_result)
            
            # Save batch every 50 images
            if (i + 1) % 50 == 0:
                tracker.save_batch(pipeline_name, batch_size=50)

        vid_end = time.time()
        vid_time = vid_end - vid_start
        vid_fps = n_frames / vid_time if vid_time > 0 else 0

        total_time_sec += vid_time
        total_frames += n_frames
        total_tp += vid_tp
        total_fp += vid_fp
        total_fn += vid_fn

        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Log video metrics using standard utility
        # Aggregate from detailed data for the video
        p_data = [d for d in tracker.detailed_data.get(pipeline_name, []) if d['video'] == video_name]
        vid_iou = np.mean([d['iou'] for d in p_data]) if p_data else 0.0
        vid_mem = np.mean([d['memory_usage_mb'] for d in p_data]) if p_data else 0.0
        
        vis_utils.log_video_metrics(logger, video_name, {
            'n_frames': n_frames,
            'fps': vid_fps,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'tp': vid_tp,
            'fp': vid_fp,
            'fn': vid_fn,
            'iou': vid_iou,
            'mAP': 0.0,
            'memory_usage_mb': vid_mem,
            'vid_time': vid_time
        })

    # Calculate overall metrics
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    # Aggregate additional metrics from detailed data
    p_data = tracker.detailed_data.get(pipeline_name, [])
    overall_iou = np.mean([d['iou'] for d in p_data]) if p_data else 0.0
    overall_mem = np.mean([d['memory_usage_mb'] for d in p_data]) if p_data else 0.0

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
        "mAP": 0.0,
        "memory_usage_mb": overall_mem,
        "processing_time_sec": total_time_sec,
        "execution_time_sec": time.time() - start_time
    }

    # Log summary using standard utility
    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)

    # Update results tracker with summary metrics
    tracker.update_summary(pipeline_name, summary_metrics)

    return {
        "pipeline": pipeline_name,
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time,
    }