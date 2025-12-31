"""
Strategy 8 Pipeline: YOLO on ROIs (Region of Interest)

Implements efficient detection using:
- Motion compensation for proposal generation
- YOLO inference only on ROI crops
- Configurable detection frequency
- Optional full-frame processing at intervals
"""

import os
import glob
import time
import datetime
import sys
import logging
from typing import Dict, Any

import cv2
import torch
import torchvision
import numpy as np
import pandas as pd

# Attempt to import ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

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


@register_pipeline("strategy_8")
def run_strategy_8_pipeline(config: Dict[str, Any]):
    """Execute Strategy 8 pipeline with YOLO on ROIs."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(f"pipelines.{pipeline_name}")
    logger.info(f"--- STARTING STRATEGY 8: {pipeline_name} ---")

    # Check dependencies
    if YOLO is None:
        logger.error(
            "❌ ultralytics library not found. Please run: pip install ultralytics"
        )
        raise ImportError("ultralytics library missing")

    # Load model
    logger.info(f"⏳ Loading Model: {config['model_name']}...")
    try:
        model = YOLO(config["model_name"])
        logger.info(f"✅ Model {config['model_name']} Loaded.")
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

    logger.info(f"📂 Found {len(video_folders)} videos. Starting...")

    # Initialize results tracker
    tracker = csv_utils.get_results_tracker()

    total_tp = total_fp = total_fn = total_time = total_frames = 0
    total_map_sum = 0.0
    total_dotd_sum = 0.0
    total_videos_processed = 0
    results_data = []

    for video_path in video_folders:
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not images:
            continue

        vid_tp = vid_fp = vid_fn = 0
        vid_dotd_list = []
        vid_all_preds = []
        vid_all_gts = []

        vid_start = time.time()
        n_frames = len(images)
        prev_gray = None
        obj_tracker = vis_utils.ObjectTracker(
            dist_thresh=50, max_frames_to_skip=4, min_hits=2
        )
        use_sahi = config.get("use_sahi", False)

        for i, img_path in enumerate(images):
            # Only run detection every N frames
            if i % config["detect_every"] != 0:
                continue
            
            img_start_time = time.time()  # Track per-image time

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
                prev_gray = curr_gray

            # Tracking
            final_preds = obj_tracker.update(raw_detections)

            # Evaluation
            key = f"{video_name}/{os.path.basename(img_path)}"
            gts = gt_data.get(key, [])  # Should be list of [x,y,w,h]

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

            # Save per-image result
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
            tracker.add_image_result(pipeline_name, image_result)

            if (i + 1) % 50 == 0:
                tracker.save_batch(pipeline_name, batch_size=50)

        vid_time = time.time() - vid_start
        fps = len(images) / vid_time if vid_time > 0 else 0
        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Log video metrics using standard utility
        # Aggregate from detailed data for the video
        p_data = [
            d
            for d in tracker.detailed_data.get(pipeline_name, [])
            if d["video"] == video_name
        ]
        vid_iou = np.mean([d["iou"] for d in p_data]) if p_data else 0.0
        vid_mem = np.mean([d["memory_usage_mb"] for d in p_data]) if p_data else 0.0

        # Calculate mAP and DotD for video
        vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
        vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

        vis_utils.log_video_metrics(
            logger,
            video_name,
            {
                "n_frames": len(images),
                "fps": fps,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "tp": vid_tp,
                "fp": vid_fp,
                "fn": vid_fn,
                "iou": vid_iou,
                "mAP": vid_map,
                "dotd": vid_dotd,
                "memory_usage_mb": vid_mem,
                "vid_time": vid_time,
            },
        )

        total_map_sum += vid_map
        total_dotd_sum += vid_dotd
        total_videos_processed += 1

        results_data.append(
            {
                "Video": video_name,
                "Frames": len(images),
                "FPS": round(fps, 2),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "TP": vid_tp,
                "FP": vid_fp,
                "FN": vid_fn,
                "Video_Time": vid_time,
            }
        )
        total_time += vid_time
        total_frames += len(images)
        total_tp += vid_tp
        total_fp += vid_fp
        total_fn += vid_fn

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
