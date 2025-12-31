"""
Strategy 12 Pipeline: GMC + Interpolation

This pipeline builds on Strategy 2 by adding frame skipping and interpolation.
1. Global Motion Compensation (GMC) to align frames.
2. Runs detection only every N frames (`detect_every`).
3. For intermediate frames, it interpolates bounding boxes linearly between keyframes.
"""

import os
import glob
import time
import datetime
import logging
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from collections import defaultdict

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

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


@register_pipeline("strategy_12")
def run_strategy_12_pipeline(config: Dict[str, Any]):
    """Execute Strategy 12: GMC + Frame Skipping + Interpolation."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(f"pipelines.{pipeline_name}")
    logger.info(f"--- STARTING STRATEGY 12: {pipeline_name} ---")

    if YOLO is None:
        raise ImportError("❌ ultralytics library not found.")

    logger.info(f"⏳ Loading YOLO Model: {config['model_name']}...")
    try:
        model = YOLO(config["model_name"])
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

    logger.info(f"📂 Found {len(video_folders)} videos. Starting processing...")
    results_tracker = csv_utils.get_results_tracker()
    total_tp = total_fp = total_fn = total_time = total_frames = 0
    total_map_sum = 0.0
    total_dotd_sum = 0.0
    total_videos_processed = 0

    for video_path in video_folders:
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not images:
            continue

        vid_tp = vid_fp = vid_fn = 0  # Re-initializing cleaner
        n_frames = len(images)
        vid_start_time = time.time()

        vid_dotd_list = []
        vid_all_preds = []
        vid_all_gts = []

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
                            kernel = cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (5, 5)
                            )
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
                        current_preds = get_roi_predictions(
                            model, frame, proposals, config
                        )
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

            img_tp, img_fp, matched_gt = 0, 0, set()
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

                if best_dist <= 30 and best_iou > 0:
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
                mAP=0.0,
                memory_usage_mb=img_mem,
            )
            results_tracker.add_image_result(pipeline_name, image_result)

        # Video-level stats
        vid_time = time.time() - vid_start_time
        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Calculate mAP and DotD for video
        vid_map = vis_utils.calculate_video_map(vid_all_preds, vid_all_gts)
        vid_dotd = vis_utils.calculate_avg_dotd(vid_dotd_list)

        vis_utils.log_video_metrics(
            logger,
            video_name,
            {
                "n_frames": n_frames,
                "fps": n_frames / vid_time,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "tp": vid_tp,
                "fp": vid_fp,
                "fn": vid_fn,
                "iou": np.mean(img_ious) if img_ious else 0.0,  # Approximate avg IoU
                "mAP": vid_map,
                "dotd": vid_dotd,
                "vid_time": vid_time,
                "memory_usage_mb": 0.0,  # Not easily tracked here per video avg comfortably
            },
        )
        total_map_sum += vid_map
        total_dotd_sum += vid_dotd
        total_videos_processed += 1

        total_time += vid_time
        total_frames += n_frames
        total_tp += vid_tp
        total_fp += vid_fp
        total_fn += vid_fn

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

    summary_metrics = {
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "mAP": overall_map,
        "dotd": overall_dotd,
        "processing_time_sec": total_time,
        "execution_time_sec": time.time() - start_time,
    }
    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)
    results_tracker.update_summary(pipeline_name, summary_metrics)
    return {"pipeline": pipeline_name, "status": "completed", **summary_metrics}
