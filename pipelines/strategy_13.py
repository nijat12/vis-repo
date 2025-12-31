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

import os
import glob
import time
import datetime
import logging
import cv2
import torch
import torchvision
import numpy as np
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


def get_native_slices(img_h, img_w, slice_wh=(640, 640), overlap_ratio=0.2):
    """Generates coordinates for overlapping tiles."""
    slice_w, slice_h = slice_wh
    step_x = int(slice_w * (1 - overlap_ratio))
    step_y = int(slice_h * (1 - overlap_ratio))
    x_points = sorted(list(set(range(0, img_w - slice_w, step_x)) | {img_w - slice_w}))
    y_points = sorted(list(set(range(0, img_h - slice_h, step_y)) | {img_h - slice_h}))
    return [(x, y, x + slice_w, y + slice_h) for y in y_points for x in x_points]


@register_pipeline("strategy_13")
def run_strategy_13_pipeline(config: Dict[str, Any]):
    """Execute Strategy 13: Motion-Gated Classifier Funnel."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(f"pipelines.{pipeline_name}")
    logger.info(f"--- STARTING STRATEGY 13: {pipeline_name} ---")

    if YOLO is None:
        raise ImportError("❌ ultralytics library not found.")

    logger.info(f"⏳ Loading Detector: {config['model_name']}...")
    logger.info(f"⏳ Loading Classifier: {config['classifier_model_name']}...")
    try:
        det_model = YOLO(config["model_name"])
        cls_model = YOLO(config["classifier_model_name"])
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

    for video_path in video_folders:
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not images:
            continue

        vid_tp, vid_fp, vid_fn, n_frames = 0, 0, 0, len(images)
        vid_start_time = time.time()

        all_predictions = defaultdict(list)
        last_keyframe_preds: List[List[float]] = []
        last_keyframe_idx = -1
        prev_gray = None

        use_interpolation = config.get("use_interpolation", False)
        detect_every = config.get("detect_every", 1) if use_interpolation else 1
        use_sahi = config.get("use_sahi", False)

        first_frame = cv2.imread(images[0])
        h_img, w_img = first_frame.shape[:2]
        tile_coords = get_native_slices(
            h_img, w_img, (config["img_size"], config["img_size"]), 0.2
        )

        images_w_metadata = [
            {
                "img_start_time": time.time(),
                "image": image,
            }
            for image in images
        ]

        # --- Pass 1: Generate Detections ---
        for i, img_metadata in enumerate(images_w_metadata):
            img_metadata["img_start_time"] = time.time()
            if i % detect_every == 0:
                frame = cv2.imread(img_metadata["image"])
                if frame is None:
                    continue

                current_preds = []
                if use_sahi:
                    current_preds = vis_utils.get_sahi_predictions(
                        det_model, frame, config
                    )
                else:
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    active_tiles, active_offsets = [], []
                    if prev_gray is not None:
                        warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                        if warped_prev is not None:
                            diff = cv2.absdiff(curr_gray, warped_prev)
                            mean, std = cv2.meanStdDev(diff)
                            motion_thresh = max(
                                20,
                                min(
                                    80,
                                    mean[0][0]
                                    + config["motion_thresh_scale"] * std[0][0],
                                ),
                            )
                            _, motion_mask = cv2.threshold(
                                diff, motion_thresh, 255, cv2.THRESH_BINARY
                            )

                            cls_cand, cls_offs = [], []
                            for x1, y1, x2, y2 in tile_coords:
                                if cv2.countNonZero(
                                    motion_mask[y1:y2, x1:x2]
                                ) > config.get("motion_pixel_threshold", 20):
                                    active_tiles.append(frame[y1:y2, x1:x2])
                                    active_offsets.append((x1, y1))
                                else:
                                    cls_cand.append(frame[y1:y2, x1:x2])
                                    cls_offs.append((x1, y1))

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
                                        and r.probs.top1conf
                                        >= config["cls_conf_thresh"]
                                    ):
                                        active_tiles.append(cls_cand[j])
                                        active_offsets.append(cls_offs[j])
                    prev_gray = curr_gray

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
                            current_preds = [
                                [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                                for x1, y1, x2, y2 in pred_boxes[keep]
                            ]

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
        for i, img_metadata in enumerate(images_w_metadata):
            final_preds = all_predictions[i]
            key = f"{video_name}/{os.path.basename(img_metadata['image'])}"
            gts = gt_data.get(key, [])

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
                    img_ious.append(best_iou)
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
                mAP=0.0,
                memory_usage_mb=img_mem,
            )
            results_tracker.add_image_result(
                pipeline_name,
                image_result,
            )

        vid_time = time.time() - vid_start_time
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
                "iou": vid_iou,
                "mAP": 0.0,
                "memory_usage_mb": vid_mem,
                "vid_time": vid_time,
            },
        )
        total_time += vid_time
        total_frames += n_frames
        total_tp += vid_tp
        total_fp += vid_fp
        total_fn += vid_fn

    # Final Summary
    avg_fps = total_frames / total_time if total_time > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = (
        2 * (overall_prec * overall_rec) / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0
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
        "processing_time_sec": total_time,
        "execution_time_sec": time.time() - start_time,
    }
    vis_utils.log_pipeline_summary(logger, pipeline_name, summary_metrics)
    results_tracker.update_summary(pipeline_name, summary_metrics)
    return {"pipeline": pipeline_name, "status": "completed", **summary_metrics}
