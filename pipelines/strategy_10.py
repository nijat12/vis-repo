"""
Strategy 10 Pipeline: Motion-Gated Native Tiling + YOLO

This pipeline implements a high-precision hybrid approach:
1. Global Motion Compensation (GMC) to stabilize the background.
2. Motion Gating: Divide the frame into 640x640 native tiles.
3. Active Tile Selection: Only process tiles with significant detected motion.
4. Native Inference: Run YOLO on active tiles without ANY resizing to maintain pixel accuracy.
"""

import os
import glob
import time
import datetime
import logging
from typing import Dict, Any
import cv2
import torch
import torchvision
import numpy as np
import pandas as pd

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
                    classes=[config["bird_class_id"]],
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
