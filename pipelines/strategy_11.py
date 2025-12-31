"""
Strategy 11 Pipeline: ROI Selection + YOLO Classification Filter + YOLO Detection

Implements efficient detection by:
1. Generating motion-based ROI proposals (from Strategy 8).
2. Filtering these ROIs through a lightweight YOLO classifier.
3. Running full YOLO detection only on ROIs that passed the classifier.
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
from typing import Dict, Any

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

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
