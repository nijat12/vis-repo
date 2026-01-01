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
