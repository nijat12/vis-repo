"""
Strategy 7 Pipeline: Motion Compensation + CNN Verifier

Implements advanced detection using:
- Global motion compensation with optical flow
- Hysteresis thresholding with high-pass filtering
- MobileNetV3-Small CNN for birdness verification
- Object tracking for temporal consistency
- DoG blob detection for tiny birds
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
import numpy as np
import pandas as pd
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

logger = logging.getLogger(__name__)

# Load MobileNetV3 model once (module-level)
# When using ProcessPoolExecutor (spawn/fork), this script is imported by workers,
# initializing a fresh copy of the model for each worker process.
_STRAT7_WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT
_strat7_model = mobilenet_v3_small(weights=_STRAT7_WEIGHTS).eval()
_strat7_model.to("cpu")
torch.set_num_threads(4)
_strat7_preprocess = _STRAT7_WEIGHTS.transforms()
_strat7_categories = _STRAT7_WEIGHTS.meta["categories"]

# Build bird class indices
_bird_keywords = {
    "bird",
    "sparrow",
    "finch",
    "warbler",
    "oriole",
    "blackbird",
    "robin",
    "jay",
    "magpie",
    "eagle",
    "hawk",
    "falcon",
    "vulture",
    "owl",
    "woodpecker",
    "kingfisher",
    "hummingbird",
    "parrot",
    "macaw",
    "cockatoo",
    "lorikeet",
    "peacock",
    "crane",
    "heron",
    "stork",
    "flamingo",
    "pelican",
    "gull",
    "tern",
    "albatross",
    "duck",
    "goose",
    "swan",
    "chicken",
    "hen",
    "cock",
    "rooster",
    "turkey",
    "ptarmigan",
    "partridge",
    "quail",
    "ostrich",
    "emu",
    "kiwi",
}
_exclude = {"kite"}
_strat7_bird_indices = [
    i
    for i, name in enumerate(_strat7_categories)
    if (
        any(k in name.lower() for k in _bird_keywords)
        and not any(x in name.lower() for x in _exclude)
    )
]


def _expand_box_xywh(box, w_img, h_img, scale=2.0):
    """Expand bounding box by scale factor."""
    x, y, w, h = box
    cx = x + w * 0.5
    cy = y + h * 0.5
    nw = w * scale
    nh = h * scale
    x0 = int(max(0, cx - nw * 0.5))
    y0 = int(max(0, cy - nh * 0.5))
    x1 = int(min(w_img, cx + nw * 0.5))
    y1 = int(min(h_img, cy + nh * 0.5))
    return x0, y0, x1, y1


@torch.inference_mode()
def strat7_birdness_scores(frame_bgr, boxes_xywh, crop_scale):
    """Compute birdness scores using MobileNetV3."""
    if len(boxes_xywh) == 0:
        return []

    h_img, w_img = frame_bgr.shape[:2]
    crops = []
    for box in boxes_xywh:
        x0, y0, x1, y1 = _expand_box_xywh(box, w_img, h_img, scale=crop_scale)
        crop_bgr = frame_bgr[y0:y1, x0:x1]
        if crop_bgr.size == 0:
            crops.append(None)
            continue
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crops.append(crop_rgb)

    tensors = []
    valid_map = []
    for i, c in enumerate(crops):
        if c is None:
            continue
        c_tensor = torch.from_numpy(c).permute(2, 0, 1).contiguous()
        tensors.append(_strat7_preprocess(c_tensor))
        valid_map.append(i)

    scores = [0.0] * len(boxes_xywh)
    if len(tensors) == 0:
        return scores

    batch = torch.stack(tensors, dim=0)
    logits = _strat7_model(batch)
    probs = torch.softmax(logits, dim=1)

    if len(_strat7_bird_indices) > 0:
        bird_probs = probs[:, _strat7_bird_indices].sum(dim=1)
    else:
        bird_probs = probs.max(dim=1).values

    bird_probs = bird_probs.cpu().numpy().tolist()
    for p, orig_idx in zip(bird_probs, valid_map):
        scores[orig_idx] = float(p)

    return scores


def process_video_worker(args):
    """
    Worker function to process a single video for Strategy 7.
    """
    video_path, config, gt_data = args
    # Setup logging for the worker
    vis_utils.setup_worker_logging(config.get("log_queue"))
    logger = logging.getLogger(config["run_name"])

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
        dist_thresh=50, max_frames_to_skip=2, min_hits=2
    )

    for i, img_path in enumerate(images):
        img_start_time = time.time()  # Track per-image time

        if i % Config.LOG_PROCESSING_IMAGES_SKIP_COUNT == 0:
            percent = ((i + 1) / n_frames) * 100
            logger.info(
                f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)"
            )
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_detections = []

        if prev_gray is not None:
            warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
            if warped_prev is not None:
                # Motion compensation and detection logic
                diff = cv2.absdiff(curr_gray, warped_prev)
                diff_f = diff.astype(np.float32)
                diff_blur = cv2.GaussianBlur(diff_f, (0, 0), 2.0)
                diff_hp = np.maximum(diff_f - diff_blur, 0).astype(np.uint8)

                t_low = float(np.percentile(diff_hp, 98.5))
                t_high = float(np.percentile(diff_hp, 99.6))
                t_low = max(8.0, min(60.0, t_low))
                t_high = max(15.0, min(85.0, t_high))

                _, mask_low = cv2.threshold(diff_hp, t_low, 255, cv2.THRESH_BINARY)
                _, mask_high = cv2.threshold(diff_hp, t_high, 255, cv2.THRESH_BINARY)

                # Optical flow
                h_img, w_img = curr_gray.shape
                scale = 0.33
                small_size = (
                    max(64, int(w_img * scale)),
                    max(64, int(h_img * scale)),
                )
                prev_s = cv2.resize(
                    warped_prev, small_size, interpolation=cv2.INTER_AREA
                )
                curr_s = cv2.resize(curr_gray, small_size, interpolation=cv2.INTER_AREA)

                flow = cv2.calcOpticalFlowFarneback(
                    prev_s, curr_s, None, 0.5, 2, 15, 2, 5, 1.2, 0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                t_mag = max(0.20, float(np.percentile(mag, 85)))
                flow_mask_s = (mag > t_mag).astype(np.uint8) * 255
                flow_mask = cv2.resize(
                    flow_mask_s, (w_img, h_img), interpolation=cv2.INTER_NEAREST
                )

                # DoG blob detection
                g1 = cv2.GaussianBlur(curr_gray, (0, 0), 1.0)
                g2 = cv2.GaussianBlur(curr_gray, (0, 0), 2.5)
                dog = cv2.absdiff(g1, g2)
                t_dog = max(10.0, min(60.0, float(np.percentile(dog, 99.4))))
                _, dog_mask = cv2.threshold(dog, t_dog, 255, cv2.THRESH_BINARY)

                # Combine masks
                thresh = cv2.bitwise_or(
                    mask_high,
                    cv2.bitwise_and(mask_low, cv2.bitwise_or(flow_mask, dog_mask)),
                )
                k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k3, iterations=1)
                thresh = cv2.dilate(thresh, k3, iterations=2)

                # Find contours
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
                                raw_detections.append([x, y, w, h])

                # Score and filter
                if len(raw_detections) > 0:
                    scored = []
                    for x, y, w, h in raw_detections:
                        x0, y0 = max(0, x), max(0, y)
                        x1, y1 = min(w_img, x + w), min(h_img, y + h)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        s_diff = float(np.mean(diff_hp[y0:y1, x0:x1]))
                        s_dog = float(np.mean(dog[y0:y1, x0:x1]))
                        s_flow = float(np.mean(flow_mask[y0:y1, x0:x1]) / 255.0)
                        score = 1.0 * s_diff + 0.7 * s_dog + 25.0 * s_flow
                        scored.append(([x, y, w, h], score))

                    if len(scored) > 0:
                        boxes = [b for (b, s) in scored]
                        scores = [s for (b, s) in scored]
                        keep = vis_utils.nms_xywh(boxes, scores, iou_thr=0.3)
                        boxes = [boxes[i] for i in keep]
                        scores = [scores[i] for i in keep]
                        K = 5
                        # Re-sort by score
                        order = np.argsort(scores)[::-1][:K]
                        # Store as [x,y,w,h,score]
                        raw_detections = [boxes[i] + [scores[i]] for i in order]

                # CNN verifier
                if config["use_verifier"] and len(raw_detections) > 0:
                    bird_scores = strat7_birdness_scores(
                        frame, raw_detections, config["crop_scale"]
                    )
                    keep = [
                        i
                        for i, s in enumerate(bird_scores)
                        if s >= config["bird_threshold"]
                    ]
                    if len(keep) < config["min_keep"]:
                        keep = list(
                            np.argsort(bird_scores)[::-1][
                                : min(config["min_keep"], len(raw_detections))
                            ]
                        )
                    if len(keep) > config["max_keep"]:
                        keep_scores = [bird_scores[i] for i in keep]
                        keep = [
                            keep[i]
                            for i in np.argsort(keep_scores)[::-1][: config["max_keep"]]
                        ]
                    # Update scores with birdness scores and filter
                    final_raw = []
                    for i in keep:
                        det = raw_detections[i]  # [x,y,w,h,old_score]
                        det[4] = float(bird_scores[i])  # Update score
                        final_raw.append(det)
                    raw_detections = final_raw

        prev_gray = curr_gray

        # Tracking
        final_preds = obj_tracker.update(raw_detections)

        # Evaluation
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


@register_pipeline("strategy_7")
def run_strategy_7_pipeline(config: Dict[str, Any]):
    """Execute Strategy 7 pipeline with motion compensation and CNN verifier."""
    pipeline_name = config["run_name"]
    logger = logging.getLogger(pipeline_name)
    logger.info(f"--- STARTING STRATEGY 7 (PARALLEL): {pipeline_name} ---")

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
