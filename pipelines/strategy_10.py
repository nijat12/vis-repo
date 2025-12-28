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
    if img_w > slice_w: x_points.append(img_w - slice_w)
    
    y_points = list(range(0, img_h - slice_h, step_y))
    if img_h > slice_h: y_points.append(img_h - slice_h)
    
    # Unique sorted points
    x_points = sorted(list(set(x_points)))
    y_points = sorted(list(set(y_points)))
    
    coords = []
    for y in y_points:
        for x in x_points:
            coords.append((x, y, x + slice_w, y + slice_h))
    return coords

@register_pipeline("strategy_10")
def run_strategy_10_pipeline():
    """Execute Strategy 10: Motion-Gated Native Tiling."""
    logger.info("=" * 70)
    logger.info("STARTING STRATEGY 10 (Motion-Gated Native Tiling + YOLO)")
    logger.info("=" * 70)
    
    cfg = Config.STRATEGY_10_CONFIG
    
    if YOLO is None:
        logger.error("❌ ultralytics library not found.")
        raise ImportError("ultralytics library missing")
    
    logger.info(f"⏳ Loading YOLO: {cfg['model_name']}...")
    try:
        model = YOLO(cfg['model_name'])
        logger.info(f"✅ Model Loaded.")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time_global = time.time()

    video_folders = sorted(glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, '*')))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [video_folders[i] for i in Config.VIDEO_INDEXES]
        else:
            video_folders = video_folders[:min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)]

    if not video_folders:
        raise RuntimeError(f"No video folders found in {Config.LOCAL_TRAIN_DIR}")

    logger.info(f"📂 Found {len(video_folders)} videos. Starting...")

    # Initialize results tracker
    tracker = csv_utils.get_results_tracker()

    total_tp = total_fp = total_fn = total_time_sec = total_frames = 0
    results_data = []

    for video_path in video_folders:
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        if not images:
            continue

        vid_tp = vid_fp = vid_fn = 0
        vid_start = time.time()
        n_frames = len(images)
        prev_gray = None
        
        # Determine tile grid once
        first_frame = cv2.imread(images[0])
        h_img, w_img = first_frame.shape[:2]
        tile_coords = get_native_slices(h_img, w_img, slice_wh=(cfg['img_size'], cfg['img_size']), overlap_ratio=0.2)
        logger.info(f"🧩 Grid generated: {len(tile_coords)} tiles per frame.")

        for i, img_path in enumerate(images):
            img_start_time = time.time()
            if i % 50 == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")

            frame = cv2.imread(img_path)
            if frame is None: continue
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. GMC + Motion Mask + Keyframe Logic
            active_tiles = []
            active_offsets = []
            
            # Keyframe Full Scan: Process all tiles to establish a recall baseline
            is_keyframe = cfg['keyframe_interval'] > 0 and i % cfg['keyframe_interval'] == 0
            
            if is_keyframe:
                if i % 50 == 0: logger.info(f"⚡ Keyframe Full Scan on frame {i+1}")
                for (x1, y1, x2, y2) in tile_coords:
                    active_tiles.append(frame[y1:y2, x1:x2])
                    active_offsets.append((x1, y1))
            
            # Standard Motion Gating on non-keyframes
            elif prev_gray is not None:
                warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                if warped_prev is not None:
                    # Difference & Dynamic Threshold
                    diff = cv2.absdiff(curr_gray, warped_prev)
                    mean, std = cv2.meanStdDev(diff)
                    final_thresh = max(20, min(80, mean[0][0] + cfg["motion_thresh_scale"] * std[0][0]))
                    _, thresh = cv2.threshold(diff, final_thresh, 255, cv2.THRESH_BINARY)
                    
                    # Optional: Morphological Dilation to expand motion regions
                    if cfg.get("use_morphological_dilation", False):
                        kernel = np.ones((3,3), np.uint8)
                        thresh = cv2.dilate(thresh, kernel, iterations=1)

                    # 2. Gating: Check each tile for motion
                    for (x1, y1, x2, y2) in tile_coords:
                        tile_mask = thresh[y1:y2, x1:x2]
                        # Use configurable pixel threshold
                        if cv2.countNonZero(tile_mask) > cfg.get("motion_pixel_threshold", 20):
                            active_tiles.append(frame[y1:y2, x1:x2])
                            active_offsets.append((x1, y1))

            # 3. Native Inference (No resizing)
            raw_detections = []
            if active_tiles:
                # Batch across active tiles
                results = model(active_tiles, imgsz=cfg['img_size'], verbose=False, conf=cfg['conf_thresh'], classes=[cfg['bird_class_id']])
                
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
                    keep = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
                    final = pred_boxes[keep].numpy()
                    for x1, y1, x2, y2 in final:
                        raw_detections.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

            prev_gray = curr_gray

            # --- EVALUATION ---
            img_filename = os.path.basename(img_path)
            key = f"{video_name}/{img_filename}"
            gts = gt_data.get(key, [])
            matched_gt = set()
            img_tp = img_fp = 0

            for p_box in raw_detections:
                best_dist = 10000
                best_idx = -1
                for idx, g_box in enumerate(gts):
                    if idx in matched_gt: continue
                    d = vis_utils.calculate_center_distance(p_box, g_box)
                    if d < best_dist: best_dist = d; best_idx = idx

                if best_dist <= 30:
                    vid_tp += 1; img_tp += 1
                    matched_gt.add(best_idx)
                else:
                    vid_fp += 1; img_fp += 1

            img_fn = len(gts) - len(matched_gt)
            vid_fn += img_fn
            
            img_processing_time = time.time() - img_start_time
            
            image_result = csv_utils.create_image_result(
                video_name=video_name, frame_name=img_filename, image_path=img_path,
                predictions=raw_detections, ground_truths=gts, tp=img_tp, fp=img_fp, fn=img_fn,
                processing_time_sec=img_processing_time,
                iou=0.0, mAP=0.0, memory_usage_mb=0.0
            )
            tracker.add_image_result("strategy_10", image_result)
            if (i + 1) % 50 == 0: tracker.save_batch("strategy_10", batch_size=50)

        # Video Stats
        vid_time = time.time() - vid_start
        vid_fps = n_frames / vid_time if vid_time > 0 else 0
        
        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        logger.info(f"{'Video':<8} | {'Fr':<4} | {'FPS':<5} | {'P':<4} | {'R':<4} | {'F1':<4} | {'Time'}")
        logger.info(f"{video_name:<8} | {n_frames:<4} | {vid_fps:<5.1f} | {prec:<4.2f} | {rec:<4.2f} | {f1:<4.2f} | {str(datetime.timedelta(seconds=int(vid_time)))}")

        results_data.append({
            'Video': video_name, 'Frames': n_frames, 'FPS': round(vid_fps, 2),
            'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4),
            'TP': vid_tp, 'FP': vid_fp, 'FN': vid_fn, 'Video_Time': vid_time
        })
        
        total_time_sec += vid_time; total_frames += n_frames
        total_tp += vid_tp; total_fp += vid_fp; total_fn += vid_fn

    # Final Summary
    logger.info("=" * 65)
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    logger.info("FINAL RESULTS (Strategy 10 - Motion-Gated Native Tiling + YOLO):")
    logger.info(f"Total Frames:   {total_frames}")
    logger.info(f"Average FPS:    {avg_fps:.2f}")
    logger.info(f"Precision:      {overall_prec:.4f}")
    logger.info(f"Recall:         {overall_rec:.4f}")
    logger.info(f"F1-Score:       {overall_f1:.4f}")
    logger.info(f"TP:             {total_tp}")
    logger.info(f"FP:             {total_fp}")
    logger.info(f"FN:             {total_fn}")
    logger.info(f"⏱️ Process took: {str(datetime.timedelta(seconds=int(time.time() - start_time_global)))}")
    logger.info("=" * 65)
    
    
    
    tracker.update_summary("strategy_10", {
        "total_frames": total_frames, 
        "avg_fps": avg_fps,
        "precision": overall_prec, 
        "recall": overall_rec, 
        "f1_score": overall_f1,
        "execution_time_sec": time.time() - start_time_global,
        "tp": total_tp, "fp": total_fp, "fn": total_fn
    })
    
    return {
        "pipeline": "strategy_10",
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time_global,
    }

if __name__ == "__main__":
    vis_utils.setup_logging()
    run_strategy_10_pipeline()