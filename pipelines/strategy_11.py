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

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

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

def get_filtered_roi_predictions(det_model, cls_model, img_bgr, cfg, frame_idx, motion_mask=None):
    """
    1. Grid-based multi-scale classification (224px and 448px).
    2. Overlapping tiles to prevent split objects.
    3. Motion-gating (optional) to skip empty skies.
    4. Hit verification with 640px detector.
    """
    if det_model is None or cls_model is None:
        return []

    h_img, w_img, _ = img_bgr.shape
    
    # 1. Generate Grids
    grid_small = get_tiled_coords(h_img, w_img, cfg['cls_img_size'], cfg['cls_overlap'])
    grid_large = get_tiled_coords(h_img, w_img, cfg['cls_scale2_size'], cfg['cls_overlap'])
    
    active_crops = []
    active_info = [] # (x0, y0, scale_size)

    # 2. Extract & Filter by Motion (if mask provided)
    for (x0, y0, x1, y1) in grid_small:
        if motion_mask is not None:
            if cv2.countNonZero(motion_mask[y0:y1, x0:x1]) < 10:
                continue
        active_crops.append(img_bgr[y0:y1, x0:x1])
        active_info.append((x0, y0, cfg['cls_img_size']))

    for (x0, y0, x1, y1) in grid_large:
        if motion_mask is not None:
            if cv2.countNonZero(motion_mask[y0:y1, x0:x1]) < 10:
                continue
        # We extract 448px, YOLO will resize to 224px for classification
        active_crops.append(img_bgr[y0:y1, x0:x1])
        active_info.append((x0, y0, cfg['cls_scale2_size']))

    if not active_crops:
        return []

    # 3. Stage 1: Batch Classification (imgsz=224)
    # The classifier model is presumably trained on 224px.
    cls_results = cls_model(active_crops, imgsz=cfg['cls_img_size'], verbose=False)
    
    # Identify unique centers for verification
    verification_centers = []
    
    for idx, res in enumerate(cls_results):
        top1_idx = res.probs.top1
        top1_conf = float(res.probs.top1conf)
        
        is_bird = "bird" in res.names[top1_idx].lower()
        if is_bird and top1_conf >= cfg['cls_conf_thresh']:
            # Calculate global center of this tile
            x0, y0, sz = active_info[idx]
            cx, cy = x0 + sz/2, y0 + sz/2
            verification_centers.append((cx, cy))

    if not verification_centers:
        return []

    # 4. Stage 2: Verification with 640px Detector
    # Group centers that are close to each other to minimize duplicated 640px crops
    final_verification_crops = []
    final_verification_offsets = []
    
    merged_centers = []
    while verification_centers:
        curr = verification_centers.pop(0)
        merged_centers.append(curr)
        verification_centers = [c for c in verification_centers if np.sqrt((c[0]-curr[0])**2 + (c[1]-curr[1])**2) > 200]

    for (cx, cy) in merged_centers:
        # Generate 640x640 ROI centered on cx, cy
        x0 = int(max(0, cx - cfg['img_size']/2))
        y0 = int(max(0, cy - cfg['img_size']/2))
        x1 = int(min(w_img, x0 + cfg['img_size']))
        y1 = int(min(h_img, y0 + cfg['img_size']))
        
        # Adjust if too close to right/bottom edge
        if x1 - x0 < cfg['img_size']: x0 = max(0, x1 - cfg['img_size'])
        if y1 - y0 < cfg['img_size']: y0 = max(0, y1 - cfg['img_size'])
        
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size > 0:
            final_verification_crops.append(crop)
            final_verification_offsets.append((x0, y0))

    if not final_verification_crops:
        return []

    # Final Detection Pass
    det_results = det_model(final_verification_crops, imgsz=cfg['img_size'], verbose=False, conf=cfg['conf_thresh'], classes=cfg['model_classes'])

    all_boxes = []
    all_scores = []

    for j, res in enumerate(det_results):
        boxes = res.boxes
        if len(boxes) > 0:
            local_boxes = boxes.xyxy.cpu()
            local_scores = boxes.conf.cpu()
            x_off, y_off = final_verification_offsets[j]
            
            shifted_boxes = local_boxes.clone()
            shifted_boxes[:, 0] += x_off; shifted_boxes[:, 1] += y_off
            shifted_boxes[:, 2] += x_off; shifted_boxes[:, 3] += y_off
            
            all_boxes.append(shifted_boxes)
            all_scores.append(local_scores)

    if not all_boxes:
        return []

    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)
    keep = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=cfg['iou_thresh'])
    final = pred_boxes[keep].numpy()

    return [[float(x1), float(y1), float(x2-x1), float(y2-y1)] for x1, y1, x2, y2 in final]

@register_pipeline("strategy_11")
def run_strategy_11_pipeline():
    """Execute Strategy 11: ROI + Classifier + Detector."""
    logger = logging.getLogger("pipelines.strategy_11")
    logger.info("STARTING STRATEGY 11 (ROI -> Classifier -> Detector)")
    
    cfg = Config.STRATEGY_11_CONFIG
    
    if YOLO is None:
        logger.error("❌ ultralytics library not found.")
        raise ImportError("ultralytics library missing")
    
    # Load models
    logger.info(f"⏳ Loading Detector: {cfg['model_name']}...")
    logger.info(f"⏳ Loading Classifier: {cfg['classifier_model_name']}...")
    try:
        det_model = YOLO(cfg['model_name'])
        cls_model = YOLO(cfg['classifier_model_name'])
        logger.info(f"✅ Models Loaded.")
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

    tracker = csv_utils.get_results_tracker()
    total_tp = total_fp = total_fn = total_time_sec = total_frames = 0

    for video_path in video_folders:
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        if not images:
            continue

        vid_tp = vid_fp = vid_fn = 0
        vid_start = time.time()
        n_frames = len(images)
        prev_gray = None
        obj_tracker = vis_utils.ObjectTracker(dist_thresh=50, max_frames_to_skip=4, min_hits=2)

        for i, img_path in enumerate(images):
            img_start_time = time.time()
            if i % 50 == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")

            frame = cv2.imread(img_path)
            if frame is None: continue
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            raw_detections = []

            # Stage 1: Classifier-Gated Detection
            if i % cfg['detect_every'] == 0:
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
                    det_model, cls_model, frame, cfg, i, motion_mask=motion_mask
                )

            prev_gray = curr_gray
            final_preds = obj_tracker.update(raw_detections)

            # Evaluation
            img_filename = os.path.basename(img_path)
            key = f"{video_name}/{img_filename}"
            gts = gt_data.get(key, [])
            matched_gt = set()
            img_tp = img_fp = 0

            for p_box in final_preds:
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
            
            # IoU
            img_ious = []
            matched_gt_indices = set()
            for p_box in final_preds:
                best_iou = 0
                best_idx = -1
                for g_idx, g_box in enumerate(gts):
                    if g_idx in matched_gt_indices: continue
                    iou = vis_utils.box_iou_xywh(p_box, g_box)
                    if iou > best_iou: best_iou = iou; best_idx = g_idx
                if best_idx != -1 and best_iou > 0:
                    img_ious.append(best_iou)
                    matched_gt_indices.add(best_idx)
            
            img_avg_iou = np.mean(img_ious) if img_ious else 0.0
            img_processing_time = time.time() - img_start_time
            img_mem = vis_utils.get_memory_usage()
            
            image_result = csv_utils.create_image_result(
                video_name=video_name, frame_name=img_filename, image_path=img_path,
                predictions=final_preds, ground_truths=gts, tp=img_tp, fp=img_fp, fn=img_fn,
                processing_time_sec=img_processing_time,
                iou=img_avg_iou, mAP=0.0, memory_usage_mb=img_mem
            )
            tracker.add_image_result("strategy_11", image_result)
            if (i + 1) % 50 == 0: tracker.save_batch("strategy_11", batch_size=50)

        # Video Stats
        vid_time = time.time() - vid_start
        vid_fps = n_frames / vid_time if vid_time > 0 else 0
        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        p_data = [d for d in tracker.detailed_data.get("strategy_11", []) if d['video'] == video_name]
        vid_iou = np.mean([d['iou'] for d in p_data]) if p_data else 0.0
        vid_mem = np.mean([d['memory_usage_mb'] for d in p_data]) if p_data else 0.0

        vis_utils.log_video_metrics(logger, video_name, {
            'n_frames': n_frames, 'fps': vid_fps, 'precision': prec, 'recall': rec, 'f1_score': f1,
            'tp': vid_tp, 'fp': vid_fp, 'fn': vid_fn, 'iou': vid_iou, 'mAP': 0.0,
            'memory_usage_mb': vid_mem, 'vid_time': vid_time
        })
        
        total_time_sec += vid_time; total_frames += n_frames
        total_tp += vid_tp; total_fp += vid_fp; total_fn += vid_fn

    # Final Summary
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    p_data = tracker.detailed_data.get("strategy_11", [])
    overall_iou = np.mean([d['iou'] for d in p_data]) if p_data else 0.0
    overall_mem = np.mean([d['memory_usage_mb'] for d in p_data]) if p_data else 0.0

    summary_metrics = {
        "total_frames": total_frames, "avg_fps": avg_fps, "precision": overall_prec, "recall": overall_rec, "f1_score": overall_f1,
        "tp": total_tp, "fp": total_fp, "fn": total_fn, "iou": overall_iou, "mAP": 0.0, "memory_usage_mb": overall_mem,
        "processing_time_sec": total_time_sec, "execution_time_sec": time.time() - start_time_global
    }

    vis_utils.log_pipeline_summary(logger, "strategy_11", summary_metrics)
    tracker.update_summary("strategy_11", summary_metrics)
    
    return {
        "pipeline": "strategy_11",
        "total_frames": total_frames, "avg_fps": avg_fps, "precision": overall_prec, "recall": overall_rec, "f1_score": overall_f1,
        "execution_time": time.time() - start_time_global,
    }

if __name__ == "__main__":
    vis_utils.setup_logging()
    run_strategy_11_pipeline()
