"""
Strategy 2 Pipeline: GMC + Dynamic Thresholding + YOLO Refiner

This pipeline migrates the logic from Strategy 2 Colab:
1. Global Motion Compensation (GMC) to align frames.
2. Dynamic Statistical Thresholding (mean + 4*std) to isolate motion.
3. Morphological opening (5x5 kernel) to remove noise.
4. Contour filtering (area, aspect ratio, border) to generate ROIs.
5. YOLO inference on these ROIs for final bird detection.
6. Persistence tracking for temporal consistency.
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


def get_roi_predictions(model, img_bgr, proposals_xywh, img_size, conf_thresh, classes, roi_scale, min_roi, max_rois):
    """Run YOLO only on ROI crops around proposals."""
    if model is None or not proposals_xywh:
        return []

    h, w, _ = img_bgr.shape
    crops = []
    offsets = []

    # Take top N proposals
    use_props = proposals_xywh[:min(len(proposals_xywh), max_rois)]

    for b in use_props:
        x0, y0, x1, y1 = _expand_roi_xywh(b, w, h, scale=roi_scale, min_size=min_roi)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((x0, y0))

    if len(crops) == 0:
        return []

    # Batch Inference
    results = model(crops, imgsz=img_size, verbose=False, conf=conf_thresh, classes=classes)

    all_boxes = []
    all_scores = []

    for j, res in enumerate(results):
        boxes = res.boxes
        if len(boxes) > 0:
            local_boxes = boxes.xyxy.cpu()
            local_scores = boxes.conf.cpu()
            x_off, y_off = offsets[j]
            
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
    
    keep = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
    final = pred_boxes[keep].numpy()

    final_preds = []
    for x1, y1, x2, y2 in final:
        final_preds.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    return final_preds


@register_pipeline("strategy_2")
def run_strategy_2_pipeline():
    """Execute Strategy 2 pipeline: GMC + Dynamic Threshold + YOLO Refiner."""
    logger.info("=" * 70)
    logger.info("STARTING STRATEGY 2 PIPELINE (GMC + Dynamic Thresholding + YOLO)")
    logger.info("=" * 70)
    
    cfg = Config.STRATEGY_2_CONFIG
    
    if YOLO is None:
        logger.error("❌ ultralytics library not found. Please run: pip install ultralytics")
        raise ImportError("ultralytics library missing")
    
    logger.info(f"⏳ Loading YOLO Model: {cfg['model_name']}...")
    try:
        model = YOLO(cfg['model_name'])
        logger.info(f"✅ Model {cfg['model_name']} Loaded.")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()

    video_folders = sorted(glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, '*')))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [video_folders[i] for i in Config.VIDEO_INDEXES]
        else:
            video_folders = video_folders[:min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)]

    if not video_folders:
        raise RuntimeError(f"No video folders found in {Config.LOCAL_TRAIN_DIR}")

    logger.info(f"📂 Found {len(video_folders)} videos. Starting Strategy 2 processing...")

    # Initialize results tracker
    tracker = csv_utils.get_results_tracker()

    total_tp = total_fp = total_fn = total_time = total_frames = 0
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
        # Persistence Tracking exactly as in Strategy 2 notebook (min_hits=3, dist_thresh=50)
        obj_tracker = vis_utils.ObjectTracker(dist_thresh=50, max_frames_to_skip=5, min_hits=cfg['min_hits'])

        for i, img_path in enumerate(images):
            img_start_time = time.time()
            
            if i % 50 == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")

            frame = cv2.imread(img_path)
            if frame is None:
                continue
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h_img, w_img = curr_gray.shape

            proposals = []
            if prev_gray is not None:
                # 1. GMC: Align frames
                warped_prev = vis_utils.align_frames(prev_gray, curr_gray)
                if warped_prev is not None:
                    # 2. Dynamic Thresholding (Notebook Logic)
                    diff = cv2.absdiff(curr_gray, warped_prev)
                    mean, std = cv2.meanStdDev(diff)
                    # Threshold = Mean + 4 * Sigma
                    dynamic_thresh = mean[0][0] + cfg['dynamic_multiplier'] * std[0][0]
                    # Clamp threshold between 20 and 80
                    final_thresh = max(cfg['min_threshold'], min(cfg['max_threshold'], dynamic_thresh))
                    _, thresh = cv2.threshold(diff, final_thresh, 255, cv2.THRESH_BINARY)

                    # 3. Morphological Opening (Notebook Logic)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.dilate(thresh, kernel, iterations=1)

                    # 4. Contour Filtering (Notebook Logic)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        # Filter noise: 50 < Area < 5000
                        if 50 < area < 5000:
                            x, y, w, h = cv2.boundingRect(cnt)
                            # Aspect Ratio Filter: 0.2 < W/H < 4.0
                            aspect_ratio = float(w) / h
                            if 0.2 < aspect_ratio < 4.0:
                                # Border check: Ignore artifacts near edge
                                border = 15
                                if x > border and y > border and (x+w) < (w_img-border) and (y+h) < (h_img-border):
                                    proposals.append([x, y, w, h])

            # 5. YOLO ROI Refiner (The "Migration" to Deep Learning)
            raw_detections = []
            if proposals:
                raw_detections = get_roi_predictions(
                    model, frame, proposals, cfg['img_size'], cfg['conf_thresh'], cfg['model_classes'],
                    roi_scale=cfg['roi_scale'], min_roi=cfg['min_roi_size'], max_rois=cfg['max_rois']
                )

            # Update for next frame
            prev_gray = curr_gray

            # 6. Persistence Tracking
            final_preds = obj_tracker.update(raw_detections)

            # --- EVALUATION ---
            key = f"{video_name}/{os.path.basename(img_path)}"
            gts = gt_data.get(key, [])
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
                    matched_gt.add(best_idx)
                else:
                    vid_fp += 1
                    img_fp += 1

            img_fn = len(gts) - len(matched_gt)
            vid_fn += img_fn
            
            img_processing_time = time.time() - img_start_time
            
            # Log result
            image_result = csv_utils.create_image_result(
                video_name=video_name,
                frame_name=os.path.basename(img_path),
                image_path=img_path,
                predictions=final_preds,
                ground_truths=gts,
                tp=img_tp,
                fp=img_fp,
                fn=img_fn,
                processing_time_sec=img_processing_time
            )
            tracker.add_image_result("strategy_2", image_result)
            
            if (i + 1) % 50 == 0:
                tracker.save_batch("strategy_2", batch_size=50)

        vid_time = time.time() - vid_start
        fps = len(images) / vid_time if vid_time > 0 else 0
        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        logger.info(f"{'Video':<8} | {'Fr':<4} | {'FPS':<5} | {'P':<4} | {'R':<4} | {'F1':<4} | {'Time'}")
        logger.info(f"{video_name:<8} | {len(images):<4} | {fps:<5.1f} | {prec:<4.2f} | {rec:<4.2f} | {f1:<4.2f} | {str(datetime.timedelta(seconds=int(vid_time)))}")

        results_data.append({
            'Video': video_name, 'Frames': len(images), 'FPS': round(fps, 2),
            'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4),
            'TP': vid_tp, 'FP': vid_fp, 'FN': vid_fn, 'Video_Time': vid_time
        })
        total_time += vid_time
        total_frames += len(images)
        total_tp += vid_tp
        total_fp += vid_fp
        total_fn += vid_fn

    logger.info("=" * 65)
    avg_fps = total_frames / total_time if total_time > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    logger.info("FINAL RESULTS (Strategy 2 - GMC+Dynamic+YOLO):")
    logger.info(f"Total Frames:   {total_frames}")
    logger.info(f"Average FPS:    {avg_fps:.2f}")
    logger.info(f"Precision:      {overall_prec:.4f}")
    logger.info(f"Recall:         {overall_rec:.4f}")
    logger.info(f"F1-Score:       {overall_f1:.4f}")
    logger.info("=" * 65)

    df = pd.DataFrame(results_data)
    output_path = Config.get_output_path("strategy_2")
    final_path = vis_utils.get_next_version_path(output_path)
    df.to_csv(final_path, index=False)
    logger.info(f"✅ CSV Saved: {final_path}")
    logger.info(f"⏱️  Process took: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
    
    tracker.update_summary("strategy_2", {
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time_sec": time.time() - start_time,
        "tp": total_tp, "fp": total_fp, "fn": total_fn
    })

    return {
        "pipeline": "strategy_2",
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time,
        "output_file": final_path
    }
