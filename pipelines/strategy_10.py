"""
Strategy 10 Pipeline: Motion Proposals + YOLO Classification

This pipeline implements a hybrid approach:
1. Frame Alignment (GMC) to compensate for camera motion.
2. Motion Proposals: Detect moving blobs using frame differencing.
3. Precision Filter: Use a YOLOv5 classifier to verify if proposals contain birds.
"""

import os
import glob
import time
import datetime
import math
import logging
from collections import defaultdict

import cv2
import pandas as pd
import numpy as np

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

# ==========================================
# 1. SETUP & MODEL LOADING
# ==========================================
def load_classifier_model():
    """
    Loads YOLOv5 (or V11 if compatible) as a classifier.
    We use 'yolov5s6' because it's trained on large images and is robust.
    """
    cfg = Config.STRATEGY_10_CONFIG
    model_name = cfg["model_name"]
    
    if YOLO is None:
        logger.error("❌ ultralytics library not found. Please run: pip install ultralytics")
        return None

    logger.info(f"⏳ Loading YOLOv11: {model_name}...")
    try:
        model = YOLO(model_name)
        logger.info(f"✅ {model_name} Loaded.")
        return model
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        return None

# ==========================================
# 2. STABILIZATION (GMC)
# ==========================================
def align_frames_gmc(prev_gray, curr_gray):
    """
    Align previous frame to current frame using feature tracking and homography.
    """
    # Detect features
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    if prev_pts is None:
        return None
    
    # Track features
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    
    # Filter to only good features
    status = status.flatten()
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    
    if len(good_prev) < 4:
        return None
    
    # Compute Homography (RANSAC)
    H, mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
    if H is None:
        return None
    
    # Warp
    height, width = prev_gray.shape
    return cv2.warpPerspective(prev_gray, H, (width, height))

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
@register_pipeline("strategy_10")
def run_strategy_10_pipeline():
    """
    Executes Strategy 10 Pipeline:
    Motion Proposals -> YOLO Classification
    """
    logger.info("STARTING STRATEGY 10 PIPELINE (Motion Proposals + YOLO Classification)")
    
    # 1. Configuration Setup
    cfg = Config.STRATEGY_10_CONFIG
    
    # 2. Load Model
    classifier = load_classifier_model()
    if classifier is None:
        raise RuntimeError("Failed to load classifier model")

    # 3. Load Ground Truth
    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_global = time.time()

    # 4. Get Videos
    video_folders = sorted(glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, '*')))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [video_folders[i] for i in Config.VIDEO_INDEXES]
        else:
            video_folders = video_folders[:min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)]

    if not video_folders:
        raise RuntimeError(f"No video folders found in {Config.LOCAL_TRAIN_DIR}")

    logger.info(f"📂 Found {len(video_folders)} videos. Starting pipeline...")

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
        prev_gray = None
        n_frames = len(images)
        
        for i, img_path in enumerate(images):
            img_start_time = time.time()
            
            if i % 50 == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")

            frame = cv2.imread(img_path)
            if frame is None:
                continue
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            final_preds = [] 
            
            # --- STAGE 1: MOTION PROPOSALS ---
            if prev_gray is not None:
                warped_prev = align_frames_gmc(prev_gray, curr_gray)
                
                if warped_prev is not None:
                    # Difference & Dynamic Threshold
                    diff = cv2.absdiff(curr_gray, warped_prev)
                    mean, std = cv2.meanStdDev(diff)
                    final_thresh = max(20, min(80, mean[0][0] + cfg["motion_thresh_scale"] * std[0][0]))
                    _, thresh = cv2.threshold(diff, final_thresh, 255, cv2.THRESH_BINARY)
                    
                    # Clean Noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                    thresh = cv2.dilate(thresh, kernel, iterations=1)
                    
                    # Find Contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    h_img, w_img, _ = frame.shape
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        # Motion filter (Min/Max size)
                        if 50 < area < 10000:
                            x, y, w, h = cv2.boundingRect(cnt)
                            
                            # Boundary Check
                            if x > 5 and y > 5 and (x+w) < (w_img-5) and (y+h) < (h_img-5):
                                
                                # --- STAGE 2: YOLO CLASSIFICATION ---
                                # Crop with padding context
                                pad = 20
                                x1 = max(0, x-pad); y1 = max(0, y-pad)
                                x2 = min(w_img, x+w+pad); y2 = min(h_img, y+h+pad)
                                
                                crop = frame[y1:y2, x1:x2]
                                
                                if crop.size > 0:
                                    # Convert BGR to RGB for YOLO
                                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                    
                                    # Run Inference
                                    # classes=[14] is typically Bird in COCO
                                    results = classifier(crop_rgb, verbose=False, conf=cfg["conf_thresh"], classes=[cfg["bird_class_id"]])
                                    
                                    # Check predictions
                                    if len(results[0].boxes) > 0:
                                        # If YOLO sees a bird in this box, keep the motion box
                                        final_preds.append([float(x), float(y), float(w), float(h)])

            prev_gray = curr_gray
            
            # --- EVALUATION ---
            img_filename = os.path.basename(img_path)
            key = f"{video_name}/{img_filename}"
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

                if best_dist <= 30: # 30px center distance threshold
                    vid_tp += 1
                    img_tp += 1
                    matched_gt.add(best_idx)
                else:
                    vid_fp += 1
                    img_fp += 1

            img_fn = len(gts) - len(matched_gt)
            vid_fn += img_fn
            
            # Processing time
            img_processing_time = time.time() - img_start_time
            
            # Log to ResultsTracker
            image_result = csv_utils.create_image_result(
                video_name=video_name,
                frame_name=img_filename,
                image_path=img_path,
                predictions=final_preds,
                ground_truths=gts,
                tp=img_tp,
                fp=img_fp,
                fn=img_fn,
                processing_time_sec=img_processing_time
            )
            tracker.add_image_result("strategy_10", image_result)
            
            if (i + 1) % 50 == 0:
                tracker.save_batch("strategy_10", batch_size=50)

        # Video Stats
        vid_end = time.time()
        vid_time = vid_end - vid_start
        vid_fps = n_frames / vid_time if vid_time > 0 else 0

        total_time_sec += vid_time
        total_frames += n_frames
        total_tp += vid_tp
        total_fp += vid_fp
        total_fn += vid_fn

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

    # Final Summary
    logger.info("=" * 65)
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    logger.info("FINAL RESULTS (Strategy 10 - Motion + YOLO):")
    logger.info(f"Total Frames:   {total_frames}")
    logger.info(f"Average FPS:    {avg_fps:.2f}")
    logger.info(f"Precision:      {overall_prec:.4f}")
    logger.info(f"Recall:         {overall_rec:.4f}")
    logger.info(f"F1-Score:       {overall_f1:.4f}")
    logger.info("=" * 65)

    # Save to CSV
    df = pd.DataFrame(results_data)
    output_path = Config.get_output_path("strategy_10")
    final_path = vis_utils.get_next_version_path(output_path)
    df.to_csv(final_path, index=False)
    
    logger.info(f"✅ CSV Saved: {final_path}")
    logger.info(f"⏱️ Process took: {str(datetime.timedelta(seconds=int(time.time() - start_global)))}")
    
    # Update tracker summary
    tracker.update_summary("strategy_10", {
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time_sec": time.time() - start_global,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    })

    return {
        "pipeline": "strategy_10",
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_global,
        "output_file": final_path
    }

if __name__ == "__main__":
    # Setup basic logging for standalone run
    vis_utils.setup_logging()
    run_strategy_10_pipeline()