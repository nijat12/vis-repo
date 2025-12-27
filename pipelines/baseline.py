"""
Baseline Pipeline: YOLOv5n with 4x3 Tiled Inference

This pipeline implements the baseline strategy using:
- YOLOv5n pretrained model
- 4x3 grid tiling with overlap for better small object detection
- Batch inference optimization
- Center distance matching for evaluation
"""

import os
import glob
import time
import datetime
import sys
import logging
import math
from collections import defaultdict

import cv2
import torch
import torchvision
import pandas as pd

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

logger = logging.getLogger(__name__)


def get_tiled_predictions(model, img, img_size, device):
    """
    Splits image into a 4x3 Grid (12 tiles) and runs inference.
    Optimization: Sends all 12 tiles in ONE BATCH to maximize throughput.
    
    Args:
        model: YOLOv5 model
        img: Input image (BGR format)
        img_size: Target size for inference
        device: torch device (cpu/cuda)
    
    Returns:
        List of predictions in [x, y, w, h] format
    """
    h, w, _ = img.shape

    # Grid Configuration: 4 Cols x 3 Rows = 12 Tiles
    N_COLS = 4
    N_ROWS = 3

    h_step = h // N_ROWS
    w_step = w // N_COLS
    h_over = int(h_step * 0.20)  # 20% overlap
    w_over = int(w_step * 0.20)

    crops = []
    offsets = []

    for r in range(N_ROWS):
        for c in range(N_COLS):
            y1 = max(0, r * h_step - h_over)
            x1 = max(0, c * w_step - w_over)
            y2 = min(h, (r + 1) * h_step + h_over)
            x2 = min(w, (c + 1) * w_step + w_over)

            crops.append(img[y1:y2, x1:x2])
            offsets.append((x1, y1))

    # Batch Inference
    all_boxes = []
    all_scores = []

    # Process ALL 12 tiles in one batch
    CHUNK_SIZE = 12

    for i in range(0, len(crops), CHUNK_SIZE):
        sub_crops = crops[i : i + CHUNK_SIZE]
        sub_offsets = offsets[i : i + CHUNK_SIZE]

        results = model(sub_crops, size=img_size)

        for j, det in enumerate(results.xyxy):
            if det is not None and len(det) > 0:
                det = det.clone()
                x_off, y_off = sub_offsets[j]
                # Shift crop coordinates back to full-frame
                det[:, 0] += x_off
                det[:, 1] += y_off
                det[:, 2] += x_off
                det[:, 3] += y_off
                all_boxes.append(det[:, :4])
                all_scores.append(det[:, 4])

    if not all_boxes:
        return []

    # Merge and apply NMS
    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_scores = torch.cat(all_scores, dim=0)
    keep_indices = torchvision.ops.nms(pred_boxes, pred_scores, iou_threshold=0.45)
    final_tensor = pred_boxes[keep_indices]

    # Convert to xywh format
    final_preds = []
    final_tensor = final_tensor.cpu().numpy()
    for box in final_tensor:
        x1, y1, x2, y2 = box
        final_preds.append([x1, y1, x2-x1, y2-y1])

    return final_preds


@register_pipeline("baseline")
def run_baseline_pipeline():
    """
    Execute the baseline pipeline with YOLOv5n tiled inference.
    
    Returns:
        Dict with execution metrics and results path
    """
    logger.info("STARTING BASELINE PIPELINE (YOLOv5n + 4x3 Tiling)")
    
    # Load configuration
    cfg = Config.BASELINE_CONFIG
    MODEL_NAME = cfg["model_name"]
    IMG_SIZE = cfg["img_size"]
    CONF_THRESH = cfg["conf_thresh"]
    IOU_THRESH = cfg["iou_thresh"]
    model_classes = cfg["model_classes"]
    
    # Load model
    logger.info(f"⏳ Loading Model: {MODEL_NAME}...")
    try:
        model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True, force_reload=False, trust_repo=True)
        model.conf = CONF_THRESH
        model.classes = model_classes

        if torch.cuda.is_available() and Config.IS_GPU_ALLOWED:
            device = torch.device('cuda')
            logger.info("✅ Model Loaded on GPU.")
        else:
            device = torch.device('cpu')
            logger.info("⚠️  Model Loaded on CPU.")

        model.to(device)
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    # Load ground truth
    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()

    # Select videos to process
    video_folders = sorted(glob.glob(os.path.join(Config.LOCAL_TRAIN_DIR, '*')))
    video_folders = [f for f in video_folders if os.path.isdir(f)]
    
    if Config.SHOULD_LIMIT_VIDEO:
        if Config.SHOULD_LIMIT_VIDEO == 1:
            video_folders = [video_folders[i] for i in Config.VIDEO_INDEXES]
        else:
            video_folders = video_folders[:min(len(video_folders), Config.SHOULD_LIMIT_VIDEO)]

    if not video_folders:
        raise RuntimeError(f"No video folders found in {Config.LOCAL_TRAIN_DIR}")

    logger.info(f"📂 Found {len(video_folders)} videos. Starting Batched Inference (4x3 Tiling)...")

    # Initialize results tracker
    tracker = csv_utils.get_results_tracker()

    total_tp = total_fp = total_fn = total_time_sec = total_frames = 0
    results_data = []

    logger.info(f"{'Video':<10} | {'Frames':<6} | {'FPS':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Time':<6}")

    for v_idx, video_path in enumerate(video_folders):
        video_name = os.path.basename(video_path)
        images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
        if not images:
            continue

        vid_tp = vid_fp = vid_fn = 0
        vid_start = time.time()
        n_frames = len(images)

        for i, img_path in enumerate(images):
            img_start_time = time.time()  # Track per-image time
            
            if i % 20 == 0:
                percent = ((i + 1) / n_frames) * 100
                sys.stdout.write(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")
                sys.stdout.flush()

            img = cv2.imread(img_path)
            if img is None:
                continue

            preds = get_tiled_predictions(model, img, IMG_SIZE, device)

            img_filename = os.path.basename(img_path)
            key = f"{video_name}/{img_filename}"
            gts = gt_data.get(key, [])

            # Track per-image results
            img_tp = img_fp = 0
            matched_gt = set()
            for p_box in preds:
                best_dist = 10000
                best_idx = -1
                for g_idx, g_box in enumerate(gts):
                    if g_idx in matched_gt:
                        continue
                    d = vis_utils.calculate_center_distance(p_box, g_box)
                    if d < best_dist:
                        best_dist = d
                        best_idx = g_idx

                if best_dist <= 30:
                    vid_tp += 1
                    img_tp += 1
                    matched_gt.add(best_idx)
                else:
                    vid_fp += 1
                    img_fp += 1
            
            img_fn = len(gts) - len(matched_gt)
            vid_fn += img_fn
            
            # Calculate processing time for this image
            img_processing_time = time.time() - img_start_time
            
            # Save per-image result
            image_result = csv_utils.create_image_result(
                video_name=video_name,
                frame_name=img_filename,
                image_path=img_path,
                predictions=preds,
                ground_truths=gts,
                tp=img_tp,
                fp=img_fp,
                fn=img_fn,
                processing_time_sec=img_processing_time
            )
            tracker.add_image_result("baseline", image_result)
            
            # Save batch every 50 images
            if (i + 1) % 50 == 0:
                tracker.save_batch("baseline", batch_size=50)

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

        sys.stdout.write("\r" + " " * 80 + "\r")
        logger.info(f"{video_name:<10} | {n_frames:<6} | {vid_fps:<6.1f} | {prec:<6.2f} | {rec:<6.2f} | {f1:<6.2f} | {str(datetime.timedelta(seconds=int(vid_time)))}")

        results_data.append({
            'Video': video_name, 'Frames': n_frames, 'FPS': round(vid_fps, 2),
            'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4),
            'TP': vid_tp, 'FP': vid_fp, 'FN': vid_fn, 'Video_Time': vid_time
        })

    logger.info("=" * 65)
    avg_fps = total_frames / total_time_sec if total_time_sec > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    logger.info("FINAL RESULTS (Baseline):")
    logger.info(f"Total Frames:   {total_frames}")
    logger.info(f"Average FPS:    {avg_fps:.2f}")
    logger.info(f"Precision:      {overall_prec:.4f}")
    logger.info(f"Recall:         {overall_rec:.4f}")
    logger.info(f"F1-Score:       {overall_f1:.4f}")
    logger.info("=" * 65)

    # Save results
    df = pd.DataFrame(results_data)
    output_path = Config.get_output_path("baseline")
    final_path = vis_utils.get_next_version_path(output_path)
    df.to_csv(final_path, index=False)
    logger.info(f"✅ CSV Saved: {final_path}")
    logger.info(f"⏱️  Process took: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
    
    # Update results tracker with summary metrics
    tracker.update_summary("baseline", {
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time_sec": time.time() - start_time,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn
    })

    return {
        "pipeline": "baseline",
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time,
        "output_file": final_path
    }
