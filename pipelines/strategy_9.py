"""
Strategy 9 Pipeline: SAHI Slicing + YOLO + Kalman/Hungarian (DotD)

Implements the architecture defined in the class diagram:
1. Slicing: 3840x2160 -> 640x640 slices (with overlap)
2. Detection: YOLO (pretrained) on slices
3. Merger: NMS to combine slice detections
4. Tracking: Kalman Filter + Hungarian Algorithm (DotD cost)
"""

import os
import glob
import time
import datetime
import logging
import numpy as np
import pandas as pd
import cv2
import torch
from scipy.optimize import linear_sum_assignment

# Attempt to import ultralytics for YOLO
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from config import Config
import vis_utils
import csv_utils
from pipelines import register_pipeline

logger = logging.getLogger(__name__)

# =========================================================================
#  Helper: Kalman Filter + Hungarian Tracker (The "Strategy 9" Logic)
# =========================================================================

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    Uses a Constant Velocity model standard in multi-object tracking.
    """
    count = 0
    def __init__(self, bbox):
        # Initialize state vector [u, v, s, r, u_dot, v_dot, s_dot]
        from filterpy.kalman import KalmanFilter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        
        # State Transition Matrix F
        self.kf.F = np.array([[1,0,0,0,1,0,0], 
                              [0,1,0,0,0,1,0], 
                              [0,0,1,0,0,0,1], 
                              [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0], 
                              [0,0,0,0,0,1,0], 
                              [0,0,0,0,0,0,1]])
        
        # Measurement Function H
        self.kf.H = np.array([[1,0,0,0,0,0,0], 
                              [0,1,0,0,0,0,0], 
                              [0,0,1,0,0,0,0], 
                              [0,0,0,1,0,0,0]])

        # Covariance Matrices (tuned for small objects)
        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        """
        Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
        [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
        """
        w = bbox[2]-bbox[0]
        h = bbox[3]-bbox[1]
        x = bbox[0]+w/2.
        y = bbox[1]+h/2.
        s = w*h
        r = w/float(h) if h > 0 else 1.0
        return np.array([x,y,s,r]).reshape((4,1))

    def convert_x_to_bbox(self, x, score=None):
        """
        Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        """
        w = np.sqrt(x[2]*x[3])
        h = x[2]/w
        if(score==None):
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class SortDotDTracker:
    """
    SORT Tracker modified to use 'DotD' (Distance based cost) for the Hungarian Algorithm.
    Recommended for small objects where IoU is unreliable.
    """
    def __init__(self, max_age=15, min_hits=3, dist_threshold=100.0):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.trackers = []
        self.frame_count = 0

    def get_dotd_cost_matrix(self, trackers, detections):
        """
        Compute cost matrix using Euclidean distance between centers.
        """
        if len(trackers) == 0 or len(detections) == 0:
            return np.zeros((len(trackers), len(detections)))
            
        cost_matrix = np.zeros((len(trackers), len(detections)))
        
        for t, trk in enumerate(trackers):
            # Tracker state: [x1, y1, x2, y2]
            t_box = trk.get_state()[0]
            t_cx = (t_box[0] + t_box[2]) / 2
            t_cy = (t_box[1] + t_box[3]) / 2
            
            for d, det in enumerate(detections):
                # Detection: [x1, y1, x2, y2]
                d_cx = (det[0] + det[2]) / 2
                d_cy = (det[1] + det[3]) / 2
                
                # Euclidean distance
                dist = np.sqrt((t_cx - d_cx)**2 + (t_cy - d_cy)**2)
                cost_matrix[t, d] = dist
                
        return cost_matrix

    def update(self, dets):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],...]
        """
        self.frame_count += 1
        
        # 1. Predict existing tracks using Kalman Filter
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # 2. Association (Hungarian Algorithm with DotD cost)
        matched, unmatched_dets, unmatched_trks = [], [], []
        
        if len(dets) > 0 and len(self.trackers) > 0:
            cost_matrix = self.get_dotd_cost_matrix(self.trackers, dets)
            
            # Hungarian Algorithm (linear_sum_assignment)
            row_inds, col_inds = linear_sum_assignment(cost_matrix)
            
            # Filter by threshold
            for r, c in zip(row_inds, col_inds):
                if cost_matrix[r, c] > self.dist_threshold:
                    unmatched_trks.append(r)
                    unmatched_dets.append(c)
                else:
                    matched.append([r, c])
                    
            # Handle unmatched
            for t in range(len(self.trackers)):
                if t not in row_inds:
                    unmatched_trks.append(t)
            for d in range(len(dets)):
                if d not in col_inds:
                    unmatched_dets.append(d)
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(self.trackers)))

        # 3. Update matched trackers
        for t, d in matched:
            self.trackers[t].update(dets[d, :])

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # 5. Output management
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            # Standard SORT logic: return if hit streak is good or age is young enough
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # Return format: [x, y, w, h] (converting back from x1,y1,x2,y2)
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # Remove dead tracks
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if len(ret) > 0:
            # Convert x1,y1,x2,y2 to x,y,w,h for output
            final_res = []
            for r in ret:
                r = r[0]
                final_res.append([r[0], r[1], r[2]-r[0], r[3]-r[1]])
            return final_res
            
        return []

# =========================================================================
#  Helper: Slicing Logic (SAHI style)
# =========================================================================

def slice_image(image, slice_wh=(640, 640), overlap_ratio=0.2):
    """
    Divides image into overlapping slices.
    Returns: list of (slice_img, x_offset, y_offset)
    """
    img_h, img_w, _ = image.shape
    slice_w, slice_h = slice_wh
    
    X_points = list(range(0, img_w - slice_w, int(slice_w * (1 - overlap_ratio))))
    Y_points = list(range(0, img_h - slice_h, int(slice_h * (1 - overlap_ratio))))
    
    # Add final points to ensure full coverage
    X_points.append(img_w - slice_w)
    Y_points.append(img_h - slice_h)
    
    # Remove duplicates
    X_points = sorted(list(set(X_points)))
    Y_points = sorted(list(set(Y_points)))
    
    slices = []
    for y in Y_points:
        for x in X_points:
            s_img = image[y:y+slice_h, x:x+slice_w]
            slices.append((s_img, x, y))
            
    return slices

def nms_global(detections, iou_thresh=0.5):
    """
    Merges overlapping detections from different slices.
    detections: list of [x1, y1, x2, y2, conf]
    """
    if len(detections) == 0:
        return []

    boxes = torch.tensor([d[:4] for d in detections], dtype=torch.float32)
    scores = torch.tensor([d[4] for d in detections], dtype=torch.float32)
    
    keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_thresh)
    
    return [detections[i] for i in keep_indices]


# =========================================================================
#  Pipeline Execution
# =========================================================================

@register_pipeline("strategy_9")
def run_strategy_9_pipeline():
    """
    Execute Strategy 9 Pipeline:
    SAHI Slicer -> YOLO -> NMS -> Kalman/Hungarian Tracker
    """
    logger.info("STARTING STRATEGY 9 PIPELINE (SAHI + YOLO + Kalman/Hungarian)")

    # 1. Configuration Setup
    try:
        cfg = Config.STRATEGY_9_CONFIG
    except AttributeError:
        logger.error("❌ STRATEGY_9_CONFIG not found in Config class!")
        raise

    # 2. Load Model (YOLO)
    if YOLO is None:
        logger.error("❌ ultralytics library not found. Please run: pip install ultralytics")
        raise ImportError("ultralytics library missing")

    logger.info(f"⏳ Loading YOLO Model: {cfg['model_path']}...")
    try:
        # Load the model specified in config
        model = YOLO(cfg['model_path']) 
        logger.info(f"✅ Model {cfg['model_path']} Loaded.")
    except Exception as e:
        logger.error(f"❌ Model Load Error: {e}")
        raise

    # 3. Load Ground Truth
    gt_data = vis_utils.load_json_ground_truth(Config.LOCAL_JSON_PATH)
    if not gt_data:
        raise RuntimeError("Failed to load ground truth data")

    start_time = time.time()

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
        
        # Initialize Strategy 9 Specific Tracker (Kalman + Hungarian)
        # Parameters pulled directly from config
        kf_tracker = SortDotDTracker(
            max_age=cfg['max_age'], 
            min_hits=cfg['min_hits'], 
            dist_threshold=cfg['tracker_dist_thresh']
        )

        for i, img_path in enumerate(images):
            img_start_time = time.time()
            
            if i % 50 == 0:
                percent = ((i + 1) / n_frames) * 100
                logger.info(f"👉 Processing [{video_name}] Frame {i+1}/{n_frames} ({percent:.1f}%)")

            frame = cv2.imread(img_path)
            if frame is None:
                continue

            # --- STEP 1: SAHI SLICING ---
            slices = slice_image(frame, slice_wh=(cfg['slice_size'], cfg['slice_size']), overlap_ratio=cfg['overlap'])
            
            # --- STEP 2: BATCH INFERENCE ---
            batch_imgs = [s[0] for s in slices]
            
            # Run Inference
            results = model(batch_imgs, verbose=False, conf=cfg['conf_thresh'], classes=[cfg['bird_class_id']])

            # --- STEP 3: MERGE DETECTIONS ---
            raw_detections = [] 

            for idx, res in enumerate(results):
                off_x, off_y = slices[idx][1], slices[idx][2]
                boxes = res.boxes
                for box in boxes:
                    # Get local coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Transform to global coordinates
                    g_x1 = x1 + off_x
                    g_y1 = y1 + off_y
                    g_x2 = x2 + off_x
                    g_y2 = y2 + off_y
                    
                    raw_detections.append([g_x1, g_y1, g_x2, g_y2, conf])
            
            # NMS Merger
            merged_detections = nms_global(raw_detections, iou_thresh=0.3)
            
            # Prepare for tracker
            if merged_detections:
                tracker_input = np.array(merged_detections)
            else:
                tracker_input = np.empty((0, 5))

            # --- STEP 4: TRACKING (Kalman + Hungarian) ---
            final_preds = kf_tracker.update(tracker_input)

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
            
            # Logging to CSV
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
            tracker.add_image_result("strategy_9", image_result)
            
            if (i + 1) % 50 == 0:
                tracker.save_batch("strategy_9", batch_size=50)

        # Video Stats
        vid_time = time.time() - vid_start
        fps = len(images) / vid_time if vid_time > 0 else 0
        prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
        rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        # Identical Table Row Formatting to Strategy 8
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

    # Final Summary - Identical Formatting to Strategy 8
    logger.info("=" * 65)
    avg_fps = total_frames / total_time if total_time > 0 else 0
    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0

    logger.info("FINAL RESULTS (Strategy 9 - SAHI+YOLO+Kalman):")
    logger.info(f"Total Frames:   {total_frames}")
    logger.info(f"Average FPS:    {avg_fps:.2f}")
    logger.info(f"Precision:      {overall_prec:.4f}")
    logger.info(f"Recall:         {overall_rec:.4f}")
    logger.info(f"F1-Score:       {overall_f1:.4f}")
    logger.info("=" * 65)

    df = pd.DataFrame(results_data)
    output_path = Config.get_output_path("strategy_9")
    final_path = vis_utils.get_next_version_path(output_path)
    df.to_csv(final_path, index=False)
    
    logger.info(f"✅ CSV Saved: {final_path}")
    logger.info(f"⏱️  Process took: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")
    
    tracker.update_summary("strategy_9", {
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time_sec": time.time() - start_time,
        "tp": total_tp, "fp": total_fp, "fn": total_fn
    })

    return {
        "pipeline": "strategy_9",
        "total_frames": total_frames,
        "avg_fps": avg_fps,
        "precision": overall_prec,
        "recall": overall_rec,
        "f1_score": overall_f1,
        "execution_time": time.time() - start_time,
        "output_file": final_path
    }