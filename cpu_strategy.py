import os
import time
import glob
import cv2
import numpy as np
import sys
from vis_utils import calculate_iou

class CpuStrategy:
    def __init__(self, train_dir):
        self.train_dir = train_dir

    def align_frames(self, prev_gray, curr_gray):
        """Calculates camera motion and warps prev_gray to match curr_gray."""
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_pts is None: return None

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 4: return None

        H, mask = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        if H is None: return None

        height, width = prev_gray.shape
        warped_prev = cv2.warpPerspective(prev_gray, H, (width, height))
        return warped_prev

    def run(self, gt_data, progress_queue=None):
        video_folders = sorted(glob.glob(os.path.join(self.train_dir, '*')))
        video_folders = [f for f in video_folders if os.path.isdir(f)]

        results_data = []
        total_videos = len(video_folders)

        if progress_queue:
             progress_queue.put({'type': 'init', 'total_videos': total_videos, 'strategy': 'CPU_Strat'})

        for v_idx, video_path in enumerate(video_folders):
            video_name = os.path.basename(video_path)
            images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
            if not images: continue

            vid_tp = vid_fp = vid_fn = 0
            vid_start = time.time()
            n_frames = len(images)
            prev_gray = None

            for i, img_path in enumerate(images):
                if progress_queue and i % 50 == 0:
                    progress_queue.put({
                        'type': 'progress',
                        'strategy': 'CPU_Strat',
                        'video': video_name,
                        'frame': i + 1,
                        'total_frames': n_frames
                    })

                frame = cv2.imread(img_path)
                if frame is None: continue
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                preds = []

                if prev_gray is not None:
                    warped_prev = self.align_frames(prev_gray, curr_gray)
                    if warped_prev is not None:
                        diff = cv2.absdiff(curr_gray, warped_prev)
                        _, thresh = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
                        kernel = np.ones((3,3), np.uint8)
                        thresh = cv2.dilate(thresh, kernel, iterations=2)
                        thresh = cv2.erode(thresh, kernel, iterations=1)
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        for cnt in contours:
                            area = cv2.contourArea(cnt)
                            if 100 < area < 2000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                h_img, w_img = curr_gray.shape
                                border = 15
                                if x > border and y > border and (x+w) < (w_img-border) and (y+h) < (h_img-border):
                                    preds.append([x, y, w, h])
                
                prev_gray = curr_gray
                
                key = f"{video_name}/{os.path.basename(img_path)}"
                gts = gt_data.get(key, [])
                matched_gt = set()

                for p_box in preds:
                    best_iou = 0
                    best_idx = -1
                    for idx, g_box in enumerate(gts):
                        if idx in matched_gt: continue
                        iou = calculate_iou(p_box, g_box)
                        if iou > best_iou: best_iou = iou; best_idx = idx
                    
                    if best_iou >= 0.20:
                        vid_tp += 1
                        matched_gt.add(best_idx)
                    else:
                        vid_fp += 1
                vid_fn += len(gts) - len(matched_gt)

            vid_time = time.time() - vid_start
            fps = len(images) / vid_time if vid_time > 0 else 0
            prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
            rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            results_data.append({
                'Video': video_name, 'Frames': len(images), 'FPS': round(fps, 2),
                'Precision': round(prec, 4), 'Recall': round(rec, 4), 'F1': round(f1, 4),
                'TP': vid_tp, 'FP': vid_fp, 'FN': vid_fn
            })

            if progress_queue:
                progress_queue.put({
                    'type': 'video_done',
                    'strategy': 'CPU_Strat',
                    'video': video_name,
                    'stats': results_data[-1]
                })

        return results_data
