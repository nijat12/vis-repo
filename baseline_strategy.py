import os
import time
import glob
import torch
import cv2
import sys
from vis_utils import calculate_iou

# Constants
MODEL_NAME = 'yolov5l6'
IMG_SIZE = 1280
CONF_THRESH = 0.05
IOU_THRESH = 0.45

class BaselineStrategy:
    def __init__(self, train_dir, device='cuda'):
        self.train_dir = train_dir
        self.device_name = device
        self.model = None

    def load_model(self):
        try:
            self.model = torch.hub.load('ultralytics/yolov5', MODEL_NAME, pretrained=True, force_reload=False)
            self.model.conf = CONF_THRESH
            self.model.classes = [14]  # Class 14 = Bird
            
            if torch.cuda.is_available() and self.device_name == 'cuda':
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
                
            self.model.to(self.device)
            return True
        except Exception as e:
            print(f"❌ Model Load Error: {e}", file=sys.stderr)
            return False

    def run(self, gt_data, progress_queue=None):
        if self.model is None:
            if not self.load_model():
                return []

        video_folders = sorted(glob.glob(os.path.join(self.train_dir, '*')))
        video_folders = [f for f in video_folders if os.path.isdir(f)][:1]

        results_data = []
        total_videos = len(video_folders)
        
        # Report Total Videos found
        if progress_queue:
            progress_queue.put({'type': 'init', 'total_videos': total_videos, 'strategy': 'Baseline'})

        for v_idx, video_path in enumerate(video_folders):
            video_name = os.path.basename(video_path)
            images = sorted(glob.glob(os.path.join(video_path, '*.jpg')))
            
            if not images: continue
            
            n_frames = len(images)
            vid_tp = vid_fp = vid_fn = 0
            vid_start = time.time()
            
            for i, img_path in enumerate(images):
                # Send frame progress
                if progress_queue and i % 10 == 0:
                    progress_queue.put({
                        'type': 'progress',
                        'strategy': 'Baseline',
                        'video': video_name,
                        'frame': i + 1,
                        'total_frames': n_frames
                    })

                img_filename = os.path.basename(img_path)
                lookup_key = f"{video_name}/{img_filename}"
                
                img = cv2.imread(img_path)
                if img is None: continue
                
                results = self.model(img, size=IMG_SIZE)
                preds = []
                # Handle results
                results_numpy = results.xyxy[0].cpu().numpy() if torch.cuda.is_available() else results.xyxy[0].numpy()
                for det in results_numpy:
                    x1, y1, x2, y2, conf, cls = det
                    preds.append([x1, y1, x2-x1, y2-y1])
                
                gts = gt_data.get(lookup_key, [])
                matched_gt = set()
                
                for p_box in preds:
                    best_iou = 0
                    best_gt_idx = -1
                    for idx, g_box in enumerate(gts):
                        if idx in matched_gt: continue
                        iou = calculate_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                    
                    if best_iou >= IOU_THRESH:
                        vid_tp += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        vid_fp += 1
                vid_fn += len(gts) - len(matched_gt)

            vid_time = time.time() - vid_start
            vid_fps = n_frames / vid_time if vid_time > 0 else 0
            
            prec = vid_tp / (vid_tp + vid_fp) if (vid_tp + vid_fp) > 0 else 0
            rec = vid_tp / (vid_tp + vid_fn) if (vid_tp + vid_fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            results_data.append({
                'Video': video_name,
                'Frames': n_frames,
                'FPS': round(vid_fps, 2),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1': round(f1, 4),
                'TP': vid_tp, 'FP': vid_fp, 'FN': vid_fn
            })

            # Send video complete signal
            if progress_queue:
                progress_queue.put({
                    'type': 'video_done', 
                    'strategy': 'Baseline',
                    'video': video_name,
                    'stats': results_data[-1]
                })

        return results_data
