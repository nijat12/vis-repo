# Bird Detection Strategies: From Deep Learning to Motion Physics

This repository implements a suite of Computer Vision pipelines designed to detect small, fast-moving birds in 4K video. The strategies range from standard Deep Learning approaches (YOLO) to classical Computer Vision techniques (Optical Flow, Frame Differencing) and hybrid "Search & Verify" models.

## 🏆 Performance Benchmark (Video 0002)

The following table summarizes the performance on **Video 0002** (Standard Flight Conditions).

| Strategy | Precision | Recall | F1-Score | TP | FP | FN | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 0.1680 | **0.3084** | **0.2175** | **305** | 1510 | 684 | **Highest Recall.** Finds the most birds but generates significant noise. |
| **Strategy 9** | **1.0000** | 0.0091 | 0.0180 | 9 | **0** | 980 | **Perfect Precision.** SAHI is extremely accurate but misses 99% of targets (likely due to strict thresholds). |
| **Strategy 8** | 0.6000 | 0.0030 | 0.0060 | 3 | 2 | 986 | **High Precision.** Very clean, but the ROI proposals are missing almost all birds. |
| **Strategy 2** | 0.5000 | 0.0354 | 0.0661 | 35 | 35 | 954 | Balanced Precision, but Motion Detection is filtering out too many real birds. |
| **Strategy 10**| 0.4262 | 0.0263 | 0.0495 | 26 | 35 | 963 | Similar performance to Strategy 2; Motion Gating is too aggressive. |
| **Strategy 7** | 0.1018 | 0.1132 | 0.1072 | 112 | 988 | 877 | Second best Recall, but the CNN Verifier (MobileNet) is letting too much noise through. |

---

## 1. Baseline: Tiled YOLO Inference (`baseline.py`)
**Type:** Deep Learning (Single Stage)  
**Goal:** Overcome the "small object problem" by artificially increasing input resolution via fixed tiling.

### Algorithm Logic
1.  **Tiling:** The 4K source frame ($3840 \times 2160$) is sliced into a **4x3 Grid (12 tiles)**.
    * *Overlap:* Tiles include overlap to prevent objects from being split at boundaries.
2.  **Batch Inference:** All 12 tiles are stacked and fed into a **YOLO** model in a single batch.
3.  **Merger:** Detections are mapped back to global 4K coordinates. Global NMS (Non-Maximum Suppression) merges duplicate detections at tile borders.

---

## 2. Strategy 2: GMC + Dynamic Thresholding (`strategy_2.py`)
**Type:** Hybrid (Motion + Verifier)  
**Goal:** Detect objects based on independent movement, using a lightweight verifier to reduce false positives.

### Algorithm Logic
1.  **Global Motion Compensation (GMC):**
    * Detects features (Shi-Tomasi) and tracks them (Lucas-Kanade Optical Flow).
    * Computes a **Homography Matrix** to stabilize the background against camera motion.
2.  **Dynamic Thresholding:**
    * Computes difference between stabilized frames.
    * Uses adaptive thresholding ($T = \mu + 4\sigma$) to handle variable environmental noise (e.g., wind).
3.  **YOLO Refiner:**
    * Extracts Regions of Interest (ROIs) from motion blobs.
    * Passes crops to **YOLO** to confirm if the moving blob is a bird.

---

## 3. Strategy 7: Motion + MobileNetV3 (`strategy_7.py`)
**Type:** Hybrid (Optical Flow + CNN)  
**Goal:** A lightweight alternative to YOLO, using a specialized CNN for binary "Bird/No-Bird" classification.

### Algorithm Logic
1.  **Sparse Optical Flow:** Generates a dense grid of flow vectors to detect relative motion (Egomotion cancellation).
2.  **Hysteresis Thresholding:** Applies high-pass filtering to isolate fast-moving objects.
3.  **CNN Verification:**
    * Crops motion candidates.
    * Feeds crops into **MobileNetV3-Small** (pretrained on ImageNet).
    * Checks for bird-related classes (sparrow, eagle, etc.) to validate the detection.

---

## 4. Strategy 8: YOLO on ROIs (`strategy_8.py`)
**Type:** Two-Stage Detector (Motion-Guided Proposal Based)  
**Goal:** Maximize efficiency by only running YOLO on "potentially interesting" parts of the image, significantly reducing computational overhead in 4K video.

### Algorithm Logic
1.  **Global Motion Compensation (GMC):** To prevent the algorithm from being triggered by camera movement, it aligns the previous frame to the current frame using a homography matrix calculated from tracked features.
2.  **Motion Proposals:** Computes the absolute difference zwischen stabilized frames and applies morphological operations (Close, Dilate) to isolate moving blobs. Contours are filtered by area ($50 < A < 5000$) to identify candidate bird locations.
3.  **Context-Aware ROI Expansion:** Each motion candidate is expanded by a scale factor (e.g., 2.0x) and constrained to a minimum size (e.g., $256 \times 256$). This ensures the bird is fully centered and provides enough local context for the YOLO model to succeed.
4.  **Batched Selective Inference:**
    *   **Partial Scanning:** YOLO is run *only* on the extracted crops, which are processed in a single batch for high throughput.
    *   **Temporal Scheduling:** To further optimize, inference is run only every $N$ frames (`detect_every`).
    *   **Full-Frame Refresh:** Periodically (every $M$ frames), a full scan of the image is performed to verify the entire scene and detect objects that may have been missed by motion detection.
5.  **Tracking & Persistence:** High-confidence detections are passed to an internal `ObjectTracker` to maintain stable IDs across frames.

---

## 5. Strategy 9: SAHI + Kalman Tracker (`strategy_9.py`)
**Type:** Slicing Aided Hyper Inference (SAHI) + Tracking  
**Goal:** The most robust pipeline for tiny objects, integrating temporal tracking to fix missed detections.

### Algorithm Logic
1.  **SAHI Slicing:** Slices the image into $640 \times 640$ patches (standard sliding window).
2.  **Detection:** Runs YOLO on every slice.
3.  **Tracking (Kalman + Hungarian):**
    * Feeds detections into a **Kalman Filter** (Constant Velocity Model).
    * Uses the **Hungarian Algorithm** (with IoU or Center Distance cost) to link detections across frames.
    * *Smoothing:* Can predict bird position even if detection fails for a few frames (occlusion/blur).

---

## 6. Strategy 10: Motion-Gated Native Tiling (`strategy_10.py`)
**Type:** Hybrid Efficiency-Precision Optimized Detector  
**Goal:** Achieve maximum pixel accuracy for tiny bird detection while maintaining high throughput by gating inference with motion analysis.

### Algorithm Logic
1.  **Stage 1: Background Stabilization (GMC):** Similar to Strategy 8, it uses a Homography-based Global Motion Compensation to stabilize the 4K frame, ensuring that subsequent motion analysis only detects independently moving objects.
2.  **Stage 2: Motion-Gated Tiling:**
    *   **Native Grid:** The frame is divided into a fixed grid of $640 \times 640$ tiles (the native resolution of the YOLO model).
    *   **Dynamic Thresholding:** A difference frame is calculated and an adaptive threshold ($T = \mu + k\sigma$) is applied.
    *   **Activity Selection:** For each tile, the algorithm calculates the percentage of "active" pixels. Only tiles exceeding a configurable `motion_pixel_threshold` are marked for processing.
3.  **Stage 3: Native Resolution Inference:**
    *   Instead of resizing the entire 4K image down to $640 \times 640$ (which destroys pixel-level detail of small birds), the algorithm runs YOLO *only* on the active native-resolution crops.
    *   This preserves every single pixel of the target, maximizing the model's ability to distinguish birds from noise.
4.  **Stage 4: Strategic Full Scans (Keyframes):**
    *   To prevent "track loss" if a bird temporarily stops moving relative to the background (or for objects already present), the algorithm triggers a **Full Scan Keyframe** every $N$ frames.
    *   During a keyframe, *every* tile in the grid is processed, regardless of motion.
5.  **Post-Processing:** Detections from active tiles are merged across overlap boundaries using Global NMS.

---

## 7. Strategy 11: ROI Classifier Filter + Detector (`strategy_11.py`)
**Type:** Three-Stage Hybrid Detector (Motion -> Classification -> Detection)  
**Goal:** Optimal efficiency and precision by using a "Fail-Fast" architecture.

### Algorithm Logic
1.  **Stage 1: Motion Proposals (GMC + Differencing):** Uses stabilized frame differencing to identify candidate motion blobs, exactly like Strategy 8.
2.  **Stage 2: Lightweight Classification (The "Gater"):**
    *   Each candidate ROI crop is first passed to a **YOLO Classification model** (e.g., `yolo12n-cls`).
    *   This model is significantly faster than the detection model.
    *   If the classifier's top prediction is not a bird or the confidence is below a threshold (`cls_conf_thresh`), the ROI is immediately discarded.
3.  **Stage 3: High-Precision Detection:**
    *   Only ROIs that were "verified" as bird-like by the classifier are passed to the **YOLO Detection model**.
    *   This reduces the number of expensive detection inferences, especially in noisy environments (moving leaves, wind).
4.  **Benefits:** Combines the high recall of motion detection with the speed of classification and the high precision of localized detection.