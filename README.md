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
    *   Detects features (e.g., Shi-Tomasi corners) in the previous and current frames.
    *   Tracks these features using Lucas-Kanade Optical Flow.
    *   Calculates a **Homography Matrix** that maps the previous frame to the current one, effectively stabilizing the background against camera motion (pan, tilt).

2.  **Motion Mask Generation:**
    *   The previous frame is warped using the homography to align it with the current frame.
    *   A difference image is computed between the stabilized previous frame and the current frame. Any remaining differences are due to independent object motion.
    *   A **Dynamic Threshold** ($T = \mu + k\sigma$) is applied to the difference image to create a binary motion mask. This adapts to changing light conditions and environmental noise (wind, rain).
    *   Morphological operations (opening and dilation) are used to clean up the mask, removing small noise and closing gaps in detected motion blobs.

3.  **ROI-based YOLO Refiner:**
    *   **Proposal Generation:** The algorithm finds contours in the final motion mask. Each contour represents a "motion blob"—a candidate region where an object might be moving.
    *   **Filtering:** Contours are filtered based on area and aspect ratio to discard obvious noise (e.g., tiny pixel groups, long thin artifacts).
    *   **ROI Extraction & Expansion:** For each valid contour, a bounding box is created. This initial **Region of Interest (ROI)** is often too tight. To provide the detector with more visual context, the ROI is expanded by a configurable scaling factor (e.g., 3.0x). A minimum size (e.g., 192x192 pixels) is also enforced.
    *   **Selective Inference:** These expanded ROI crops are extracted from the original, full-color frame and passed to a **YOLO detector**. This is highly efficient, as the expensive detector only runs on a few small, targeted areas of the image instead of the entire 4K frame.
    *   **Persistence:** A simple object tracker is used to maintain object identity across frames, reducing flicker and improving stability.

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

---

## 8. Strategy 12: GMC + Dynamic Thresholding + Interpolation (`strategy_12.py`)
**Type:** Hybrid (Motion + Verifier + Interpolation)  
**Goal:** Enhance the throughput of the motion detection pipeline by strategically skipping frames and estimating object paths. This strategy is a direct enhancement of **Strategy 2**.

### Algorithm Logic
1.  **Core Motion Detection:** It uses the same motion detection pipeline as **Strategy 2**: Global Motion Compensation (GMC) followed by dynamic thresholding to generate Regions of Interest (ROIs) where motion is occurring.
2.  **Frame Skipping (Temporal Scheduling):** Instead of running the expensive motion analysis and YOLO detection on every single frame, it only processes "keyframes" every `N` frames (set by `detect_every`).
3.  **Linear Interpolation:** For the `N-1` frames between these keyframes, the pipeline does not perform any detection. It estimates the positions of birds by taking the bounding boxes from the previous keyframe and the current one and linearly interpolating their position and size.
4.  **Efficiency vs. Accuracy:** This approach significantly boosts the frames-per-second (FPS) by reducing the total number of inferences. The trade-off is a potential decrease in accuracy for objects that exhibit highly non-linear motion between keyframes.

---

## 9. Strategy 13: Motion-Gated Classifier Funnel (`strategy_13.py`)
**Type:** Multi-Stage Hybrid Detector (Motion -> Classification -> Detection)  
**Goal:** Achieve maximum efficiency by creating a "funnel" that uses a cascade of fast, cheap tests to progressively discard uninteresting image regions, ensuring the expensive detector only runs where absolutely necessary. This strategy is a hybrid of **Strategy 10** and **Strategy 11**.

### Algorithm Logic
The pipeline creates a highly efficient "gated funnel" to decide which parts of an image are worth a full detection scan.
1.  **GMC & Tiling:** The frame is stabilized using Global Motion Compensation and divided into a grid of tiles.
2.  **Stage 1: Motion Gate (from Strategy 10):** For each tile, a fast check for independent motion is performed. If motion is detected, the tile is considered "active" and is sent directly to the final detection stage.
3.  **Stage 2: Classifier Gate (from Strategy 11):** Tiles that were static (i.e., failed the motion check) are then passed to a second, much faster test: a lightweight YOLO **classification model**. If this model predicts a bird is present with sufficient confidence, the tile is "rescued" and also marked as active. This critical step allows the pipeline to find birds that are stationary or moving too slowly for motion detection to catch.
4.  **Stage 3: High-Precision Detection:** Only the tiles that were passed by *either* the motion gate or the classifier gate are processed by the full, expensive YOLO **detection model**. All other tiles are skipped entirely.
5.  **Benefits:** This architecture combines the high recall of motion detection for moving targets with the ability of a classifier to find static targets, creating a robust system that dramatically reduces computational load without sacrificing detection capabilities.