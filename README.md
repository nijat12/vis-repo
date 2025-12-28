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
**Type:** Two-Stage Detector (Proposal based)  
**Goal:** Maximize efficiency by only running YOLO on "interesting" parts of the image.

### Algorithm Logic
1.  **Motion Proposals:** Uses GMC and Frame Differencing to generate candidate bounding boxes.
2.  **ROI Expansion:** Expands the bounding boxes (scale 2.0x) to provide context.
3.  **Selective Inference:**
    * Instead of processing the whole 4K frame, it runs **YOLO** *only* on the cropped ROIs.
    * *Benefit:* Extremely fast on empty skies; scales computation linearly with the number of moving objects.

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

## 6. Strategy 10: Motion Proposals + YOLO Classifier (`strategy_10.py`)
**Type:** Hybrid Two-Stage Detector  
**Goal:** High-recall motion detection followed by high-precision classification.

### Algorithm Logic
1.  **Stage 1: Motion Proposals:**
    * Uses GMC to stabilize the frame.
    * Applies dynamic thresholding to find moving "blobs".
2.  **Stage 2: Precision Verification:**
    * **Crop & Pad:** Crops the moving blob from the high-res 4K frame with ~20px padding.
    * **Classification:** Feeds the crop into **YOLO** (Classification Mode).
3.  **Filtering:**
    * If YOLO confirms the crop contains a bird, the detection is kept.
    * If YOLO sees noise/leaves, the detection is discarded.