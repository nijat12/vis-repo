# Bird Detection Strategies: From Deep Learning to Motion Physics

This repository implements a suite of Computer Vision pipelines designed to detect small, fast-moving birds in 4K video. The strategies range from standard Deep Learning approaches (YOLO) to classical Computer Vision techniques (Optical Flow, Frame Differencing) and hybrid "Search & Verify" models.

## 🏆 Performance Benchmark (Jan 1st Run)

The following table summarizes the performance on the test set.

| Strategy | Avg FPS | Precision | Recall | F1-Score | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **baseline_base_001** | 0.69 | 3.41% | 0.10% | 0.19% | Reference Baseline. Very low recall. |
| **baseline_base_001_sahi** | 0.44 | 2.39% | 54.50% | 4.58% | Massive recall boost from SAHI. |
| **strategy_2_001** | 0.41 | 6.73% | 2.18% | 3.29% | Motion filtering improves precision. |
| **strategy_2_001_sahi** | 0.03 | 2.15% | **62.30%** | 4.15% | **Highest Recall**, but extremely slow (0.03 FPS). |
| **strategy_8_001** | 1.48 | **10.20%** | 0.80% | 1.48% | Best Precision (non-SAHI). |
| **strategy_8_001_sahi** | 1.82 | 4.65% | 38.66% | **8.29%** | **Highest F1-Score**. Good speed/accuracy balance. |
| **strategy_10_001** | 0.16 | 2.38% | 54.18% | 4.56% | Native Tiling matches SAHI recall but slower. |
| **strategy_10_001_sahi** | 0.43 | 2.39% | 54.50% | 4.58% | Identical to Baseline SAHI. |
| **strategy_11_001** | 0.90 | 0.20% | 9.04% | 0.40% | Classifier stage seems to hurt precision here. |
| **strategy_12_001** | **3.54** | 7.85% | 0.87% | 1.56% | **Fastest**. Interpolation boosts FPS significantly. |
| **strategy_12_001_sahi** | 2.25 | 2.53% | 43.89% | 4.78% | Excellent trade-off: High FPS & High Recall. |
| **strategy_13_001** | 0.16 | 2.30% | 52.23% | 4.40% | Complex hybrid, similar to Strat 10. |
| **strategy_13_001_sahi** | 0.46 | 2.39% | 54.50% | 4.58% | Converges to Baseline SAHI performance. |

---

# 📚 Detailed Strategy Pipelines

## 1. Baseline Base Approache (`baseline.py` - `baseline_base`)

The baseline module establishes the fundamental performance metrics using standard YOLO inference techniques.

### 1.1 Baseline Base: Naive Inference
**Type:** Single-Stage Global Inference
**Goal:** Establish a "lower bound" for performance using standard YOLO usage.

**Mechanism:**
1.  **Resizing:** The 4K input image ($3840 \times 2160$) is resized down to the model's native input size (e.g., $640 \times 640$).
2.  **Inference:** A single forward pass of the YOLO model is performed.
3.  **Limitations:** Small birds (often < 20 pixels wide) are decimated during resizing, vanishing completely or becoming indistinguishable from noise.

#### 🔄 SAHI Behavior
When `use_sahi` is enabled, this pipeline effectively transforms into a **brute-force tiled inference** engine. Instead of resizing the 4K image, it slices it into overlapping patches and runs YOLO on each.
*   **Impact:** Recall skyrockets (0.1% -> 54.5%) because small birds are preserved at native resolution.
*   **Cost:** Processing time increases significantly (FPS drops from 0.69 to 0.44).

#### 📊 Performance vs Baseline
Relative to `baseline_base_001` (Non-SAHI):
*   **Precision:** -30% (3.41% -> 2.39%) - More noise introduced.
*   **Recall:** +54,400% (0.10% -> 54.50%) - Massive improvement.
*   **F1-Score:** +2,310% (0.19% -> 4.58%).

---

## 2. Strategy 2: GMC + Dynamic Thresholding (`strategy_2.py`)
**Type:** Hybrid (Motion Proposal -> Detector Refinement)
**Goal:** Filter out 99% of the static background to focus compute only on moving objects.

**Pipeline:**
1.  **Global Motion Compensation (GMC):**
    *   **Feature Tracking:** Detects Shi-Tomasi corners and tracks them between $Frame_{t-1}$ and $Frame_t$ using Lucas-Kanade Optical Flow.
    *   **Homography:** Computes a transformation matrix to warp $Frame_{t-1}$ to align with $Frame_t$, cancelling out camera motion (pan/tilt/zoom).
2.  **Motion Detection:**
    *   **Frame Differencing:** Computes the absolute difference between the current frame and the aligned previous frame.
    *   **Dynamic Thresholding:** Calculates the mean ($\mu$) and standard deviation ($\sigma$) of the difference image. Sets an adaptive threshold $T = \mu + k\sigma$. This robustly handles changing lighting conditions better than fixed thresholds.
    *   **Morphology:** Applies Opening (Erosion followed by Dilation) to remove salt-and-pepper noise.
3.  **Proposal Generation:**
    *   Contours are found on the binary motion mask.
    *   **Filtering:** Contours are filtered by Area ($50 < Area < 5000$) and Aspect Ratio to discard noise.
4.  **YOLO Refiner:**
    *   **ROI Expansion:** Valid motion contours are converted to bounding boxes and expanded by a scale factor (e.g., 2.0x) to provide context.
    *   **Selective Inference:** YOLO is run *only* on these cropped Regions of Interest (ROIs).
5.  **Persistence:** A simple Object Tracker maintains identities to smooth out flickering detections.

#### 🔄 SAHI Behavior
Enabling SAHI (`strategy_2_..._sahi`) dramatically alters the behavior. It completely overrides the strategy's core logic.
*   **Skipped:** Global Motion Compensation (GMC), Dynamic Thresholding, Morphological Filtering, and ROI Generation.
*   **Replaced With:** Brute-force full-frame SAHI inference.
*   **Impact:** Recall jumps to **62.30%** (highest in benchmark) because strict motion filters no longer discard subtle birds.
*   **Cost:** Extreme performance penalty. FPS drops to **0.03**, making it unsuitable for real-time use.

#### 📊 Performance vs Baseline
Relative to `baseline_base_001`:
*   **Precision:** -37% (3.41% -> 2.15%).
*   **Recall:** +62,200% (0.10% -> 62.30%).
*   **F1-Score:** +2,084% (0.19% -> 4.15%).

---

## 3 Baseline + Tiling: Grid Inference (`baseline.py` - `baseline_w_tiling`)
**Type:** Tiled Inference (Batch Processing)
**Goal:** Solve the small object problem by maintaining high resolution.

**Mechanism:**
1.  **Grid Generation:** The 4K frame is divided into a fixed **4x3 grid (12 tiles)**.
2.  **Overlap:** A 20% overlap is applied between adjacent tiles to ensure objects sitting on the "seam" are not split and missed.
3.  **Resolution:** Each tile is extracted at near-native resolution, preserving the fine details of small birds.
4.  **Batch Inference:** All 12 tiles are stacked into a single tensor and processed in one batch for GPU efficiency.
5.  **Coordinate Mapping:** Detections in tile-space are mapped back to global 4K image coordinates.

---

## 4 Baseline + Tiling + NMS (`baseline.py` - `baseline_w_tiling_nms`)
**Type:** Tiled Inference with Global Post-Processing
**Goal:** Remove duplicate detections caused by tile overlaps.

**Mechanism:**
1.  **Steps 1-5:** Same as *Baseline Tiling*.
2.  **Global NMS:** After mapping all detections to global coordinates, a Global Non-Maximum Suppression (NMS) pass is applied. This merges multiple bounding boxes that refer to the same object (common in the overlap regions), calculating the final box and confidence score.

---

## 5. Strategy 7: Motion + MobileNetV3 (`strategy_7.py`)
**Type:** Classical Computer Vision + Light CNN Classifier
**Goal:** A "YOLO-free" approach relying on classical physics and a lightweight classifier for verification.

**Pipeline:**
1.  **Advanced Motion Masking:**
    *   **GMC:** Stabilizes the frame (as in Strat 2).
    *   **High-Pass Filter:** Subtracts a blurred version of the difference frame to highlight sharp changes (edges/motion).
    *   **Optical Flow:** Computes dense Farneback Optical Flow. Magnitude thresholding identifies coherent motion.
    *   **DoG:** Difference of Gaussians is used to spot "blob-like" objects (birds).
    *   **Fusion:** A combined mask is generated via `(HighPass OR (Flow AND DoG))`.
2.  **Scoring System:**
    *   Candidates are extracted from contours.
    *   A heuristic score is calculated: $Score = w_1 \cdot Diff + w_2 \cdot DoG + w_3 \cdot Flow$.
3.  **MobileNet Verification:**
    *   Candidate crops are fed into **MobileNetV3-Small** (pretrained on ImageNet).
    *   **Birdness Score:** The model checks for semantic classes related to birds (eagle, kite, sparrow, etc.).
    *   If the "Birdness" probability exceeds a threshold, the candidate is accepted.
4.  **Note:** This strategy generates bounding boxes purely from motion contours, not regression.

---

## 6. Strategy 8: YOLO on ROIs (`strategy_8.py`)
**Type:** Motion-Guided Two-Stage Detector
**Goal:** Maximize 4K throughput by treating detection as a "verification" step for motion.

**Pipeline:**
1.  **Motion Proposals:** Uses the stabilized frame differencing (GMC) method to find moving blobs.
2.  **Context-Aware ROI Expansion:**
    *   Motion blobs are often tight to the object edges.
    *   The ROI is expanded significantly (scale 2.0x, min 256px) to ensure the YOLO detector sees the "bird in the sky" context, not just feathers.
3.  **Temporal Scheduling (`detect_every`):**
    *   Inference is not run every frame. It runs every $N$ frames to save compute.
    *   Between detection frames, the Object Tracker holds the state.
4.  **Strategic Full Scans:**
    *   To catch stationary birds or recover from motion failures, a **Full Frame Scan** is triggered every $M$ frames.
5.  **Selective Inference:** YOLO runs on the batch of ROIs. If no motion is detected, the expensive detector is skipped entirely for that frame.

#### 🔄 SAHI Behavior
Surprisingly, enabling SAHI on this strategy **improves FPS** (1.48 -> 1.82) while boosting Recall significantly.
*   **Skipped:** Motion Proposal Generation. The system no longer looks for moving blobs to define ROIs.
*   **Retained:** `detect_every` scheduling.
*   **Replaced With:** Full-frame SAHI inference running at the scheduled intervals (e.g., every 5 frames).
*   **Impact:** Recall improves from 0.8% to 38.66%.
*   **F1-Score:** Reaches **8.29%**, the highest in the benchmark.

#### 📊 Performance vs Baseline
Relative to `baseline_base_001`:
*   **Precision:** +36% (3.41% -> 4.65%).
*   **Recall:** +38,560% (0.10% -> 38.66%).
*   **F1-Score:** +4,263% (0.19% -> 8.29%).

---

## 7. Strategy 9: SAHI + Kalman Tracker (`strategy_9.py`)
**Type:** Slicing Aided Hyper Inference (SAHI) + Advanced Tracking
**Goal:** The "Brute Force" precision approach.

**Pipeline:**
1.  **SAHI Slicing:** The image is sliced into overlapping $640 \times 640$ patches covering the entire 4K frame.
2.  **Full Coverage Inference:** YOLO is run on *every* slice. This ensures no small bird is lost due to resizing.
3.  **Global Merger:** Detections from all slices are projected to global coordinates and merged via NMS.
4.  **Kalman Filter Tracking:**
    *   Each detection initializes a **Kalman Filter** state (estimating Position + Velocity).
    *   **prediction step:** The filter predicts where the bird will be in the next frame.
5.  **Hungarian Association (DotD):**
    *   New detections are matched to existing tracks using the **Hungarian Algorithm**.
    *   **Cost Metric:** Instead of IoU (which fails for tiny, fast objects), it uses **DotD (Distance of the Detection)**—the Euclidean distance between the predicted center and the detected center.

---

## 8. Strategy 10: Motion-Gated Native Tiling (`strategy_10.py`)
**Type:** Tiled Hybrid
**Goal:** Combine the precision of Tiling with the efficiency of Motion Gating.

**Pipeline:**
1.  **Native Grid:** The frame is logically divided into fixed $640 \times 640$ tiles (matching YOLO's native resolution).
2.  **GMC Stabilization:** Background motion is cancelled out.
3.  **Active Tile Selection:**
    *   Motion difference is calculated for the whole frame.
    *   The algorithm checks each tile: *Does this tile contain significant motion pixels?*
    *   **Active Tiles:** Tiles with motion are added to the inference batch.
    *   **Inactive Tiles:** Tiles with only sky/background are skipped.
4.  **Inference:** YOLO runs on the Active Tiles without resizing.
5.  **Result:** 1:1 pixel accuracy for moving objects, with 0 compute wasted on empty sky.

#### 🔄 SAHI Behavior & Comparison
Since Strategy 10 is already a "Native Tiling" approach, enabling SAHI has **negligible impact** on detection performance but changes the mechanism.
*   **Skipped:** Motion Gating (Active Tile Selection). The system processes *all* tiles regardless of whether motion was detected.
*   **Replaced With:** Standard SAHI tiled inference.
*   **Observation:** `strategy_10_001` and `strategy_10_001_sahi` have nearly identical metrics (Recall ~54%, F1 ~4.58%).
*   **Convergence:** Motion Gated Tiling (Strat 10) effectively converges to Brute Force Tiling (Baseline SAHI).

#### 📊 Performance vs Baseline
Relative to `baseline_base_001`:
*   **Recall:** +54,080% (0.10% -> 54.18%).
*   **FPS:** Slower (0.16 vs 0.69).

---

## 9. Strategy 11: ROI Classifier Filter + Detector (`strategy_11.py`)
**Type:** Three-Stage Cascade (Motion -> Classify -> Detect)
**Goal:** "Fail Fast" architecture to minimize heavy detection compute.

**Pipeline:**
1.  **Stage 1: Motion Proposals:** Candidate ROIs are generated via GMC + Frame Differencing.
2.  **Stage 2: The Classifier Gate:**
    *   ROIs are passed to a **YOLO-Classify** model (e.g., `yolo12n-cls`).
    *   This model is extremely fast/lightweight compared to the detector.
    *   **Check:** Is the content "Bird-like"?
    *   If NO: The ROI is discarded immediately.
3.  **Stage 3: The Detector:**
    *   Only "verified" ROIs reach the **YOLO-Detect** model.
    *   This filters out moving leaves, clouds, or compression artifacts that might look like motion but aren't birds.

#### 🔄 SAHI Behavior
**Critical Failure:** Enabling SAHI (`strategy_11_001_sahi`) results in **0% Recall**.
*   **Skipped:** The entire Classifier Gate (Stage 2) and Motion Proposals (Stage 1). The pipeline bypasses the "Fail Fast" logic.
*   **Replaced With:** Full-frame detector inference via SAHI.
*   **Diagnosis:** Despite bypassing the classifier, the implementation exhibits incompatibility (possibly due to tracking or scheduling conflicts in the SAHI branch), rendering it ineffective. Do not use SAHI with Strategy 11.

---

## 10. Strategy 12: GMC + Interpolation (`strategy_12.py`)
**Type:** Temporal Optimization
**Goal:** Maximize FPS by skipping frames and mathematically interpolating positions.

**Pipeline:**
1.  **Keyframe Processing:** Every $N$ frames (e.g., 5), the full detection pipeline (Strategy 2: GMC + YOLO Refiner) is run.
2.  **Linear Interpolation:**
    *   For the intermediate frames ($1..N-1$), no inference is run.
    *   The system takes the bounding boxes from Keyframe $A$ and Keyframe $B$.
    *   It calculates a linear path for each object: $Pos_t = Pos_A + (Pos_B - Pos_A) \times \frac{t}{N}$.
3.  **Result:** Extremely high FPS. Accuracy depends on the linearity of the bird's flight.

#### 🔄 SAHI Behavior
Strategy 12 pairs excellently with SAHI. The temporal interpolation mitigates the heavy cost of SAHI inference.
*   **Impact:** We get the high recall of SAHI (43.89%) with a respectable real-time framerate (**2.25 FPS**).
*   **Comparison:** This is the most balanced "High Performance" configuration.

#### 📊 Performance vs Baseline
Relative to `baseline_base_001`:
*   **Precision:** -25% (3.41% -> 2.53%).
*   **Recall:** +43,790% (0.10% -> 43.89%).
*   **FPS:** 3.2x faster (0.69 -> 2.25).

---

## 11. Strategy 13: Motion-Gated Classifier Funnel (`strategy_13.py`)
**Type:** The "Kitchen Sink" Hybrid
**Goal:** A funnel that catches everything while costing as little as possible.

**Pipeline:**
1.  **Tiling:** Frame is divided into tiles.
2.  **Gate 1: Motion Check:**
    *   Is there motion in the tile?
    *   **Yes:** Send to Detector (High Confidence of moving object).
    *   **No:** Send to Gate 2.
3.  **Gate 2: Classifier Check:**
    *   Run lightweight Classifier on the static tile.
    *   Is there a stationary bird (e.g., perched)?
    *   **Yes:** Send to Detector.
    *   **No:** Discard tile.
4.  **Gate 3: Detector:**
    *   Run YOLO inference on tiles passed by Gate 1 OR Gate 2.
5.  **Interpolation:** Can optionally apply Strat 12's interpolation logic between frames.
6.  **Summary:** Catches moving birds (via Motion), catches static birds (via Classifier), and skips empty sky (via Logic).

#### 🔄 SAHI Behavior & Comparison
Similar to Strategy 10, Strategy 13 converges to the performance of the Baseline SAHI when SAHI is enabled.
*   **Observation:** `strategy_13_001_sahi` metrics (Recall 54.50%, F1 4.58%) are identical to `baseline_base_001_sahi`.
*   **Conclusion:** The complex gating logic becomes redundant when SAHI's slicing mechanism takes over, effectively scanning the whole image (or all relevant slices).

#### 📊 Performance vs Baseline
Relative to `baseline_base_001`:
*   **Recall:** +52,130% (0.10% -> 52.23%).
*   **F1-Score:** +2,215% (0.19% -> 4.40%).
