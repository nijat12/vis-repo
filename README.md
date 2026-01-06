# 🦅 Bird Detection Strategies: From Deep Learning to Motion Physics

This repository implements a suite of Computer Vision pipelines designed to detect small, fast-moving birds in 4K video. The strategies range from standard Deep Learning approaches (YOLO) to classical Computer Vision techniques (Optical Flow, Frame Differencing) and hybrid "Search & Verify" models.

## 🚀 Setup and Installation

### 1. Execution Environment 
#### 1.1 Zip file
The project is managed via a `setup.sh` script that automates the following:
- Installs system dependencies (`apt-get`).
- Installs `pyenv` and **Python 3.14.2**.
- Creates a Python virtual environment (`.venv`).
- Installs all required packages from `requirements.txt`.
- Creates a default `runtime_config.json`.

To run the setup, execute:
```bash
bash setup.sh
```
#### 1.2 Debian VM using Github (Cloud deployment)

To run the setup, copy the following file from the repository and execute in your VM:
```bash
bash setup_git_debian.sh
```

### 2. Data Structure (Required)
The model expects a specific data layout. You must create a `data_local` directory inside the `vis-repo` folder and populate it as follows.

**Annotation File:**
- The annotation file `train.json` must be placed in the root of the `vis-repo` directory (next to `main.py`).

**Image Data:**
- The image frames must be organized into subdirectories within `data_local/trainxs/`. Each subdirectory represents a video sequence.

The final structure should look like this:
```
vis-repo/
├── data_local/
│   └── trainxs/
│       ├── 0001/            # Video Sequence 1
│       │   ├── 00001.jpg
│       │   ├── 00002.jpg
│       │   └── ...
│       ├── 0002/            # Video Sequence 2
│       │   ├── 00001.jpg
│       │   ├── 00002.jpg
│       │   └── ...
│       └── ...
├── main.py
├── train.json             # Annotations file
└── ...
```

### 3. Runtime Configuration
The `setup.sh` script will create a `runtime_config.json` file. This file allows you to enable or disable specific pipelines for a run. You can edit this file to customize which strategies are executed.

### 4. Execution

To run the model, execute:
```bash
bash start.sh
```
Note: The script will automatically kill any previous instances of the model and restart it.

---

# 📚 Detailed Strategy Pipelines
## 1. Baseline (`baseline_base`)
The baseline module establishes the fundamental performance metrics using standard YOLO inference techniques.

**Type:** Single-Stage Global Inference
**Goal:** Establish a "lower bound" for performance using standard YOLO usage.

**Mechanism:**
1.  **Resizing:** The 4K input image ($3840 \times 2160$) is resized down to the model's native input size (e.g., $640 \times 640$).
2.  **Inference:** A single forward pass of the YOLO model is performed.
3.  **Limitations:** Small birds (often < 20 pixels wide) are decimated during resizing, vanishing completely or becoming indistinguishable from noise.

---

## 2. Strategy 1: Native Tiling (`baseline.py` - `strategy_1`)
**Type:** Tiled Inference (Batch Processing)
**Goal:** Solve the small object problem by maintaining high resolution via manual tiling.

**Mechanism:**
1.  **Grid Generation:** The 4K frame is divided into a fixed **4x3 grid (12 tiles)**.
2.  **Overlap:** A 20% overlap is applied between adjacent tiles to ensure objects on seams are not missed.
3.  **Batch Inference:** All 12 tiles are stacked and processed in one batch.
4.  **Coordinate Mapping:** Detections are mapped back to global 4K coordinates.

---

## 3. Strategy 3: Native Tiling + NMS (`baseline.py` - `strategy_3`)
**Type:** Tiled Inference with Global Post-Processing
**Goal:** Remove duplicate detections caused by tile overlaps from Strategy 1.

**Mechanism:**
1.  **Steps 1-4:** Same as *Strategy 1*.
2.  **Global NMS:** A Global Non-Maximum Suppression (NMS) pass is applied to merge duplicate detections from overlap regions.

---

## 4. Strategy 4: SAHI Tiling (`baseline.py` - `strategy_4`)
**Type:** Library-Based Tiled Inference
**Goal:** Establish a "best-in-class" tiling baseline using the SAHI library.

**Mechanism:** This pipeline is activated by running `baseline_base` with `use_sahi=True`.
1.  **SAHI Slicing:** The SAHI library slices the 4K image into overlapping patches (e.g., $640 \times 640$).
2.  **Inference:** YOLO is run on each patch.
3.  **Merging:** SAHI merges the detections and applies NMS.
*   **Impact:** Recall skyrockets (0.1% -> 54.5%) compared to `baseline_base`.
*   **Cost:** Processing time increases significantly (FPS drops from 0.69 to 0.44).

---

## 5. Strategy 5: GMC + Dynamic Thresholding (`strategy_5.py`)
**Type:** Hybrid (Motion Proposal -> Detector Refinement)
**Goal:** Filter out static background to focus compute only on moving objects.

**Pipeline:**
1.  **GMC:** Shi-Tomasi corners and Lucas-Kanade Optical Flow are used to compute a homography that cancels camera motion.
2.  **Motion Detection:** An adaptive threshold is applied to the frame difference to create a motion mask.
3.  **Proposal Generation:** Contours are found and filtered by area and aspect ratio.
4.  **YOLO Refiner:** YOLO is run *only* on cropped Regions of Interest (ROIs) generated from the motion proposals.

---

## 6. Strategy 8: Motion-Guided ROIs (`strategy_8.py`)
**Type:** Motion-Guided Two-Stage Detector
**Goal:** Maximize 4K throughput by treating detection as a "verification" step for motion.

**Pipeline:**
1.  **Motion Proposals:** Uses the stabilized frame differencing (GMC) method to find moving blobs.
2.  **Temporal Scheduling (`detect_every`):** Inference is run only every $N$ frames to save compute.
3.  **Strategic Full Scans:** A full-frame scan is triggered every $M$ frames to catch stationary objects.
4.  **Selective Inference:** YOLO runs on the batch of ROIs from motion proposals.

---

## 7. Strategy 9: SAHI with Temporal Scheduling (`strategy_8.py` - `strategy_9`)
**Type:** Temporally-Sparse Tiled Inference
**Goal:** Achieve a balance of high recall and high speed.

**Pipeline:** This strategy combines the temporal scheduling of Strategy 8 with the powerful tiling of SAHI.
1.  **Temporal Scheduling (`detect_every`):** As in Strategy 8, inference is not run on every frame.
2.  **SAHI Slicing:** On keyframes, the entire frame is processed using the SAHI library.
3.  **Result:** This achieves the **highest F1-Score (8.29%)** in the benchmark by combining the high recall of SAHI with the speed benefits of frame skipping.

---

## 8. Strategy 10: Motion-Gated Native Tiling (`strategy_10.py`)
**Type:** Tiled Hybrid
**Goal:** Combine the precision of Tiling with the efficiency of Motion Gating.

**Pipeline:**
The frame is divided into native tiles. GMC is used to identify tiles with significant motion. Only these "active" tiles are sent to the detector, saving compute on static background regions.

---

## 9. Strategy 11: ROI Classifier Filter (`strategy_11.py`)
**Type:** Three-Stage Cascade (Motion -> Classify -> Detect)
**Goal:** Use a lightweight classifier as a "fail-fast" gate to minimize expensive detection work.

**Pipeline:**
Motion proposals are first sent to a fast YOLO-Classify model. Only if the ROI is classified as "bird-like" is it passed to the full YOLO-Detect model.

---

## 10. Strategy 12: Temporal Interpolation (`strategy_12.py`)
**Type:** Temporal Optimization
**Goal:** Maximize FPS by skipping frames and mathematically interpolating bounding box positions.

**Pipeline:**
Full inference is run only on keyframes (e.g., every 5th frame). For intermediate frames, the positions of detected objects are linearly interpolated between keyframes.
*   **`strategy_12a`**: The keyframe detection uses **GMC + YOLO Refiner** (as in Strategy 5). This is the fastest pipeline at **3.54 FPS**.
*   **`strategy_12b`**: The keyframe detection uses **SAHI**. This is an excellent trade-off, yielding high recall (43.89%) at a high framerate (2.25 FPS).

---

## 11. Strategy 13: The "Kitchen Sink" Hybrid Funnel (`strategy_13.py`)
**Type:** Multi-Gate Hybrid
**Goal:** A funnel that catches both moving and stationary objects efficiently by combining tiling, motion-gating, and classification.

**Pipeline:**
For each native tile, the pipeline follows a decision tree:
1.  Is there motion? -> **Run Detector.**
2.  No motion? -> **Run lightweight Classifier.**
3.  Classifier finds an object? -> **Run Detector.**
4.  Classifier finds nothing? -> **Discard tile.**

This logic is implemented in two variants:
*   **`strategy_13a`**: The standard funnel described above.
*   **`strategy_13b`**: Adds **Temporal Interpolation** (from Strategy 12) for increased throughput.

