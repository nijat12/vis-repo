# Refactoring and Feature Development Plan

This document outlines the major refactoring tasks and new feature implementations for the VIS project.

## Phase 1: Core Architectural Refactoring

**Objective:** Modify the pipeline execution logic to be driven by dynamic configurations, allowing each pipeline to run across multiple confidence thresholds and execution modes (e.g., SAHI vs. legacy).

- [X] **1.1. Update `config.py`:**
  - [X] Introduce a global `CONF_THRESHOLDS: List[float] = [0.01, 0.1, 0.2, 0.5]` list.
  - [X] Remove the static `conf_thresh` key from all individual pipeline configuration dictionaries (e.g., `BASELINE_CONFIG`, `STRATEGY_8_CONFIG`).
  - [X] Add a new `SAHI_CONFIG` dictionary containing parameters for SAHI-based slicing (`slice_size`, `overlap_ratio`, etc.).
  - [X] Add new config dictionaries for `STRATEGY_12_CONFIG` and `STRATEGY_13_CONFIG`.

- [X] **1.2. Refactor `main.py` for Dynamic Execution:**
  - [X] Create a `generate_run_configurations` function. This generator will be the new heart of the execution loop. It will:
    - Iterate through pipelines enabled in `runtime_config.json` (allowing different VMs to run different sets of pipelines).
    - For each pipeline, create nested loops for each value in `Config.CONF_THRESHOLDS`.
    - For each pipeline, generate variants for `use_sahi=True` and `use_sahi=False`.
    - For `strategy_13`, generate variants for `use_interpolation=True` and `use_interpolation=False`.
    - Yield a unique configuration dictionary for each permutation (e.g., `{ 'run_name': 'strategy_8_sahi_01', 'base_pipeline': 'strategy_8', 'conf_thresh': 0.1, 'use_sahi': True }`).
  - [X] Modify the `main()` function to call this generator and pass the resulting list of configuration dictionaries to the `multiprocessing.Pool`.
  - [X] Refactor `run_single_pipeline` to accept a single `run_config: Dict` argument instead of `pipeline_name: str`.
  - [X] Inside `run_single_pipeline`, dynamically build the final config for the pipeline by merging the base config from `Config.get_pipeline_config(run_config['base_pipeline'])` with the dynamic parameters from `run_config`.

- [X] **1.3. Standardize Pipeline Signatures:**
  - [X] In `pipelines/__init__.py`, and all files in `pipelines/`, modify the signature of all registered pipeline functions (e.g., `run_strategy_8_pipeline`) from `()` to `(config: Dict)`.
  - [X] Update the implementation of each pipeline to retrieve all parameters from the `config` dictionary passed as an argument, rather than fetching them directly from the `Config` class.

- [X] **1.4. Refactor `baseline.py` to Expose Variants:**
  - [X] Remove the `run_all_baseline_variants` orchestrator function.
  - [X] Register each baseline variant (`baseline_base`, `baseline_w_tiling`, `baseline_w_tiling_and_nms`) as a separate pipeline using the `@register_pipeline` decorator.
  - [X] This allows the main `generate_run_configurations` function to treat them as independent pipelines that can be selected in `runtime_config.json` and create all required permutations (conf_thresh, sahi) for each one.

## Phase 2: SAHI Integration

**Objective:** Integrate the `sahi` library as an alternative to the manual tiling and NMS logic, controlled by a configuration flag.

- [X] **2.1. Add SAHI dependency:**
  - [X] Verify that `sahi` is in `requirements.txt`. (Verified).

- [X] **2.2. Abstract Tiling Logic in Each Pipeline:**
  - [X] For each pipeline (`baseline.py`, `strategy_2.py`, etc.), refactor the prediction logic.
  - [X] Implement a conditional check: `if config.get('use_sahi', False):`.
  - [X] In the `if` block, add a call to a new SAHI-based prediction function (e.g., `get_sahi_predictions`) that uses `sahi.get_sliced_prediction`. This function can be a local helper or a shared utility in `vis_utils.py`.
  - [X] Keep the existing manual tiling/ROI code inside the `else` block to maintain the legacy path.
  - [X] This affects: `baseline.py`, `strategy_2.py`, `strategy_8.py`, `strategy_10.py`, `strategy_11.py`.

## Phase 3: New Pipeline Development

- [X] **3.1. Build Strategy 12 (GMC + Interpolation):**
  - [X] **3.1.1. Create `pipelines/strategy_12.py`:**
    - [X] Initialize with the structure of `strategy_2.py` to reuse its GMC logic.
    - [X] Integrate the dual-path SAHI/legacy tiling logic from Phase 2.
  - [X] **3.1.2. Implement Interpolation Logic:**
    - [X] Use the `detect_every` parameter from the `config` dictionary to run detection only on keyframes.
    - [X] Use `vis_utils.ObjectTracker` to maintain object identities between keyframes. Store the results from the last processed keyframe.
    - [X] When a new keyframe is processed, match objects between the old and new sets.
    - [X] For each object ID present in both keyframes, perform a linear interpolation of the bounding box coordinates `(x, y, w, h)` for all the skipped frames in between.
    - [X] Add the newly generated interpolated boxes to the results for the intermediate frames.

- [X] **3.2. Build Strategy 13 (Motion-Gated Classifier Funnel):**
  - [X] **3.2.1. Create `pipelines/strategy_13.py`:**
    - [X] Implement the full pipeline logic on a tile-by-tile basis, integrating the dual-path SAHI/legacy tiling.
  - [X] **3.2.2. Implement Gating Funnel Logic:**
    - [X] For each tile:
        1. Apply motion detection from `strategy_10`.
        2. If motion is present, proceed to the main YOLO detector.
        3. If no motion, pass the tile to the classifier model from `strategy_11`.
        4. If the classifier is positive, proceed to the main YOLO detector.
        5. If both are negative, the tile is skipped entirely.
  - [X] **3.2.3. Create Interpolation Variant:**
    - [X] The `generate_run_configurations` function in `main.py` will create two runs for this strategy: one with `use_interpolation=False` and one with `use_interpolation=True`.
    - [X] Inside `strategy_13.py`, add a conditional check for `config.get('use_interpolation', False)` to enable the `detect_every` and box interpolation logic from Strategy 12.

## Phase 4: Documentation

- [X] **4.1. Update `README.md` for Strategy 2:**
  - [X] Add a new, detailed section for `Strategy 2`.
  - [X] Explain the concept of using background subtraction (GMC) to stabilize the camera view and isolate true object motion.
  - [X] Detail how contours from the motion mask are extracted and filtered to create Regions of Interest (ROIs).
  - [X] Emphasize that running the expensive YOLO detector only on these small ROIs is the key efficiency gain over a full-frame scan.

## Phase 5: Advanced Metrics Implementation

**Objective:** Integrate a more comprehensive set of evaluation metrics as required for the assignment, including mAP, HOTA, and others.

- [x] **5.1. Implement AP/mAP Calculation:**
  - [x] Modify pipeline evaluation logic to store detection confidence scores alongside bounding boxes for each frame.
  - [x] Create a new function in `vis_utils.py` (e.g., `calculate_video_map`) that takes all detections (with scores) and ground truths for a single video.
  - [x] Inside this function, implement the standard COCO-style mAP calculation:
    - Sort detections by confidence.
    - Calculate the precision-recall curve by iterating through sorted detections.
    - Compute the area under the P-R curve to get the Average Precision (AP) for the "bird" class.
  - [x] In each pipeline, at the end of each video, call this function and log the resulting "mAP per video". (Done for Baseline, Strategy 13, others pending)

- [x] **5.2. Log Average DotD (Dot Distance):**
  - [x] In the evaluation loop within each pipeline, when a True Positive is identified, store the calculated center distance (`best_dist`).
  - [x] At the end of each video and at the end of the entire run, calculate the average of these stored distances.
  - [x] Add "Average DotD" to the video-level and summary-level logs.

- [x] **5.3. Add Missing Time/Memory Metrics:**
  - [x] In the final summary block of each pipeline, add "Average Time per Video" (`total_time / num_videos`).
  - [x] In the final summary block, add "Average Time per Image" (`total_time / total_frames`).

- [ ] **5.4. Propagate Metrics to Other Strategies:**
  - [ ] Update `strategy_2.py`: Return scores, calculate mAP/DotD.
  - [ ] Update `strategy_7.py`: Return scores, calculate mAP/DotD.
  - [ ] Update `strategy_8.py`: Return scores, calculate mAP/DotD.
  - [ ] Update `strategy_9.py`: Return scores, calculate mAP/DotD.
  - [ ] Update `strategy_10.py`: Return scores, calculate mAP/DotD.
  - [ ] Update `strategy_11.py`: Return scores, calculate mAP/DotD.
  - [ ] Update `strategy_12.py`: Return scores, calculate mAP/DotD.
