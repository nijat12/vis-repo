"""
Configuration for VIS Pipeline Execution on Google Cloud VM

This module provides centralized configuration for:
- Pipeline selection (Baseline, Strategy 7, Strategy 8)
- GCS authentication and data paths
- Processing parameters
- VM killswitch functionality
"""

import os
import json
from typing import List, Dict, Any


class Config:
    """Central configuration for VIS pipeline execution."""
    
    # ==========================================
    # PIPELINE SELECTION
    # ==========================================
    # Enable/disable pipelines independently
    # Set to True to run, False to skip
    ENABLED_PIPELINES: List[str] = [
        "baseline",      # YOLO with 4x3 tiled inference
        "strategy_2",    # GMC + Dynamic Thresholding + YOLO Refiner
        "strategy_7",  # Motion compensation + CNN verifier
        "strategy_8",  # YOLO on ROIs
        "strategy_9",  # SAHI Slicing + YOLO + Kalman/Hungarian (DotD)
        "strategy_10", # Motion Proposals + YOLO Classification
    ]
    
    # ==========================================
    # GCS CONFIGURATION
    # ==========================================
    # Service account key file path (relative to project root)
    SERVICE_ACCOUNT_KEY: str = "colab-upload-bot-key.json"
    
    # GCS bucket and paths
    BUCKET_NAME: str = "vis-data-2025"
    GCS_TRAIN_ZIP: str = "trainxs.zip"
    GCS_JSON_URL: str = "train.json"
    
    # Project Root
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    
    # Local data paths
    LOCAL_BASE_DIR: str = os.path.join(PROJECT_ROOT, "data_local")
    LOCAL_TRAIN_DIR: str = os.path.join(LOCAL_BASE_DIR, "trainxs")
    LOCAL_JSON_PATH: str = os.path.join(PROJECT_ROOT, "train.json")
    LOCAL_ZIP_PATH: str = os.path.join(LOCAL_BASE_DIR, "trainxs.zip")
    
    # ==========================================
    # PROCESSING PARAMETERS
    # ==========================================
    # Video selection
    # Set to 0 to process all videos
    # Set to N to process first N videos
    # Set to 1 and use VIDEO_INDEXES to select specific videos
    SHOULD_LIMIT_VIDEO: int = 1
    VIDEO_INDEXES: List[int] = [1]  # Only used if SHOULD_LIMIT_VIDEO == 1
    
    # GPU/CPU settings
    IS_GPU_ALLOWED: bool = False  # Set to True if GPU is available
    
    # Output directory for results
    OUTPUT_DIR: str = "./metrics"
    
    UNIFIED_MODEL_NAME: str = "yolo12l.pt"
    
    # ==========================================
    # BASELINE PIPELINE CONFIG
    # ==========================================
    BASELINE_CONFIG: Dict[str, Any] = {
        "model_name": UNIFIED_MODEL_NAME,
        "img_size": 640,
        "conf_thresh": 0.01,
        "iou_thresh": 0.45,
        "model_classes": [14],  # Bird class only
        "output_csv": "baseline_tiled_cpu.csv",
    }
    
    # ==========================================
    # STRATEGY 2 CONFIG (GMC + Dynamic Threshold + YOLO)
    # ==========================================
    STRATEGY_2_CONFIG: Dict[str, Any] = {
        "model_name": UNIFIED_MODEL_NAME,
        "img_size": 640,
        "conf_thresh": 0.01,
        "model_classes": [14],
        "roi_scale": 3.0,
        "min_roi_size": 192,
        "max_rois": 5,
        "dynamic_multiplier": 4.0,  # Multiplier for StdDev in thresholding
        "min_threshold": 20,       # Min clamp for dynamic threshold
        "max_threshold": 80,       # Max clamp for dynamic threshold
        "min_hits": 3,             # For persistence tracking
        "output_csv": "strat_2_cpu.csv",
    }
    
    # ==========================================
    # STRATEGY 7 CONFIG
    # ==========================================
    STRATEGY_7_CONFIG: Dict[str, Any] = {
        "use_verifier": True,
        "crop_scale": 4.0,
        "bird_threshold": 0.01,
        "min_keep": 2,
        "max_keep": 5,
        "output_csv": "strat_7_cpu.csv",
    }
    
    # ==========================================
    # STRATEGY 8 CONFIG
    # ==========================================
    STRATEGY_8_CONFIG: Dict[str, Any] = {
        "model_name": UNIFIED_MODEL_NAME,
        "img_size": 640,
        "conf_thresh": 0.01,
        "iou_thresh": 0.45,
        "model_classes": [14],
        "roi_scale": 3.0,
        "min_roi_size": 192,
        "max_rois": 3,
        "fullframe_every": 0,  # 0 disables full-frame processing
        "detect_every": 3,  # Run YOLO every N frames
        "output_csv": "strat_8_cpu.csv",
    }
    
    # ==========================================
    # STRATEGY 9 CONFIG (SAHI + YOLO + Kalman)
    # ==========================================
    STRATEGY_9_CONFIG: Dict[str, Any] = {
        "model_path": UNIFIED_MODEL_NAME,  # Using YOLO
        "conf_thresh": 0.2,          # Confidence threshold (from diagram)
        "slice_size": 640,           # Slice dimensions (from diagram)
        "img_size": 640,             # Inference size
        "overlap": 0.2,              # Overlap ratio (from diagram)
        "bird_class_id": 14,         # YOLO Bird class ID
        "tracker_dist_thresh": 50,   # Pixel distance for DotD association
        "max_age": 15,               # Max frames to keep lost tracks
        "min_hits": 2,               # Min hits to confirm track
        "output_csv": "strat_9_cpu.csv",
    }
    
    # ==========================================
    # STRATEGY 10 CONFIG (Motion + YOLO)
    # ==========================================
    STRATEGY_10_CONFIG: Dict[str, Any] = {
        "model_name": UNIFIED_MODEL_NAME,
        "img_size": 640,
        "conf_thresh": 0.1,          # Confidence for YOLO classification
        "motion_thresh_scale": 4.0,  # multiplier for motion threshold
        "bird_class_id": 14,
        "output_csv": "strat_10_cpu.csv",
    }
    
    # ==========================================
    # PARALLEL EXECUTION
    # ==========================================
    # Number of CPU workers for parallel pipeline execution
    # Set to 1 for sequential execution
    # Set to len(ENABLED_PIPELINES) to run all pipelines in parallel
    MAX_WORKERS: int = 1
    
    # ==========================================
    # VM KILLSWITCH
    # ==========================================
    # Automatically shutdown VM after pipeline completion
    ENABLE_KILLSWITCH: bool = True
    
    # Delay before shutdown (seconds) - gives time to review logs
    KILLSWITCH_DELAY_SECONDS: int = 60
    
    # ==========================================
    # LOGGING
    # ==========================================
    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL: str = "INFO"
    
    # Enable Google Cloud Logging (for VM execution)
    ENABLE_CLOUD_LOGGING: bool = True
    
    # Logging directory
    LOG_DIR: str = "logs"
    
    # Local log file
    LOG_FILE: str = "main.log"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration and raise errors if invalid."""
        if not cls.ENABLED_PIPELINES:
            raise ValueError("At least one pipeline must be enabled in ENABLED_PIPELINES")
        
        if cls.SHOULD_LIMIT_VIDEO == 1 and not cls.VIDEO_INDEXES:
            raise ValueError("VIDEO_INDEXES must be specified when SHOULD_LIMIT_VIDEO == 1")
        
        if cls.MAX_WORKERS < 1:
            raise ValueError("MAX_WORKERS must be at least 1")
        
        # Check if service account key exists (if specified)
        if cls.SERVICE_ACCOUNT_KEY and not os.path.exists(cls.SERVICE_ACCOUNT_KEY):
            print(f"⚠️  Warning: Service account key not found at {cls.SERVICE_ACCOUNT_KEY}")
            print("    Will attempt to use VM default credentials")
    
    @classmethod
    def get_runtime_killswitch(cls) -> bool:
        """
        Check if killswitch is enabled, checking runtime_config.json first.
        Allows enabling/disabling at runtime without restarting the process.
        """
        runtime_config_path = os.path.join(cls.PROJECT_ROOT, "runtime_config.json")
        if os.path.exists(runtime_config_path):
            try:
                with open(runtime_config_path, 'r') as f:
                    data = json.load(f)
                    return data.get("ENABLE_KILLSWITCH", cls.ENABLE_KILLSWITCH)
            except Exception as e:
                print(f"⚠️  Error reading runtime_config.json: {e}")
        
        return cls.ENABLE_KILLSWITCH

    @classmethod
    def get_output_path(cls, pipeline_name: str) -> str:
        """Get the full output path for a pipeline's CSV file."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        if pipeline_name == "baseline":
            filename = cls.BASELINE_CONFIG["output_csv"]
        elif pipeline_name == "strategy_2":
            filename = cls.STRATEGY_2_CONFIG["output_csv"]
        elif pipeline_name == "strategy_7":
            filename = cls.STRATEGY_7_CONFIG["output_csv"]
        elif pipeline_name == "strategy_8":
            filename = cls.STRATEGY_8_CONFIG["output_csv"]
        elif pipeline_name == "strategy_9":
            filename = cls.STRATEGY_9_CONFIG["output_csv"]
        elif pipeline_name == "strategy_10":
            filename = cls.STRATEGY_10_CONFIG["output_csv"]
        else:
            filename = f"{pipeline_name}_output.csv"
        
        return os.path.join(cls.OUTPUT_DIR, filename)
