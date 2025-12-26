"""
Configuration for VIS Pipeline Execution on Google Cloud VM

This module provides centralized configuration for:
- Pipeline selection (Baseline, Strategy 7, Strategy 8)
- GCS authentication and data paths
- Processing parameters
- VM killswitch functionality
"""

import os
from typing import List, Dict, Any


class Config:
    """Central configuration for VIS pipeline execution."""
    
    # ==========================================
    # PIPELINE SELECTION
    # ==========================================
    # Enable/disable pipelines independently
    # Set to True to run, False to skip
    ENABLED_PIPELINES: List[str] = [
        "baseline",      # YOLOv5n with 4x3 tiled inference
        "strategy_7",  # Motion compensation + CNN verifier
        "strategy_8",  # YOLO on ROIs
    ]
    
    # ==========================================
    # GCS CONFIGURATION
    # ==========================================
    # Service account key file path (relative to project root)
    SERVICE_ACCOUNT_KEY: str = "colab-upload-bot-key.json"
    
    # GCS bucket and paths
    BUCKET_NAME: str = "vis-data-2025"
    GCS_TRAIN_DIR: str = f"gs://{BUCKET_NAME}/trainxs"
    GCS_JSON_URL: str = f"gs://{BUCKET_NAME}/train.json"
    
    # Local data paths
    LOCAL_BASE_DIR: str = "./data_local"
    LOCAL_TRAIN_DIR: str = os.path.join(LOCAL_BASE_DIR, "trainxs")
    LOCAL_JSON_PATH: str = "./train.json"
    
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
    
    # ==========================================
    # BASELINE PIPELINE CONFIG
    # ==========================================
    BASELINE_CONFIG: Dict[str, Any] = {
        "model_name": "yolov5n",
        "img_size": 1280,
        "conf_thresh": 0.01,
        "iou_thresh": 0.45,
        "model_classes": [14],  # Bird class only
        "output_csv": "baseline_tiled_cpu.csv",
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
        "model_name": "yolov5n",
        "img_size": 1280,
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
    # PARALLEL EXECUTION
    # ==========================================
    # Number of CPU workers for parallel pipeline execution
    # Set to 1 for sequential execution
    # Set to len(ENABLED_PIPELINES) to run all pipelines in parallel
    MAX_WORKERS: int = len(ENABLED_PIPELINES)
    
    # ==========================================
    # VM KILLSWITCH
    # ==========================================
    # Automatically shutdown VM after pipeline completion
    ENABLE_KILLSWITCH: bool = False
    
    # Delay before shutdown (seconds) - gives time to review logs
    KILLSWITCH_DELAY_SECONDS: int = 60
    
    # ==========================================
    # LOGGING
    # ==========================================
    # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL: str = "INFO"
    
    # Enable Google Cloud Logging (for VM execution)
    ENABLE_CLOUD_LOGGING: bool = True
    
    # Local log file
    LOG_FILE: str = "run.log"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration and raise errors if invalid."""
        if not cls.ENABLED_PIPELINES:
            raise ValueError("At least one pipeline must be enabled in ENABLED_PIPELINES")
        
        valid_pipelines = {"baseline", "strategy_7", "strategy_8"}
        for pipeline in cls.ENABLED_PIPELINES:
            if pipeline not in valid_pipelines:
                raise ValueError(f"Invalid pipeline: {pipeline}. Must be one of {valid_pipelines}")
        
        if cls.SHOULD_LIMIT_VIDEO == 1 and not cls.VIDEO_INDEXES:
            raise ValueError("VIDEO_INDEXES must be specified when SHOULD_LIMIT_VIDEO == 1")
        
        if cls.MAX_WORKERS < 1:
            raise ValueError("MAX_WORKERS must be at least 1")
        
        # Check if service account key exists (if specified)
        if cls.SERVICE_ACCOUNT_KEY and not os.path.exists(cls.SERVICE_ACCOUNT_KEY):
            print(f"⚠️  Warning: Service account key not found at {cls.SERVICE_ACCOUNT_KEY}")
            print("    Will attempt to use VM default credentials")
    
    @classmethod
    def get_output_path(cls, pipeline_name: str) -> str:
        """Get the full output path for a pipeline's CSV file."""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        
        if pipeline_name == "baseline":
            filename = cls.BASELINE_CONFIG["output_csv"]
        elif pipeline_name == "strategy_7":
            filename = cls.STRATEGY_7_CONFIG["output_csv"]
        elif pipeline_name == "strategy_8":
            filename = cls.STRATEGY_8_CONFIG["output_csv"]
        else:
            filename = f"{pipeline_name}_output.csv"
        
        return os.path.join(cls.OUTPUT_DIR, filename)
