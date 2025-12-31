"""
CSV Results Utility for VIS Pipeline

Handles:
- Real-time results saving to prevent data loss
- Multi-sheet Excel format with transposed summary
- Per-image detailed tracking for each pipeline
- Automatic GCS upload with timestamped filenames
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import subprocess

from config import Config

logger = logging.getLogger(__name__)


class ResultsTracker:
    """
    Manages real-time results tracking and saving.

    Features:
    - Summary CSV (transposed: pipelines as columns, metrics as rows)
    - Detailed CSVs per pipeline
    - Automatic GCS upload with timestamp
    - Incremental saving to prevent data loss
    """

    def __init__(self, start_time: Optional[datetime] = None):
        """
        Initialize the results tracker.

        Args:
            start_time: Timestamp for the run (defaults to now)
        """
        self.start_time = start_time or datetime.now()

        # Generate timestamp: YYMMDD_HHMMSS
        self.timestamp = self.start_time.strftime("%y%m%d_%H%M%S")

        # Base names for tracking
        self.base_name = f"results_{self.timestamp}"

        # Data storage
        self.summary_data: Dict[str, Dict[str, Any]] = {}
        self.detailed_data: Dict[str, List[Dict[str, Any]]] = {}

        # Tracking saved files for GCS upload
        self.saved_files: List[str] = []
        self.pipeline_configs = {}  # Store runtime configs

        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

        logger.info(f"📊 Results tracker initialized: {self.base_name}")

    def update_summary(
        self,
        pipeline_name: str,
        metrics: Dict[str, Any],
        config: Dict[str, Any] = None,
    ):
        """Update summary metrics for a pipeline."""
        if pipeline_name not in self.summary_data:
            self.summary_data[pipeline_name] = {}
        self.summary_data[pipeline_name].update(metrics)

        if config:
            self.pipeline_configs[pipeline_name] = config

        self._save_local()

    def add_image_result(self, pipeline_name: str, image_data: Dict[str, Any]):
        """
        Add per-image detailed result for a pipeline.
        """
        if pipeline_name not in self.detailed_data:
            self.detailed_data[pipeline_name] = []

        self.detailed_data[pipeline_name].append(image_data)

    def save_batch(self, pipeline_name: str, batch_size: int = 100):
        """
        Save results after processing a batch of images.
        """
        if pipeline_name in self.detailed_data:
            num_images = len(self.detailed_data[pipeline_name])
            if num_images % batch_size == 0:
                logger.info(
                    f"💾 Saving batch for {pipeline_name} ({num_images} images processed)"
                )
                self._save_local()

    def _save_local(self):
        """
        Save results to local CSV files with enhanced structure.
        - Detailed CSVs include config headers.
        - Summary CSV includes overall and per-video tables.
        """
        if not self.detailed_data:
            logger.debug("No detailed data to save yet.")
            return

        try:
            self.saved_files = []

            # 1. Detailed CSVs per pipeline (with config headers)
            for pipeline_name, data in self.detailed_data.items():
                if data:
                    config_data = Config.get_pipeline_config(pipeline_name)
                    det_path = os.path.join(
                        Config.OUTPUT_DIR, f"{self.base_name}_{pipeline_name}.csv"
                    )

                    with open(det_path, "w") as f:
                        f.write(f"# Pipeline: {pipeline_name}\n")
                        for key, value in config_data.items():
                            if key != "log_queue":
                                f.write(f"# {key}: {value}\n")

                    detailed_df = pd.DataFrame(data)
                    detailed_df.to_csv(det_path, index=False, mode="a")
                    self.saved_files.append(det_path)

            # 2. Combined Summary CSV
            if self.summary_data:
                summary_path = os.path.join(
                    Config.OUTPUT_DIR, f"{self.base_name}_summary.csv"
                )

                with open(summary_path, "w") as f:
                    # Consolidated Configuration Table
                    f.write("PIPELINE CONFIGURATIONS\n")
                    # Use stored configs, fallback to empty dict if missing
                    all_configs = {
                        p: {
                            k: v
                            for k, v in self.pipeline_configs.get(p, {}).items()
                            if k != "log_queue"
                        }
                        for p in self.summary_data.keys()
                        if p != "log_queue"
                    }
                    config_df = pd.DataFrame(all_configs).fillna("")
                    config_df.to_csv(f)
                    f.write("\n\n")

                    # Overall Summary Table (Metrics Only)
                    f.write("OVERALL SUMMARY\n")

                    # Supplement summary_data with aggregated metrics from detailed_data
                    enhanced_summary = {}
                    for pipeline_name, metrics in self.summary_data.items():
                        enhanced_metrics = metrics.copy()
                        p_data = self.detailed_data.get(pipeline_name, [])
                        if p_data:
                            # Use pandas for easy aggregation
                            df = pd.DataFrame(p_data)
                            if "iou" in df.columns:
                                enhanced_metrics["iou"] = df["iou"].mean()
                            if "mAP" in df.columns:
                                enhanced_metrics["mAP"] = df["mAP"].mean()
                            elif "AP" in df.columns:
                                enhanced_metrics["AP"] = df["AP"].mean()
                            if "processing_time_sec" in df.columns:
                                enhanced_metrics["avg_processing_time_sec"] = df[
                                    "processing_time_sec"
                                ].mean()
                                enhanced_metrics["total_processing_time_sec"] = df[
                                    "processing_time_sec"
                                ].sum()
                            if "memory_usage_mb" in df.columns:
                                enhanced_metrics["avg_memory_usage_mb"] = df[
                                    "memory_usage_mb"
                                ].mean()
                        enhanced_summary[pipeline_name] = enhanced_metrics

                    overall_metrics_df = pd.DataFrame(enhanced_summary)
                    overall_metrics_df.to_csv(f)
                    f.write("\n\n")

                    # Per-Video Summaries (Metrics Only)
                    all_video_names = sorted(
                        list(
                            set(
                                img["video"]
                                for _, p_data in self.detailed_data.items()
                                for img in p_data
                            )
                        )
                    )

                    for video_name in all_video_names:
                        f.write(f"VIDEO SUMMARY: {video_name}\n")
                        video_summary_data = {}

                        for pipeline_name in self.summary_data.keys():
                            video_frames = [
                                d
                                for d in self.detailed_data.get(pipeline_name, [])
                                if d["video"] == video_name
                            ]
                            if not video_frames:
                                continue

                            vid_df = pd.DataFrame(video_frames)

                            tp = vid_df["tp"].sum()
                            fp = vid_df["fp"].sum()
                            fn = vid_df["fn"].sum()

                            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            f1_score = (
                                2 * (precision * recall) / (precision + recall)
                                if (precision + recall) > 0
                                else 0.0
                            )

                            video_time = vid_df["processing_time_sec"].sum()
                            video_fps = (
                                len(vid_df) / video_time if video_time > 0 else 0.0
                            )

                            video_summary_data[pipeline_name] = {
                                "total_frames": len(vid_df),
                                "avg_fps": video_fps,
                                "precision": precision,
                                "recall": recall,
                                "f1_score": f1_score,
                                "tp": tp,
                                "fp": fp,
                                "fn": fn,
                                "iou": vid_df["iou"].mean(),
                                "avg_processing_time_sec": vid_df[
                                    "processing_time_sec"
                                ].mean(),
                                "total_processing_time_sec": video_time,
                                "avg_memory_usage_mb": vid_df["memory_usage_mb"].mean(),
                            }

                        if video_summary_data:
                            video_df = pd.DataFrame(video_summary_data)
                            video_df.to_csv(f, mode="a")

                        f.write("\n\n")

                self.saved_files.append(summary_path)
                logger.debug(
                    f"✅ Saved local CSV results ({len(self.saved_files)} files)"
                )

        except Exception as e:
            logger.error(f"❌ Failed to save local results: {e}", exc_info=True)

    def upload_to_gcs(self):
        """Upload all generated CSV files to GCS bucket."""
        if not self.saved_files:
            logger.warning("⚠️  No files to upload")
            return False

        success = True
        try:
            from google.cloud import storage

            # Use service account key if it exists, otherwise use default credentials
            if Config.SERVICE_ACCOUNT_KEY and os.path.exists(
                Config.SERVICE_ACCOUNT_KEY
            ):
                logger.info(f"Using service account key: {Config.SERVICE_ACCOUNT_KEY}")
                client = storage.Client.from_service_account_json(
                    Config.SERVICE_ACCOUNT_KEY
                )
            else:
                client = storage.Client()

            bucket = client.bucket(Config.BUCKET_NAME)

            for local_path in self.saved_files:
                if not os.path.exists(local_path):
                    continue

                filename = os.path.basename(local_path)
                blob_name = f"results/{filename}"
                blob = bucket.blob(blob_name)

                logger.info(f"☁️  Uploading {filename} to GCS...")
                blob.upload_from_filename(local_path)

            logger.info("✅ All results uploaded to GCS")
            return True

        except Exception as e:
            logger.error(f"❌ GCS upload error: {e}")
            return False

    def finalize(self):
        """
        Finalize results tracking.
        """
        logger.info("🏁 Finalizing results...")

        # Save final local copy
        self._save_local()

        # Upload to GCS
        self.upload_to_gcs()

        # Generate summary report
        logger.info("RESULTS SUMMARY")
        logger.info(f"Files saved to: {Config.OUTPUT_DIR}")
        logger.info(f"Total files: {len(self.saved_files)}")

        if self.summary_data:
            logger.info("Pipeline Metrics:")
            for pipeline_name, metrics in self.summary_data.items():
                logger.info(f"{pipeline_name.upper()}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")

        # Return path to summary file as primary return value
        summary_path = os.path.join(Config.OUTPUT_DIR, f"{self.base_name}_summary.csv")
        gcs_summary_path = (
            f"gs://{Config.BUCKET_NAME}/results/{self.base_name}_summary.csv"
        )
        return summary_path, gcs_summary_path


def create_image_result(
    video_name: str,
    frame_name: str,
    image_path: str,
    predictions: List,
    ground_truths: List,
    tp: int,
    fp: int,
    fn: int,
    processing_time_sec: float = 0.0,
    iou: float = 0.0,
    memory_usage_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Create a standardized image result dictionary.

    Args:
        video_name: Name of the video
        frame_name: Name of the frame/image file
        image_path: Full path to the image
        predictions: List of prediction boxes
        ground_truths: List of ground truth boxes
        tp: True positives count
        fp: False positives count
        fn: False negatives count
        processing_time_sec: Time in seconds to process this image
        iou: Intersection over Union for the image
        memory_usage_mb: Memory usage in megabytes

    Returns:
        Dictionary with standardized image-level metrics
    """
    return {
        "video": video_name,
        "frame": frame_name,
        "image_path": image_path,
        "num_predictions": len(predictions),
        "num_ground_truths": len(ground_truths),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "processing_time_sec": round(processing_time_sec, 4),
        "iou": round(iou, 4),
        "memory_usage_mb": round(memory_usage_mb, 2),
    }


# Global results tracker instance
_results_tracker: Optional[ResultsTracker] = None


def get_results_tracker() -> ResultsTracker:
    """Get or create the global results tracker instance."""
    global _results_tracker
    if _results_tracker is None:
        _results_tracker = ResultsTracker()
    return _results_tracker


def reset_results_tracker():
    """Reset the global results tracker (for testing)."""
    global _results_tracker
    _results_tracker = None
