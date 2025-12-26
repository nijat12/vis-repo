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
    - Transposed summary sheet (pipelines as columns, metrics as rows)
    - Transposed detail sheets (frames as columns, metrics as rows)
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
        
        # Local file paths
        self.local_filename = f"results_{self.timestamp}.xlsx"
        self.local_path = os.path.join(Config.OUTPUT_DIR, self.local_filename)
        
        # GCS path
        self.gcs_path = f"gs://{Config.BUCKET_NAME}/results/{self.local_filename}"
        
        # Data storage
        self.summary_data: Dict[str, Dict[str, Any]] = {}
        self.detailed_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # Ensure output directory exists
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        
        logger.info(f"📊 Results tracker initialized: {self.local_filename}")
    
    def update_summary(self, pipeline_name: str, metrics: Dict[str, Any]):
        """
        Update summary metrics for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline (e.g., 'baseline', 'strategy_7')
            metrics: Dictionary of metrics (precision, recall, f1, etc.)
        """
        self.summary_data[pipeline_name] = metrics
        logger.debug(f"Updated summary for {pipeline_name}")
        self._save_local()
    
    def add_image_result(self, pipeline_name: str, image_data: Dict[str, Any]):
        """
        Add per-image detailed result for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            image_data: Dictionary with image-level metrics
                Expected keys: video, frame, image_path, tp, fp, fn, predictions, ground_truth
        """
        if pipeline_name not in self.detailed_data:
            self.detailed_data[pipeline_name] = []
        
        self.detailed_data[pipeline_name].append(image_data)
    
    def save_batch(self, pipeline_name: str, batch_size: int = 100):
        """
        Save results after processing a batch of images.
        
        Args:
            pipeline_name: Name of the pipeline
            batch_size: Number of images per batch (for logging)
        """
        if pipeline_name in self.detailed_data:
            num_images = len(self.detailed_data[pipeline_name])
            if num_images % batch_size == 0:
                logger.info(f"💾 Saving batch for {pipeline_name} ({num_images} images processed)")
                self._save_local()
    
    def _save_local(self):
        """
        Save results to local Excel file with multiple sheets.
        All sheets are TRANSPOSED: data extends per column, labels per row.
        """
        try:
            with pd.ExcelWriter(self.local_path, engine='openpyxl') as writer:
                # Sheet 1: Transposed Summary (pipelines as columns, metrics as rows)
                if self.summary_data:
                    summary_df = pd.DataFrame(self.summary_data)
                    # Already transposed: pipelines are columns, metrics are rows
                    summary_df.to_excel(writer, sheet_name='Summary')
                
                # Sheets 2+: Transposed detailed per-image results
                # Frames as columns, metrics as rows
                for pipeline_name, data in self.detailed_data.items():
                    if data:
                        detailed_df = pd.DataFrame(data)
                        
                        # Create column names combining video and frame
                        if 'video' in detailed_df.columns and 'frame' in detailed_df.columns:
                            detailed_df['video_frame'] = detailed_df['video'] + '_' + detailed_df['frame']
                            
                            # Set video_frame as index
                            detailed_df = detailed_df.set_index('video_frame')
                            
                            # Drop redundant columns
                            cols_to_drop = ['video', 'frame', 'image_path']
                            detailed_df = detailed_df.drop(columns=[c for c in cols_to_drop if c in detailed_df.columns])
                            
                            # Transpose: metrics become rows, frames become columns
                            transposed_df = detailed_df.T
                        else:
                            # Fallback: standard transpose
                            transposed_df = detailed_df.T
                        
                        # Truncate sheet name to 31 chars (Excel limit)
                        sheet_name = f"{pipeline_name}_details"[:31]
                        transposed_df.to_excel(writer, sheet_name=sheet_name)
            
            logger.debug(f"✅ Saved local results to {self.local_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save local results: {e}")
    
    def upload_to_gcs(self):
        """Upload results file to GCS bucket."""
        if not os.path.exists(self.local_path):
            logger.warning(f"⚠️  Local file not found: {self.local_path}")
            return False
        
        try:
            logger.info(f"☁️  Uploading results to GCS: {self.gcs_path}")
            
            result = subprocess.run(
                ["gsutil", "cp", self.local_path, self.gcs_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✅ Results uploaded to GCS: {self.gcs_path}")
                return True
            else:
                logger.error(f"❌ GCS upload failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ GCS upload timeout (5 min)")
            return False
        except Exception as e:
            logger.error(f"❌ GCS upload error: {e}")
            return False
    
    def finalize(self):
        """
        Finalize results tracking.
        - Save final local copy
        - Upload to GCS
        - Generate summary report
        """
        logger.info("🏁 Finalizing results...")
        
        # Save final local copy
        self._save_local()
        
        # Upload to GCS
        self.upload_to_gcs()
        
        # Generate summary report
        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Local file: {self.local_path}")
        logger.info(f"GCS file: {self.gcs_path}")
        logger.info(f"Timestamp: {self.timestamp}")
        
        if self.summary_data:
            logger.info("\nPipeline Metrics:")
            for pipeline_name, metrics in self.summary_data.items():
                logger.info(f"\n{pipeline_name.upper()}:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        if self.detailed_data:
            logger.info("\nDetailed Results:")
            for pipeline_name, data in self.detailed_data.items():
                logger.info(f"  {pipeline_name}: {len(data)} images tracked")
        
        logger.info("=" * 70)
        
        return self.local_path, self.gcs_path


def create_image_result(video_name: str, frame_name: str, image_path: str, 
                       predictions: List, ground_truths: List, 
                       tp: int, fp: int, fn: int,
                       processing_time_sec: float = 0.0) -> Dict[str, Any]:
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
    
    Returns:
        Dictionary with standardized image-level metrics
    """
    return {
        'video': video_name,
        'frame': frame_name,
        'image_path': image_path,
        'num_predictions': len(predictions),
        'num_ground_truths': len(ground_truths),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'processing_time_sec': round(processing_time_sec, 4),
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
