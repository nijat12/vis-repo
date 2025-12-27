"""
VIS Pipeline - Main Orchestrator

Executes selected pipelines (Baseline, Strategy 7, Strategy 8) in parallel on CPU.
Handles GCS authentication, data download, and VM killswitch.
"""

import os
import sys
import time
import logging
import warnings
from multiprocessing import Pool
from typing import Dict, Any

# Suppress Torch/YOLO internal deprecation warnings as early as possible
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*is deprecated.*")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
import vis_utils
import csv_utils

# Import pipeline modules
from pipelines import get_pipeline

logger = logging.getLogger(__name__)


def run_single_pipeline(pipeline_name: str) -> Dict[str, Any]:
    """
    Execute a single pipeline.
    
    Args:
        pipeline_name: Name of pipeline to run
        
    Returns:
        Dictionary with pipeline results
    """
    try:
        # Re-initialize logging with a unique file for this pipeline
        log_filename = f"{pipeline_name}.log"
        vis_utils.setup_logging(log_name=log_filename)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Starting pipeline: {pipeline_name.upper()}")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Get and run pipeline
        pipeline_func = get_pipeline(pipeline_name)
        results = pipeline_func()
        
        execution_time = time.time() - start_time
        results['execution_time_sec'] = execution_time
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Pipeline {pipeline_name.upper()} COMPLETED")
        logger.info(f"   Execution time: {execution_time:.2f} seconds")
        logger.info(f"{'='*70}\n")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Pipeline {pipeline_name} FAILED: {e}", exc_info=True)
        return {
            'pipeline': pipeline_name,
            'status': 'failed',
            'error': str(e),
            'execution_time_sec': 0
        }


def main():
    """Main execution function."""
    try:
        # Setup logging
        vis_utils.setup_logging()
        logger.info("="*70)
        logger.info("VIS PIPELINE - STARTING")
        logger.info("="*70)
        
        # Validate configuration
        logger.info("📋 Validating configuration...")
        Config.validate()
        logger.info(f"✅ Configuration valid. Enabled pipelines: {Config.ENABLED_PIPELINES}")
        
        # Authenticate with GCS
        logger.info("\n🔐 Authenticating with Google Cloud Storage...")
        try:
            vis_utils.authenticate_gcs(key_file=Config.SERVICE_ACCOUNT_KEY)
            logger.info("✅ GCS authentication successful")
        except RuntimeError as e:
            logger.critical(f"❌ GCS authentication FAILED: {e}")
            logger.critical("   Cannot proceed without GCS access. Aborting.")
            
            # Trigger killswitch if enabled (to save resources)
            if Config.ENABLE_KILLSWITCH:
                logger.info("🔴 Triggering killswitch due to authentication failure...")
                import vm_utils
                vm_utils.shutdown_vm(delay_seconds=10)
            
            sys.exit(1)
        
        # Download training data from GCS
        logger.info("\n📥 Checking/downloading training data...")
        try:
            vis_utils.check_and_download_data()
            logger.info("✅ Training data ready")
        except Exception as e:
            logger.critical(f"❌ Data download FAILED: {e}")
            
            if Config.ENABLE_KILLSWITCH:
                logger.info("🔴 Triggering killswitch due to data download failure...")
                import vm_utils
                vm_utils.shutdown_vm(delay_seconds=10)
            
            sys.exit(1)
        
        # Determine number of workers
        num_workers = min(Config.MAX_WORKERS, len(Config.ENABLED_PIPELINES))
        logger.info(f"\n⚙️  Parallel execution: {num_workers} workers for {len(Config.ENABLED_PIPELINES)} pipelines")
        
        # Execute pipelines in parallel
        logger.info("\n" + "="*70)
        logger.info("EXECUTING PIPELINES")
        logger.info("="*70)
        
        overall_start = time.time()
        
        if num_workers == 1:
            # Sequential execution
            results = [run_single_pipeline(p) for p in Config.ENABLED_PIPELINES]
        else:
            # Parallel execution
            with Pool(processes=num_workers) as pool:
                results = pool.map(run_single_pipeline, Config.ENABLED_PIPELINES)
        
        overall_time = time.time() - overall_start
        
        # Log results summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("="*70)
        logger.info(f"Total execution time: {overall_time:.2f} seconds")
        logger.info(f"Pipelines completed: {len(results)}")
        
        for result in results:
            pipeline = result.get('pipeline', 'unknown')
            status = result.get('status', 'completed')
            exec_time = result.get('execution_time_sec', 0)
            
            if status == 'failed':
                logger.error(f"  ❌ {pipeline}: FAILED - {result.get('error', 'Unknown error')}")
            else:
                logger.info(f"  ✅ {pipeline}: {exec_time:.2f}s")
        
        # Finalize and upload consolidated results
        logger.info("\n📊 Finalizing results...")
        tracker = csv_utils.get_results_tracker()
        local_path, gcs_path = tracker.finalize()
        
        logger.info("="*70)
        logger.info("✅ ALL PIPELINES COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info(f"Results: {local_path}")
        logger.info(f"GCS: {gcs_path}")
        logger.info("="*70)
        
        # Trigger killswitch if enabled
        if Config.ENABLE_KILLSWITCH:
            logger.info(f"\n🔴 Killswitch enabled - VM will shutdown in {Config.KILLSWITCH_DELAY_SECONDS} seconds")
            import vm_utils
            vm_utils.trigger_killswitch(delay_seconds=Config.KILLSWITCH_DELAY_SECONDS)
        else:
            logger.info("\n💡 Killswitch disabled - VM will remain running")
            logger.info("   Remember to stop the VM manually to avoid charges!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"\n❌ CRITICAL ERROR: {e}", exc_info=True)
        
        if Config.ENABLE_KILLSWITCH:
            logger.info("🔴 Triggering killswitch due to critical error...")
            import vm_utils
            vm_utils.trigger_killswitch(delay_seconds=10)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
