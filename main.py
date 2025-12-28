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

# ==========================================
# SUPPRESS C++ LEVEL LOGGING (gRPC, ABSL)
# ==========================================
# 1. Suppress gRPC informational messages and warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""

# 2. Suppress the specific "fork_posix.cc" warning
# This tells gRPC not to attempt registering fork handlers, avoiding the thread conflict warning.
# Safe to use if your child processes don't need to reuse the parent's gRPC connections.
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

# 3. Suppress Abseil (absl) and TensorFlow logs (if present)
# '3' filters out INFO, WARNING, and ERROR logs (only FATAL remain)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
        
        logger.info(f"🚀 Starting pipeline: {pipeline_name.upper()}")
        
        start_time = time.time()
        
        # Get and run pipeline
        pipeline_func = get_pipeline(pipeline_name)
        results = pipeline_func()
        
        execution_time = time.time() - start_time
        results['execution_time_sec'] = execution_time
        
        logger.info(f"✅ Pipeline {pipeline_name.upper()} COMPLETED")
        logger.info(f"   Execution time: {execution_time:.2f} seconds")
        
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
        logger.info("VIS PIPELINE - STARTING")
        
        # Validate configuration
        logger.info("📋 Validating configuration...")
        Config.validate()
        logger.info(f"✅ Configuration valid. Enabled pipelines: {Config.get_runtime_pipelines()}")
        
        # Authenticate with GCS
        logger.info("🔐 Authenticating with Google Cloud Storage...")
        try:
            vis_utils.authenticate_gcs(key_file=Config.SERVICE_ACCOUNT_KEY)
            logger.info("✅ GCS authentication successful")
        except RuntimeError as e:
            logger.critical(f"❌ GCS authentication FAILED: {e}")
            logger.critical("   Cannot proceed without GCS access. Aborting.")
            
            # Trigger killswitch if enabled (to save resources)
            if Config.get_runtime_killswitch():
                logger.info("🔴 Triggering killswitch due to authentication failure...")
                import vm_utils
                vm_utils.shutdown_vm(delay_seconds=10)
            
            sys.exit(1)
        
        # Download training data from GCS
        logger.info("📥 Checking/downloading training data...")
        try:
            vis_utils.check_and_download_data()
            logger.info("✅ Training data ready")
        except Exception as e:
            logger.critical(f"❌ Data download FAILED: {e}")
            
            if Config.get_runtime_killswitch():
                logger.info("🔴 Triggering killswitch due to data download failure...")
                import vm_utils
                vm_utils.shutdown_vm(delay_seconds=10)
            
            sys.exit(1)
        
        # Determine number of workers
        num_workers = min(Config.MAX_WORKERS, len(Config.get_runtime_pipelines()))
        logger.info(f"⚙️  Parallel execution: {num_workers} workers for {len(Config.get_runtime_pipelines())} pipelines")
        
        # Execute pipelines in parallel
        logger.info("EXECUTING PIPELINES")
        
        overall_start = time.time()
        
        if num_workers == 1:
            # Sequential execution
            results = [run_single_pipeline(p) for p in Config.get_runtime_pipelines()]
        else:
            # Parallel execution
            with Pool(processes=num_workers) as pool:
                results = pool.map(run_single_pipeline, Config.get_runtime_pipelines())
        
        overall_time = time.time() - overall_start
        
        # Log results summary
        logger.info("PIPELINE EXECUTION SUMMARY")
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
        logger.info("📊 Finalizing results...")
        tracker = csv_utils.get_results_tracker()
        local_path, gcs_path = tracker.finalize()
        
        logger.info("✅ ALL PIPELINES COMPLETED SUCCESSFULLY")
        logger.info(f"Results: {local_path}")
        logger.info(f"GCS: {gcs_path}")
        
        # Trigger killswitch if enabled
        if Config.get_runtime_killswitch():
            logger.info(f"🔴 Killswitch enabled - VM will shutdown in {Config.KILLSWITCH_DELAY_SECONDS} seconds")
            import vm_utils
            vm_utils.shutdown_vm(delay_seconds=Config.KILLSWITCH_DELAY_SECONDS)
        else:
            logger.info("💡 Killswitch disabled - VM will remain running")
            logger.info("   Remember to stop the VM manually to avoid charges!")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR: {e}", exc_info=True)
        
        if Config.get_runtime_killswitch():
            logger.info("🔴 Triggering killswitch due to critical error...")
            import vm_utils
            vm_utils.shutdown_vm(delay_seconds=10)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
