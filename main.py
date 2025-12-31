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
from typing import Dict, Any, List

# Suppress Torch/YOLO internal deprecation warnings as early as possible
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cuda.amp.autocast.*is deprecated.*",
)

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


def generate_run_configurations() -> List[Dict[str, Any]]:
    """
    Generates a list of all pipeline run configurations based on settings.
    This creates permutations for conf_thresh, sahi, interpolation, etc.
    """
    run_configs = []

    # Pipelines enabled in runtime_config.json
    enabled_pipelines = Config.get_runtime_pipelines()

    for base_pipeline_name in enabled_pipelines:
        # Loop through each confidence threshold
        for conf in Config.CONF_THRESHOLDS:
            conf_str = str(conf).replace(".", "")

            # --- SAHI vs. Legacy Tiling Variants ---
            # Strategy 7 does not use tiling, so it's excluded from SAHI runs
            # Strategy 9 is SAHI-native, so it only runs the SAHI variant
            sahi_variants = [True, False]
            if "strategy_7" in base_pipeline_name:
                sahi_variants = [False]
            elif "strategy_9" in base_pipeline_name:
                sahi_variants = [True]

            for use_sahi in sahi_variants:
                run_name = f"{base_pipeline_name}_{conf_str}"

                # --- Base Config ---
                run_config = {
                    "base_pipeline": base_pipeline_name,
                    "conf_thresh": conf,
                    "use_sahi": use_sahi,
                }

                if use_sahi:
                    run_name += "_sahi"

                # --- Interpolation Variants (for Strategy 13) ---
                if "strategy_13" in base_pipeline_name:
                    for use_interp in [True, False]:
                        interp_run_name = run_name
                        if use_interp:
                            interp_run_name += "_interpolation"

                        interp_config = run_config.copy()
                        interp_config["run_name"] = interp_run_name
                        interp_config["use_interpolation"] = use_interp
                        run_configs.append(interp_config)
                else:
                    run_config["run_name"] = run_name
                    run_configs.append(run_config)

    return run_configs


def run_single_pipeline(run_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single pipeline based on a dynamic run configuration.

    Args:
        run_config: A dictionary defining the pipeline and its parameters.

    Returns:
        Dictionary with pipeline results.
    """
    run_name = run_config["run_name"]
    base_pipeline = run_config["base_pipeline"]

    try:
        # Re-initialize logging with a unique file for this specific run
        log_filename = f"{run_name}.log"
        vis_utils.setup_logging(log_name=log_filename)

        logger.info(f"🚀 Starting pipeline: {run_name.upper()}")

        # --- Dynamically build the final config ---
        # 1. Get base config for the strategy
        final_config = Config.get_pipeline_config(base_pipeline).copy()
        # 2. Merge the dynamic run parameters
        final_config.update(run_config)
        # 3. If using SAHI, merge the SAHI-specific parameters
        if final_config.get("use_sahi", False):
            final_config.update(Config.SAHI_CONFIG)

        logger.info(
            f"   Config: { {k: v for k, v in final_config.items() if k != 'run_name'} }"
        )

        start_time = time.time()

        # Get and run the pipeline function, passing the final config
        pipeline_func = get_pipeline(base_pipeline)
        results = pipeline_func(final_config)

        execution_time = time.time() - start_time
        results["execution_time_sec"] = execution_time

        logger.info(f"✅ Pipeline {run_name.upper()} COMPLETED")
        logger.info(f"   Execution time: {execution_time:.2f} seconds")

        return results

    except Exception as e:
        logger.error(f"❌ Pipeline {run_name} FAILED: {e}", exc_info=True)
        return {
            "pipeline": run_name,
            "status": "failed",
            "error": str(e),
            "execution_time_sec": 0,
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

        # --- Generate all run configurations ---
        all_run_configs = generate_run_configurations()
        if not all_run_configs:
            logger.warning(
                "⚠️ No pipeline run configurations were generated. Check `runtime_config.json`. Exiting."
            )
            sys.exit(0)

        logger.info(
            f"✅ Configuration valid. Generated {len(all_run_configs)} run permutations."
        )
        logger.info(f"   Enabled base pipelines: {Config.get_runtime_pipelines()}")

        # Authenticate with GCS
        vis_utils.authenticate_with_gcs_and_handle_errors()

        # Download data
        vis_utils.check_and_download_data_with_error_handling()

        # Determine number of workers
        num_workers = min(Config.MAX_WORKERS, len(all_run_configs))
        logger.info(
            f"⚙️  Parallel execution: {num_workers} workers for {len(all_run_configs)} pipeline runs."
        )

        # Execute pipelines
        logger.info("EXECUTING PIPELINES")
        overall_start = time.time()

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = pool.map(run_single_pipeline, all_run_configs)
        else:
            # Sequential execution for easier debugging
            results = [run_single_pipeline(cfg) for cfg in all_run_configs]

        overall_time = time.time() - overall_start

        # Log results summary
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info(f"Total execution time: {overall_time:.2f} seconds")

        failed_runs = [r for r in results if r.get("status") == "failed"]
        logger.info(
            f"Pipelines completed: {len(results) - len(failed_runs)}/{len(results)}"
        )

        for result in results:
            pipeline = result.get("pipeline", "unknown")
            status = result.get("status", "completed")
            exec_time = result.get("execution_time_sec", 0)

            if status == "failed":
                logger.error(
                    f"  ❌ {pipeline}: FAILED - {result.get('error', 'Unknown error')}"
                )
            else:
                logger.info(f"  ✅ {pipeline}: {exec_time:.2f}s")

        # Finalize and upload consolidated results
        logger.info("📊 Finalizing results...")
        tracker = csv_utils.get_results_tracker()
        local_path, gcs_path = tracker.finalize()

        logger.info("✅ ALL PIPELINES COMPLETED SUCCESSFULLY")
        logger.info(f"Results: {local_path}")
        logger.info(f"GCS: {gcs_path}")

        vis_utils.trigger_vm_shutdown_if_enabled()

    except KeyboardInterrupt:
        logger.warning("⚠️  Execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"❌ CRITICAL ERROR IN MAIN: {e}", exc_info=True)
        vis_utils.trigger_vm_shutdown_if_enabled(force=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
