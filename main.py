import argparse
import multiprocessing
import os
import sys
import time
import contextlib
import pandas as pd
from datetime import datetime
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console, Group
from rich.panel import Panel

from vis_utils import check_and_download_data, load_json_ground_truth, LOCAL_TRAIN_DIR, LOCAL_JSON_PATH, get_next_version_path
from baseline_strategy import BaselineStrategy
from cpu_strategy import CpuStrategy
import logging
import google.cloud.logging

# Connect to GCP Logging
# StreamToLogger to redirect stdout/stderr to GCP Logger
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass

# Connect to GCP Logging
def setup_logging():
    """Configures logging based on execution mode."""
    if sys.stdout.isatty():
        # Interactive mode: Log to file, keep stdout for Rich
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("run.log"),
            ]
        )
        return False # Interactive Mode
    else:
        # Background mode: Redirect everything to GCP/stdout
        try:
            client = google.cloud.logging.Client()
            client.setup_logging()
            
            # Create dedicated loggers for stdout/stderr to avoid recursion or duplication
            # Note: google.cloud.logging.Client.setup_logging() attaches to the root logger.
            # We want to capture print() calls which go to sys.stdout.
            
            # Using the root logger for stdout capture might be safest if setup_logging() is used.
            # Let's use a specific name to denote strict stdout capture if needed, 
            # Or just use logging.info.
            
            # IMPORTANT: We must ensure we don't create an infinite loop if the logger prints to stdout.
            # Google Cloud Logging handlers usually talk to the API, but some might print to stderr on error.
            
            stdout_logger = logging.getLogger('STDOUT')
            stderr_logger = logging.getLogger('STDERR')
            
            sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
            sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
            
            print("✅ Connected to Google Cloud Logging (Background Mode).")
            
        except Exception as e:
            # If we fail to connect, we can't do much but print to original stderr (if it exists)
            # But we just overwrote it? No, only on success.
            sys.__stderr__.write(f"⚠️ Could not connect to Cloud Logging: {e}\n")
        
        return True # Background Mode

# Output path
OUTPUT_CSV_PATH = './metrics/combined_results.csv'

def worker_target(strategy_class, train_dir, gt_data, queue):
    """
    Worker function to run a strategy.
    """
    try:
        # Redirect stdout/stderr to suppress noise from worker processes
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            start_t = time.time()
            strategy = strategy_class(train_dir)
            results = strategy.run(gt_data, queue)
            end_t = time.time()
            
            queue.put({
                'type': 'done', 
                'results': results, 
                'duration': end_t - start_t,
                'strategy': strategy_class.__name__.replace('Strategy', '') # Hacky but works for now to ID
            })
    except Exception as e:
        queue.put({'type': 'error', 'error': str(e)})

def create_progress_display():
    """Creates the Rich progress bar setup."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    return progress

from collections import defaultdict
from rich import box

def generate_table(video_stats):
    """Creates the Rich table for finished videos with combined stats and a total row."""
    table = Table(title="Completed Videos", box=box.SIMPLE)
    table.add_column("Video", style="magenta")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("Prec", justify="right")
    table.add_column("Rec", justify="right")
    table.add_column("F1", justify="right", style="bold")
    
    # Sort by video name
    all_videos = sorted(video_stats.keys())
    
    # Track totals for averages
    b_totals = {'FPS': [], 'Precision': [], 'Recall': [], 'F1': []}
    s_totals = {'FPS': [], 'Precision': [], 'Recall': [], 'F1': []}
    
    for vid in all_videos:
        stats = video_stats[vid]
        base = stats.get('Baseline')
        strat = stats.get('CPU_Strat')
        
        # Accumulate totals
        if base:
            for k in b_totals: b_totals[k].append(base[k])
        if strat:
            for k in s_totals: s_totals[k].append(strat[k])
        
        # Helper to format cell
        def fmt(key, round_n=2):
            b_val = base[key] if base else None
            s_val = strat[key] if strat else None
            
            if b_val is not None and s_val is not None:
                if key == 'FPS': return f"{b_val:.1f}->{s_val:.1f}"
                return f"{b_val:.{round_n}f}->{s_val:.{round_n}f}"
            elif b_val is not None:
                if key == 'FPS': return f"{b_val:.1f}"
                return f"{b_val:.{round_n}f}"
            elif s_val is not None:
                if key == 'FPS': return f"->{s_val:.1f}"
                return f"->{s_val:.{round_n}f}"
            return "-"

        table.add_row(
            vid,
            fmt('FPS', 1),
            fmt('Precision', 2),
            fmt('Recall', 2),
            fmt('F1', 2)
        )
    
    # Add Total Row
    table.add_section()
    
    def get_avg(totals_dict, key):
        vals = totals_dict[key]
        return sum(vals) / len(vals) if vals else None

    # Helper for total row formatting
    def fmt_avg(key, round_n=2):
        b_avg = get_avg(b_totals, key)
        s_avg = get_avg(s_totals, key)
        
        if b_avg is not None and s_avg is not None:
             if key == 'FPS': return f"{b_avg:.1f}->{s_avg:.1f}"
             return f"{b_avg:.{round_n}f}->{s_avg:.{round_n}f}"
        elif b_avg is not None:
             if key == 'FPS': return f"{b_avg:.1f}"
             return f"{b_avg:.{round_n}f}"
        elif s_avg is not None:
             if key == 'FPS': return f"->{s_avg:.1f}"
             return f"->{s_avg:.{round_n}f}"
        return "-"

    table.add_row(
        "AVERAGE",
        fmt_avg('FPS', 1),
        fmt_avg('Precision', 2),
        fmt_avg('Recall', 2),
        fmt_avg('F1', 2),
        style="bold"
    )
        
    return table

def main():
    parser = argparse.ArgumentParser(description="Run VIS strategies.")
    parser.add_argument('-c', '--config', choices=['baseline', 'strat'], help="Specific strategy to run. Runs both if omitted.")
    args = parser.parse_args()

    # 1. Setup
    check_and_download_data()
    gt_data = load_json_ground_truth(LOCAL_JSON_PATH)
    if not gt_data:
        print("❌ Could not load ground truth. Exiting.")
        return

    # 2. Prepare Workers
    queue = multiprocessing.Queue()
    processes = []
    
    run_baseline = args.config in ['baseline', None]
    run_strat = args.config in ['strat', None]
    
    active_strategies = []
    
    if run_baseline:
        p = multiprocessing.Process(target=worker_target, args=(BaselineStrategy, LOCAL_TRAIN_DIR, gt_data, queue))
        processes.append(p)
        active_strategies.append('Baseline')
        
    if run_strat:
        p = multiprocessing.Process(target=worker_target, args=(CpuStrategy, LOCAL_TRAIN_DIR, gt_data, queue))
        processes.append(p)
        active_strategies.append('CPU_Strat')

import io

def run_test_mode(is_background):
    """Runs a mock workflow to verify logging."""
    print("🚀 [TEST MODE] Starting Mock Pipeline...")
    print("✅ Annotations found locally (MOCKED).")
    print("✅ Training data found in './data_local/trainsm' (MOCKED).")
    
    # Mock Stats for 3 videos
    mock_stats = defaultdict(dict)
    videos = ['TestVid_001', 'TestVid_002', 'TestVid_003']
    
    for vid in videos:
        # Simulate processing time
        time.sleep(1) 
        
        # Baseline Result
        b_stat = {'Video': vid, 'FPS': 15.0, 'Precision': 0.85, 'Recall': 0.80, 'F1': 0.82}
        mock_stats[vid]['Baseline'] = b_stat
        
        # Strategy Result
        time.sleep(0.5)
        s_stat = {'Video': vid, 'FPS': 18.5, 'Precision': 0.88, 'Recall': 0.82, 'F1': 0.85}
        mock_stats[vid]['CPU_Strat'] = s_stat

        if is_background:
            # Combined Log
            # 0001 | FPS=15.0->18.5 | F1=0.82->0.85
            log_msg = f"{vid} | FPS=15.0->18.5 | Prec=0.85->0.88 | Rec=0.80->0.82 | F1=0.82->0.85"
            logging.info(log_msg)
        else:
             print(f"[Combined] Completed {vid}")

    print("✅ [TEST MODE] Pipeline Complete.")
    
    # Generate Table Output
    table = generate_table(mock_stats)
    
    # Send to GCP explicit log
    try:
        # Use StringIO to capture output silently (without printing to redirected stdout)
        # We enforce width to ensure rows aren't wrapped/hidden
        string_io = io.StringIO()
        tmp_console = Console(record=True, width=150, file=string_io)
        tmp_console.print(table)
        final_output = tmp_console.export_text()
        
        full_log = f"--- [TEST MODE] VIS Pipeline Results {datetime.now()} ---\n\n{final_output}Execution Times:\n  - Baseline: 3.2s (Mock)\n  - CPU_Strat: 2.1s (Mock)"
        
        client = google.cloud.logging.Client()
        logger = client.logger("vis-pipeline-results")
        logger.log_text(full_log)
        print("\n✅ Final results sent to GCP Logging (vis-pipeline-results).")
    except Exception as e:
        print(f"\n⚠️ Could not send results to GCP Logging: {e}")

# ... (main function continues)


def main():
    parser = argparse.ArgumentParser(description="Run VIS strategies.")
    parser.add_argument('-c', '--config', choices=['baseline', 'strat', 'test'], help="Specific strategy to run. Runs both if omitted.")
    args = parser.parse_args()

    # 1. Setup Logging & Mode
    # Returns True if background (logging), False if interactive (Rich)
    is_background = setup_logging() 
    
    # 1.5 Handle Test Mode
    if args.config == 'test':
        run_test_mode(is_background)
        return

    check_and_download_data()
    gt_data = load_json_ground_truth(LOCAL_JSON_PATH)
    if not gt_data:
        print("❌ Could not load ground truth. Exiting.")
        return

    # 2. Prepare Workers
    queue = multiprocessing.Queue()
    processes = []
    
    run_baseline = args.config in ['baseline', None]
    run_strat = args.config in ['strat', None]
    
    if run_baseline:
        p = multiprocessing.Process(target=worker_target, args=(BaselineStrategy, LOCAL_TRAIN_DIR, gt_data, queue))
        processes.append(p)
        
    if run_strat:
        p = multiprocessing.Process(target=worker_target, args=(CpuStrategy, LOCAL_TRAIN_DIR, gt_data, queue))
        processes.append(p)

    # 3. Start Execution
    for p in processes:
        p.start()

    # 4. Monitor Loop
    # We maintain video_stats for table (interactive) OR logging (background)
    video_stats = defaultdict(dict)
    
    # We capture final results lists for CSV export
    final_results = {} 
    
    # We capture durations
    durations = {}

    complete_count = 0
    total_strategies = len(processes)
    
    progress = create_progress_display()
    
    # Task IDs map: strategy_name -> task_id
    task_ids = {}
    
    # Define a context manager that is either Rich Live or a dummy
    if not is_background:
        layout = Group(
            Panel(progress, title="Active Progress", border_style="blue"),
            generate_table(video_stats)
        )
        live_ctx = Live(layout, refresh_per_second=10)
    else:
        live_ctx = contextlib.nullcontext()
        
    with live_ctx as live:
        while complete_count < total_strategies:
            if not queue.empty():
                msg = queue.get()
                mtype = msg.get('type')
                
                if mtype == 'init':
                    strat = msg['strategy']
                    if strat not in task_ids and not is_background:
                        tid = progress.add_task(f"[{strat}] Starting...", total=100)
                        task_ids[strat] = tid
                    if is_background:
                        logging.info(f"[{strat}] Initialized. Processing {msg['total_videos']} videos.")
                        
                elif mtype == 'progress':
                    strat = msg['strategy']
                    if not is_background and strat in task_ids:
                        vid_name = msg['video']
                        frame = msg['frame']
                        total_f = msg['total_frames']
                        tid = task_ids[strat]
                        perc = (frame / total_f) * 100
                        progress.update(tid, completed=perc, description=f"[{strat}] {vid_name} ({frame}/{total_f})")
                        
                elif mtype == 'video_done':
                    strat = msg['strategy']
                    stats = msg['stats']
                    # Update stats
                    video_stats[stats['Video']][strat] = stats
                    
                    if not is_background:
                        # Regenerate table
                        new_table = generate_table(video_stats)
                        layout = Group(
                            Panel(progress, title="Active Progress", border_style="blue"),
                            new_table
                        )
                        live.update(layout)
                    else:
                        # Log it only if we have what we expect or partial
                        # We want combined if possible.
                        expected_count = 0
                        if run_baseline: expected_count += 1
                        if run_strat: expected_count += 1
                        
                        current_stats = video_stats[stats['Video']]
                        if len(current_stats) == expected_count:
                            # Both done (or single if only 1 running) -> Log combined line
                            base = current_stats.get('Baseline')
                            strat_res = current_stats.get('CPU_Strat')
                            
                            def fmt_val(key, round_n=2):
                                b_val = base[key] if base else None
                                s_val = strat_res[key] if strat_res else None
                                
                                if b_val is not None and s_val is not None:
                                    if key == 'FPS': return f"{b_val:.1f}->{s_val:.1f}"
                                    return f"{b_val:.{round_n}f}->{s_val:.{round_n}f}"
                                elif b_val is not None:
                                    if key == 'FPS': return f"{b_val:.1f}"
                                    return f"{b_val:.{round_n}f}"
                                elif s_val is not None:
                                    if key == 'FPS': return f"->{s_val:.1f}"
                                    return f"->{s_val:.{round_n}f}"
                                return "-"

                            log_line = f"{stats['Video']} | FPS={fmt_val('FPS', 1)} | Prec={fmt_val('Precision', 2)} | Rec={fmt_val('Recall', 2)} | F1={fmt_val('F1', 2)}"
                            logging.info(log_line)
                    
                elif mtype == 'done':
                    strat = msg['strategy']
                    final_results[strat] = msg['results']
                    durations[strat] = msg['duration']
                    complete_count += 1
                    
                    if is_background:
                        logging.info(f"[{strat}] COMPLETED in {msg['duration']:.2f}s")
                    
                elif mtype == 'error':
                    err = msg['error']
                    if not is_background:
                        live.console.print(f"[red]Error:[/red] {err}")
                    else:
                        logging.error(f"Worker Error: {err}")
                    complete_count += 1
            
            # Check for dead processes
            still_alive = sum(1 for p in processes if p.is_alive())
            if still_alive == 0 and queue.empty():
                break
            
            time.sleep(0.1)

    # 5. Cleanup
    for p in processes:
        p.join()
        
    print("\n💾 Saving results...")
    # Use fallback if strategy didn't report (e.g. error)
    b_res = final_results.get('Baseline', [])
    s_res = final_results.get('CPU_Strat', [])
    
    df = merge_results(b_res, s_res)
    append_to_csv(df, OUTPUT_CSV_PATH)
    
    # 7. Print Durations & Log to GCP
    print("\n⏱️ Execution Times:")
    duration_text = []
    for strat, dur in durations.items():
        line = f"  - {strat}: {dur:.2f}s ({dur/60:.2f} min)"
        print(line)
        duration_text.append(line)

    # Capture final table string
    try:
        # Use StringIO to capture output silently (without printing to redirected stdout)
        string_io = io.StringIO()
        tmp_console = Console(record=True, width=150, file=string_io)
        final_table = generate_table(video_stats)
        tmp_console.print(final_table)
        final_output = tmp_console.export_text()
        
        # Add timestamp and durations
        full_log = f"--- VIS Pipeline Results {datetime.now()} ---\n\n{final_output}Execution Times:\n" + "\n".join(duration_text)
        
        # Send to GCP
        try:
            # We use a separate client/logger to ensure it goes through even if root logger isn't configured for it
            client = google.cloud.logging.Client()
            logger = client.logger("vis-pipeline-results")
            logger.log_text(full_log)
            print("\n✅ Final results sent to GCP Logging (vis-pipeline-results).")
        except Exception as e:
            print(f"\n⚠️ Could not send results to GCP Logging: {e}")
            
    except Exception as e:
        print(f"⚠️ Error preparing final log: {e}")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True) # key for CUDA
    main()
