import os
import json
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
from datetime import datetime
import time
import argparse
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.console import Group
from rich import box
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize rich console for pretty output
console = Console()

# Directories and file paths
INPUT_DIR = Path("/data/ar2221/all_wsis")
DEFAULT_OUTPUT_DIR = Path("results")  # Default output directory
OUTPUT_DIR = DEFAULT_OUTPUT_DIR  # Will be updated based on command line arguments
SUMMARY_DIR = None  # Will be set after OUTPUT_DIR is determined
BANFF_SCORES = Path("banff_scores.csv")
CHECKPOINT_FILE = None  # Will be set after OUTPUT_DIR is determined

# Parameter combinations to run
PROB_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
FOCI_DISTANCES = [100, 200, 300, 400, 500]

# Create necessary directories will be done in main()

# Load ground truth if available for tracking progress
ground_truth = {}
if BANFF_SCORES.exists():
    try:
        # Load CSV and add debugging
        df = pd.read_csv(BANFF_SCORES)
        console.print(f"[cyan]Found {len(df)} total entries in Banff scores CSV[/cyan]")
        
        # Check for T column
        if 'T' not in df.columns:
            console.print("[bold red]Error: T column not found in Banff scores CSV![/bold red]")
        else:
            # Debug T column data types and missing values
            console.print(f"[cyan]T column data type: {df['T'].dtype}[/cyan]")
            null_count = df['T'].isna().sum()
            console.print(f"[cyan]T column null values: {null_count}[/cyan]")
            
            # Convert to numeric and handle any non-numeric values
            df['T'] = pd.to_numeric(df['T'], errors='coerce')
            
            # Check if any values were coerced to NaN
            new_null_count = df['T'].isna().sum()
            if new_null_count > null_count:
                console.print(f"[yellow]Warning: {new_null_count - null_count} non-numeric values in T column[/yellow]")
        
        # Count only the WSIs with valid T scores
        t_score_count = 0
        for _, row in df.iterrows():
            filename = row['filename']
            wsi_name = Path(filename).stem
            t_score = row.get('T', np.nan)
            
            # Debug each value
            # console.print(f"[dim]WSI: {wsi_name}, T score: {t_score}, Type: {type(t_score)}[/dim]")
            
            # Only care about T score, even if other scores are missing
            # Explicitly check for null values
            if pd.notna(t_score) and t_score != '' and t_score != ' ':
                ground_truth[wsi_name] = float(t_score)
                t_score_count += 1
        
        console.print(f"[green]Loaded {t_score_count} unique WSIs with ground truth T scores from {BANFF_SCORES}[/green]")
        if t_score_count < len(df):
            console.print(f"[yellow]Note: {len(df) - t_score_count} entries were missing T scores[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error loading ground truth scores: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        
# Result tracking for live updates
results_tracker = {
    "processed_wsis": set(),
    "processed_combinations": 0,
    "total_combinations": 0,
    "start_time": None,
    "results": [],
    "t_scores": {0: 0, 1: 0, 2: 0, 3: 0},
    "t_score_correct": {0: 0, 1: 0, 2: 0, 3: 0},
    "t_score_total": {0: 0, 1: 0, 2: 0, 3: 0}
}

# Function to load checkpoint data
def load_checkpoint():
    """Load checkpoint data to enable resuming from the last successful state"""
    if not CHECKPOINT_FILE.exists():
        console.print("[yellow]No checkpoint file found, starting from scratch[/yellow]")
        return {"processed_wsis": {}, "last_updated": None}
        
    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
            
        # Convert processed_wsis keys back to set for backward compatibility
        results_tracker["processed_wsis"] = set(checkpoint.get("processed_wsis", {}).keys())
        
        timestamp = checkpoint.get("last_updated", "unknown")
        console.print(f"[green]Loaded checkpoint from {timestamp}[/green]")
        console.print(f"[green]Found {len(checkpoint.get('processed_wsis', {}))} previously processed WSIs[/green]")
        
        return checkpoint
    except Exception as e:
        console.print(f"[yellow]Error loading checkpoint, starting from scratch: {e}[/yellow]")
        return {"processed_wsis": {}, "last_updated": None}

# Function to save checkpoint data
def save_checkpoint(wsi_name=None, parameters=None):
    """Save checkpoint data to enable resuming from the last successful state"""
    try:
        # Load existing checkpoint if it exists
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
        else:
            checkpoint = {"processed_wsis": {}, "last_updated": None}
            
        # Update with new completed WSI if provided
        if wsi_name:
            if wsi_name not in checkpoint["processed_wsis"]:
                checkpoint["processed_wsis"][wsi_name] = []
                
            # Add the parameter set if provided
            if parameters:
                param_key = f"p{parameters[0]:.2f}".replace('.', '') + f"_d{parameters[1]}"
                if param_key not in checkpoint["processed_wsis"][wsi_name]:
                    checkpoint["processed_wsis"][wsi_name].append(param_key)
        
        # Update timestamp
        checkpoint["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Save updated checkpoint
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f, indent=2)
            
    except Exception as e:
        console.print(f"[yellow]Warning: Failed to save checkpoint: {e}[/yellow]")

# Function to clear checkpoint
def clear_checkpoint():
    """Clear the checkpoint file to force a fresh start"""
    if CHECKPOINT_FILE.exists():
        try:
            os.remove(CHECKPOINT_FILE)
            console.print("[green]Checkpoint file cleared[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to clear checkpoint: {e}[/yellow]")

# Function to run subprocess with scrollable output
def run_subprocess_with_output(command, wsi_name, param_tag):
    """Run subprocess and display output in a scrollable panel"""
    
    # Create a layout for the output
    process_output = []
    max_lines = 15  # Maximum number of visible lines
    
    # Start the subprocess
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    # Dictionary to track stage times
    stages = {
        "segmentation": {"start": None, "end": None},
        "detection": {"start": None, "end": None},
        "quantification": {"start": None, "end": None},
        "grading": {"start": None, "end": None}
    }
    current_stage = None
    
    # Track stage progress outside of live display
    console.print(f"[blue]Processing {wsi_name} with {param_tag}[/blue]")
    
    # Read output line by line without Live display
    for line in iter(process.stdout.readline, ''):
        if line.strip():
            # Check line for stage information
            lower_line = line.lower()
            
            # Track stage timing
            if "segmentation" in lower_line and not stages["segmentation"]["start"]:
                current_stage = "segmentation"
                stages["segmentation"]["start"] = time.time()
                console.print(f"[cyan]► Stage: Segmentation[/cyan]")
            elif "detection" in lower_line and not stages["detection"]["start"]:
                if current_stage == "segmentation":
                    stages["segmentation"]["end"] = time.time()
                    duration = stages["segmentation"]["end"] - stages["segmentation"]["start"]
                    console.print(f"[green]✓ Segmentation completed ({duration:.2f}s)[/green]")
                current_stage = "detection"
                stages["detection"]["start"] = time.time()
                console.print(f"[cyan]► Stage: Detection[/cyan]")
            elif ("quantification" in lower_line or "stage 3" in lower_line) and not stages["quantification"]["start"]:
                if current_stage == "detection":
                    stages["detection"]["end"] = time.time()
                    duration = stages["detection"]["end"] - stages["detection"]["start"]
                    console.print(f"[green]✓ Detection completed ({duration:.2f}s)[/green]")
                current_stage = "quantification"
                stages["quantification"]["start"] = time.time()
                console.print(f"[cyan]► Stage: Quantification[/cyan]")
            elif ("grading" in lower_line or "stage 4" in lower_line or "running stage 4" in lower_line) and not stages["grading"]["start"]:
                if current_stage == "quantification":
                    stages["quantification"]["end"] = time.time()
                    duration = stages["quantification"]["end"] - stages["quantification"]["start"]
                    console.print(f"[green]✓ Quantification completed ({duration:.2f}s)[/green]")
                current_stage = "grading"
                stages["grading"]["start"] = time.time()
                console.print(f"[cyan]► Stage: Grading[/cyan]")
    
    # Finalize any remaining stage
    if current_stage and not stages[current_stage]["end"]:
        stages[current_stage]["end"] = time.time()
        duration = stages[current_stage]["end"] - stages[current_stage]["start"]
        console.print(f"[green]✓ {current_stage.capitalize()} completed ({duration:.2f}s)[/green]")
    
    # Wait for process to complete and get return code
    process.wait()
    return process.returncode == 0

# Step 1: Run the pipeline for each WSI and parameter combination
def run_pipeline(force_rerun=False, visualize=False):
    # Collect all WSI files
    wsi_files = list(INPUT_DIR.glob("*.svs"))
    if not wsi_files:
        console.print("[bold red]No WSI files found in input directory![/bold red]")
        return False
        
    console.print(f"[bold cyan]Found {len(wsi_files)} WSI files to process[/bold cyan]")
    
    # Generate all parameter combinations
    parameter_combinations = list(itertools.product(PROB_THRESHOLDS, FOCI_DISTANCES))
    total_combinations = len(wsi_files) * len(parameter_combinations)
    
    # Load checkpoint data for resuming
    checkpoint = load_checkpoint() if not force_rerun else {"processed_wsis": {}, "last_updated": None}
    processed_wsis = checkpoint.get("processed_wsis", {})
    
    # Count how many combinations are already done
    already_processed = 0
    for wsi, params in processed_wsis.items():
        already_processed += len(params)
    
    if already_processed > 0 and not force_rerun:
        console.print(f"[green]Resuming from checkpoint: {already_processed} combinations already processed[/green]")
    elif force_rerun and already_processed > 0:
        console.print(f"[yellow]Force rerun enabled: ignoring {already_processed} previously processed combinations[/yellow]")
        already_processed = 0
    
    # Initialize progress tracking
    results_tracker["start_time"] = time.time()
    results_tracker["total_combinations"] = total_combinations
    results_tracker["processed_combinations"] = already_processed
    
    # Display setup information
    console.print(Panel(f"""
[bold green]KidneyGrader Pipeline Starting[/bold green]
[cyan]WSIs to process:[/cyan] {len(wsi_files)}
[cyan]Parameter combinations:[/cyan] {len(parameter_combinations)}
[cyan]Total processing jobs:[/cyan] {total_combinations}
[cyan]Already processed:[/cyan] {already_processed} combinations
[cyan]Remaining jobs:[/cyan] {total_combinations - already_processed} combinations
[cyan]Output directory:[/cyan] {OUTPUT_DIR}
[cyan]Resume mode:[/cyan] {"Enabled" if not force_rerun else "Disabled (force rerun)"}
    """, title="Pipeline Setup"))
    
    # Create progress bars
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        overall_task = progress.add_task("[yellow]Overall Progress", total=total_combinations, completed=already_processed)
        wsi_task = progress.add_task("[green]Current WSI", total=len(parameter_combinations))
        
        for wsi_idx, wsi_path in enumerate(wsi_files):
            filename = wsi_path.name
            wsi_name = wsi_path.stem

            # Reset WSI progress
            progress.reset(wsi_task)
            progress.update(wsi_task, description=f"[green]Processing {wsi_name} ({wsi_idx+1}/{len(wsi_files)})")
            
            # If this WSI is already fully processed, skip it
            if not force_rerun and wsi_name in processed_wsis and len(processed_wsis[wsi_name]) == len(parameter_combinations):
                console.print(f"[cyan]Skipping {wsi_name}: already fully processed[/cyan]")
                progress.update(wsi_task, completed=len(parameter_combinations))
                
                # Add to tracking
                results_tracker["processed_wsis"].add(wsi_name)
                
                # Load results for this WSI to include in the results tracker
                load_processed_results(wsi_name)
                
                continue
            
            # Track this WSI
            results_tracker["processed_wsis"].add(wsi_name)
            
            # Process with each parameter combination
            for param_idx, (prob_thres, foci_dist) in enumerate(parameter_combinations):
                param_tag = f"p{prob_thres:.2f}".replace('.', '') + f"_d{foci_dist}"
                
                # Skip if already processed
                if not force_rerun and wsi_name in processed_wsis and param_tag in processed_wsis[wsi_name]:
                    console.print(f"[cyan]Skipping {wsi_name} with {param_tag}: already processed[/cyan]")
                    progress.update(wsi_task, advance=1)
                    
                    # Load result for this combination
                    load_processed_result(wsi_name, param_tag)
                    
                    continue
                
                try:
                    # Show current processing stage
                    console.print(f"[blue]Processing {wsi_name} ({wsi_idx+1}/{len(wsi_files)}) with {param_tag} ({param_idx+1}/{len(parameter_combinations)})[/blue]")
                    
                    # Run the pipeline for this combination
                    console.print(f"[dim cyan]Stage: Starting main.py...[/dim cyan]")
                    
                    # Run subprocess and stream output in real-time
                    # Run quantification and grading in a single command with the new comma-separated format
                    command = [
                        "python3", "main.py",
                        "--input_path", str(wsi_path),
                        "--output_dir", str(OUTPUT_DIR),
                        "--stage", "all",
                        "--prob_thres", str(prob_thres),
                        "--foci_dist", str(foci_dist),
                    ]
                    
                    # Add visualize flag if requested
                    if visualize:
                        command.append("--visualise")
                    
                    success = run_subprocess_with_output(
                        command,
                        wsi_name,
                        param_tag
                    )
                    
                    # Update progress
                    results_tracker["processed_combinations"] += 1
                    progress.update(overall_task, advance=1)
                    progress.update(wsi_task, advance=1)
                    
                    # Read results immediately if available
                    console.print(f"[dim cyan]Stage: Reading results...[/dim cyan]")
                    grading_file = OUTPUT_DIR / "individual_reports" / wsi_name / param_tag / "grading_report.json"
                    
                    if success and grading_file.exists():
                        try:
                            console.print(f"[dim cyan]Stage: Processing results for {wsi_name}...[/dim cyan]")
                            with open(grading_file) as f:
                                grading_data = json.load(f)
                                pred_score = grading_data.get("tubulitis_score_predicted", grading_data.get("tubulitis_score"))  # Check for both new and old field names
                                if pred_score is not None:
                                    # Handle the case when T-score is returned as a string like 't3'
                                    if isinstance(pred_score, str) and pred_score.startswith('t'):
                                        # Extract the numeric part
                                        pred_score_numeric = int(pred_score[1:])
                                        pred_score_str = pred_score
                                    else:
                                        # It's already a numeric value
                                        pred_score_numeric = float(pred_score)
                                        pred_score_str = f"t{round(pred_score_numeric)}"
                                    
                                    # Check if ground truth is already in the grading report (from newer versions)
                                    true_score = grading_data.get("tubulitis_score_ground_truth")
                                    true_score_numeric = None
                                    
                                    # Extract numeric value if ground truth is in t-score format
                                    if isinstance(true_score, str) and true_score.startswith('t'):
                                        true_score_numeric = int(true_score[1:])
                                    elif true_score is not None:
                                        true_score_numeric = float(true_score)
                                    
                                    # If not in grading report, try to get from the global dictionary
                                    if true_score_numeric is None:
                                        true_score_numeric = ground_truth.get(wsi_name, None)
                                        if true_score_numeric is not None:
                                            true_score = f"t{int(true_score_numeric)}"
                                    
                                    # Store result
                                    result_entry = {
                                        "wsi_name": wsi_name,
                                        "prob_thres": prob_thres,
                                        "foci_dist": foci_dist,
                                        "param_tag": param_tag,
                                        "predicted_score": pred_score_numeric,
                                        "predicted_score_str": pred_score_str,
                                        "predicted_category": round(pred_score_numeric),
                                        "true_score": true_score_numeric,
                                        "true_score_str": true_score
                                    }
                                    
                                    # If difference and correctness are already in the report, use those
                                    if "score_difference" in grading_data and true_score_numeric is not None:
                                        result_entry["difference"] = grading_data["score_difference"]
                                        result_entry["correct"] = grading_data["correct_category"]
                                    # Otherwise calculate them if ground truth is available
                                    elif true_score_numeric is not None:
                                        result_entry["difference"] = abs(pred_score_numeric - true_score_numeric)
                                        result_entry["correct"] = (round(pred_score_numeric) == round(true_score_numeric))
                                        
                                        # Update statistics
                                        true_cat = int(round(true_score_numeric))
                                        if true_cat in results_tracker["t_score_total"]:
                                            results_tracker["t_score_total"][true_cat] += 1
                                            if result_entry["correct"]:
                                                results_tracker["t_score_correct"][true_cat] += 1
                                    
                                    # Track T-score predictions
                                    pred_category = round(pred_score_numeric)
                                    if pred_category in results_tracker["t_scores"]:
                                        results_tracker["t_scores"][pred_category] += 1
                                        
                                    results_tracker["results"].append(result_entry)
                                    
                                    # Print immediate feedback
                                    result_msg = f"WSI: {wsi_name}, Params: {param_tag}, Predicted: {pred_score_str}"
                                    if true_score is not None:
                                        result_msg += f", True: {true_score}"
                                        if true_score_numeric is not None:
                                            result_msg += f", Diff: {abs(pred_score_numeric - true_score_numeric):.2f}"
                                    
                                    # Report completion with prediction results
                                    console.print(f"[green]✓ Completed {wsi_name} with {param_tag}: {result_msg}[/green]")
                                    
                                    # Update live stats every few combinations
                                    if len(results_tracker["results"]) % 5 == 0:
                                        show_live_stats()
                                    
                                    # Update checkpoint file
                                    console.print(f"[dim cyan]Stage: Updating checkpoint...[/dim cyan]")
                                    save_checkpoint(wsi_name, (prob_thres, foci_dist))
                        except Exception as e:
                            console.print(f"[red]Error reading results for {wsi_name} with {param_tag}: {e}[/red]")

                except Exception as e:
                    console.print(f"[red]Error processing {wsi_name} with prob={prob_thres}, dist={foci_dist}: {e}[/red]")
    
    # Update summary at the end
    console.print("[bold cyan]Processing complete. Updating summary...[/bold cyan]")
    try:
        subprocess.run(
            ["python3", "main.py", 
             "--input_path", str(wsi_files[0]),
             "--output_dir", str(OUTPUT_DIR),
             "--summary_only"],
            check=True
        )
        console.print("[bold green]Summary updated successfully.[/bold green]")
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error updating summary: {e}[/bold red]")
    
    return True

# Function to load results from a previously processed WSI
def load_processed_results(wsi_name):
    """Load results from a previously processed WSI and add to results tracker"""
    individual_reports_dir = OUTPUT_DIR / "individual_reports" / wsi_name
    
    if not individual_reports_dir.exists():
        return
        
    for param_dir in individual_reports_dir.iterdir():
        if not param_dir.is_dir() or not param_dir.name.startswith("p"):
            continue
            
        load_processed_result(wsi_name, param_dir.name)

# Function to load a single processed result
def load_processed_result(wsi_name, param_tag):
    """Load a single processed result and add to results tracker"""
    grading_file = OUTPUT_DIR / "individual_reports" / wsi_name / param_tag / "grading_report.json"
    parameters_file = OUTPUT_DIR / "individual_reports" / wsi_name / param_tag / "parameters.json"
    
    if not grading_file.exists() or not parameters_file.exists():
        return
        
    try:
        with open(grading_file) as f:
            grading_data = json.load(f)
        with open(parameters_file) as f:
            params = json.load(f)
            
        pred_score = grading_data.get("tubulitis_score_predicted", grading_data.get("tubulitis_score"))  # Check for both new and old field names
        if pred_score is not None:
            # Extract parameters from param tag if not in parameters file
            prob_thres = params.get("prob_thres")
            foci_dist = params.get("foci_dist")
            
            if prob_thres is None or foci_dist is None:
                # Try to extract from param_tag format "pXX_dYYY"
                try:
                    parts = param_tag.split("_")
                    prob_thres = float("0." + parts[0][1:]) if parts[0].startswith("p") else None
                    foci_dist = int(parts[1][1:]) if parts[1].startswith("d") else None
                except:
                    pass
            
            # Handle the case when T-score is returned as a string like 't3'
            if isinstance(pred_score, str) and pred_score.startswith('t'):
                # Extract the numeric part
                pred_score_numeric = int(pred_score[1:])
                pred_score_str = pred_score
            else:
                # It's already a numeric value
                pred_score_numeric = float(pred_score)
                pred_score_str = f"t{round(pred_score_numeric)}"
            
            # Check if ground truth is already in the grading report (from newer versions)
            true_score = grading_data.get("tubulitis_score_ground_truth")
            true_score_numeric = None
            
            # Extract numeric value if ground truth is in t-score format
            if isinstance(true_score, str) and true_score.startswith('t'):
                true_score_numeric = int(true_score[1:])
            elif true_score is not None:
                true_score_numeric = float(true_score)
            
            # If not in grading report, try to get from the global dictionary
            if true_score_numeric is None:
                true_score_numeric = ground_truth.get(wsi_name, None)
                if true_score_numeric is not None:
                    true_score = f"t{int(true_score_numeric)}"
            
            # Store result
            result_entry = {
                "wsi_name": wsi_name,
                "prob_thres": prob_thres,
                "foci_dist": foci_dist,
                "param_tag": param_tag,
                "predicted_score": pred_score_numeric,
                "predicted_score_str": pred_score_str,
                "predicted_category": round(pred_score_numeric),
                "true_score": true_score_numeric,
                "true_score_str": true_score
            }
            
            # If difference and correctness are already in the report, use those
            if "score_difference" in grading_data and true_score_numeric is not None:
                result_entry["difference"] = grading_data["score_difference"]
                result_entry["correct"] = grading_data["correct_category"]
            # Otherwise calculate them if ground truth is available
            elif true_score_numeric is not None:
                result_entry["difference"] = abs(pred_score_numeric - true_score_numeric)
                result_entry["correct"] = (round(pred_score_numeric) == round(true_score_numeric))
                
                # Update statistics
                true_cat = int(round(true_score_numeric))
                if true_cat in results_tracker["t_score_total"]:
                    results_tracker["t_score_total"][true_cat] += 1
                    if result_entry["correct"]:
                        results_tracker["t_score_correct"][true_cat] += 1
            
            # Track T-score predictions
            pred_category = round(pred_score_numeric)
            if pred_category in results_tracker["t_scores"]:
                results_tracker["t_scores"][pred_category] += 1
                
            results_tracker["results"].append(result_entry)

    except Exception as e:
        console.print(f"[yellow]Warning: Could not load result for {wsi_name}/{param_tag}: {e}[/yellow]")

# Step 2: Aggregate the predicted and actual T scores
def aggregate_scores():
    """Aggregate all prediction results with ground truth scores"""
    console.print("[bold cyan]Aggregating prediction results...[/bold cyan]")
    
    # Check if we already have results in memory
    if results_tracker["results"]:
        console.print(f"[green]Using {len(results_tracker['results'])} results from memory[/green]")
        results_df = pd.DataFrame(results_tracker["results"])
        console.print(f"[dim cyan]Stage: Converting {len(results_df)} results to DataFrame...[/dim cyan]")
    else:
        console.print(f"[dim cyan]Stage: Collecting results from files...[/dim cyan]")
        # Collect results from files
        predicted_scores = []
        individual_reports_dir = OUTPUT_DIR / "individual_reports"
        
        if not individual_reports_dir.exists():
            console.print("[bold red]No results directory found![/bold red]")
            return None
        
        # Read summary file if it exists to avoid duplicated processing
        aggregated_scores_path = SUMMARY_DIR / "aggregated_scores.csv"
        if aggregated_scores_path.exists():
            console.print(f"[green]Using existing aggregated scores from {aggregated_scores_path}[/green]")
            return pd.read_csv(aggregated_scores_path)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            wsi_dirs = list(individual_reports_dir.iterdir())
            wsi_task = progress.add_task("[green]Collecting results", total=len(wsi_dirs))
            
            for wsi_dir in wsi_dirs:
                if not wsi_dir.is_dir():
                    continue
                    
                wsi_name = wsi_dir.name
                progress.update(wsi_task, description=f"[green]Collecting results for {wsi_name}")
                
                # Check each parameter subdirectory
                for param_dir in wsi_dir.iterdir():
                    if not param_dir.is_dir() or not param_dir.name.startswith("p"):
                        continue
                        
                    # Use the parameter-specific grading report
                    grading_report_path = param_dir / "grading_report.json"
                    parameters_path = param_dir / "parameters.json"
                    
                    if grading_report_path.exists() and parameters_path.exists():
                        try:
                            with open(grading_report_path) as f:
                                grading_report = json.load(f)
                            with open(parameters_path) as f:
                                parameters = json.load(f)
                                
                            predicted_score = grading_report.get("tubulitis_score_predicted", grading_report.get("tubulitis_score"))  # Check for both new and old field names
                            if predicted_score is not None:
                                # Handle the case when T-score is returned as a string like 't3'
                                if isinstance(predicted_score, str) and predicted_score.startswith('t'):
                                    # Extract the numeric part
                                    pred_score_numeric = int(predicted_score[1:])
                                    pred_score_str = predicted_score
                                else:
                                    # It's already a numeric value
                                    pred_score_numeric = float(predicted_score)
                                    pred_score_str = f"t{round(pred_score_numeric)}"
                                
                                entry = {
                                    "wsi_name": wsi_name,
                                    "filename": wsi_name + ".svs",
                                    "predicted_score": pred_score_numeric,
                                    "predicted_score_str": pred_score_str,
                                    "predicted_category": round(pred_score_numeric),
                                    "prob_thres": parameters.get("prob_thres", None),
                                    "foci_dist": parameters.get("foci_dist", None),
                                    "param_tag": parameters.get("param_tag", param_dir.name)
                                }
                                
                                # First check if ground truth is in the grading report (from newer versions)
                                true_score = grading_report.get("tubulitis_score_ground_truth")
                                true_score_numeric = None
                                
                                # Extract numeric value if ground truth is in t-score format
                                if isinstance(true_score, str) and true_score.startswith('t'):
                                    true_score_numeric = int(true_score[1:])
                                elif true_score is not None:
                                    true_score_numeric = float(true_score)
                                
                                # If ground truth is available in the report
                                if true_score_numeric is not None:
                                    entry["true_score"] = true_score_numeric
                                    entry["true_score_str"] = true_score
                                    entry["difference"] = grading_report.get("score_difference", abs(pred_score_numeric - true_score_numeric))
                                    entry["correct"] = grading_report.get("correct_category", (round(pred_score_numeric) == round(true_score_numeric)))
                                # Otherwise check if it's in the global ground truth dictionary
                                elif wsi_name in ground_truth:
                                    true_score_numeric = ground_truth[wsi_name]
                                    entry["true_score"] = true_score_numeric
                                    entry["true_score_str"] = f"t{int(round(true_score_numeric))}"
                                    entry["difference"] = abs(pred_score_numeric - true_score_numeric)
                                    entry["correct"] = (round(pred_score_numeric) == round(true_score_numeric))
                                
                                predicted_scores.append(entry)
                        except Exception as e:
                            console.print(f"[red]Error reading results for {wsi_name}/{param_dir.name}: {e}[/red]")
                
                progress.update(wsi_task, advance=1)
        
        if not predicted_scores:
            console.print("[bold red]No prediction results found![/bold red]")
            return None
            
        results_df = pd.DataFrame(predicted_scores)

    # Load the actual scores with missing value handling
    correlation_df = results_df.copy()
    
    if not "true_score" in correlation_df.columns and BANFF_SCORES.exists():
        try:
            console.print(f"[dim cyan]Stage: Loading ground truth for comparison...[/dim cyan]")
            # Load CSV with specific handling for T scores
            actual_df = pd.read_csv(BANFF_SCORES)
            console.print(f"[cyan]Reading ground truth file with {len(actual_df)} entries[/cyan]")
            
            # Force T column to numeric, handling any non-numeric values
            actual_df["T"] = pd.to_numeric(actual_df["T"], errors='coerce')
            
            # Debug T column
            num_valid_t = actual_df["T"].notna().sum()
            console.print(f"[cyan]Found {num_valid_t} valid T scores in ground truth file[/cyan]")
            
            # Only keep rows with valid T scores
            valid_t_df = actual_df.loc[actual_df["T"].notna(), ["filename", "T"]]
            
            console.print(f"[cyan]After filtering: {len(valid_t_df)} entries with valid T scores[/cyan]")
            
            # Debug unique values
            unique_filenames = valid_t_df["filename"].nunique()
            console.print(f"[cyan]Number of unique filenames with T scores: {unique_filenames}[/cyan]")
            
            # Verify all 103 entries are present
            if len(valid_t_df) != 103:
                console.print(f"[bold yellow]Warning: Expected 103 entries with T scores but found {len(valid_t_df)}[/bold yellow]")
                
                # Show duplicate entries if there are any
                if len(valid_t_df) != unique_filenames:
                    console.print(f"[bold yellow]Found {len(valid_t_df) - unique_filenames} duplicate entries in ground truth file[/bold yellow]")
                    # Find duplicates
                    duplicates = valid_t_df["filename"].value_counts()
                    duplicates = duplicates[duplicates > 1]
                    if len(duplicates) > 0:
                        console.print(f"[yellow]Duplicate files in CSV ({len(duplicates)}):[/yellow]")
                        for dup_file, count in duplicates.items():
                            console.print(f"[yellow]  {dup_file}: appears {count} times[/yellow]")
            
            # Ensure we're matching on the right column
            console.print(f"[dim cyan]Stage: Matching predictions with ground truth...[/dim cyan]")
            correlation_df["filename_check"] = correlation_df["wsi_name"] + ".svs"
            matching_before = sum(correlation_df["filename_check"].isin(valid_t_df["filename"]))
            console.print(f"[cyan]Filename matches before merge: {matching_before}/{len(correlation_df)}[/cyan]")

            # If filename doesn't match, try to fix - this is a fallback
            if matching_before == 0 and "filename" not in correlation_df.columns:
                correlation_df["filename"] = correlation_df["wsi_name"] + ".svs"
                console.print("[yellow]Added filename column for matching[/yellow]")
            
            # Merge with predictions - use left join to keep all predictions
            console.print("[cyan]Merging ground truth with predictions...[/cyan]")
            merge_column = "filename" if "filename" in correlation_df.columns else "filename_check"
            
            correlation_df = pd.merge(
                correlation_df, 
                valid_t_df, 
                left_on=merge_column,
                right_on="filename", 
                how="left"
            )
            
            # Cleanup temporary column if needed
            if "filename_check" in correlation_df.columns:
                correlation_df = correlation_df.drop(columns=["filename_check"])
            
            # Rename T to true_score
            correlation_df = correlation_df.rename(columns={"T": "true_score"})
            
            # Debug merge results
            merged_valid = correlation_df["true_score"].notna().sum()
            console.print(f"[cyan]After merge: {merged_valid} entries have ground truth T scores[/cyan]")
            
            # Calculate difference if we have ground truth
            mask = correlation_df["true_score"].notna()
            correlation_df.loc[mask, "difference"] = (
                correlation_df.loc[mask, "predicted_score"] - correlation_df.loc[mask, "true_score"]
            ).abs()
            correlation_df.loc[mask, "correct"] = (
                correlation_df.loc[mask, "predicted_category"] == correlation_df.loc[mask, "true_score"].round().astype(int)
            )
            
            # Count unique WSIs with ground truth
            wsis_with_truth = correlation_df.loc[mask, "wsi_name"].nunique()
            console.print(f"[green]Successfully matched {wsis_with_truth} unique WSIs with ground truth T scores[/green]")
            
        except Exception as e:
            console.print(f"[bold red]Error merging with ground truth: {e}[/bold red]")
            import traceback
            console.print(traceback.format_exc())

    # Save to dedicated correlation analysis file
    output_csv = SUMMARY_DIR / "correlation_analysis.csv"
    correlation_df.to_csv(output_csv, index=False)
    console.print(f"[green]Saved correlation analysis to {output_csv}[/green]")
    
    # Count records with ground truth
    valid_count = correlation_df["true_score"].notna().sum()
    console.print(f"[green]Found {len(correlation_df)} total predictions, {valid_count} with ground truth[/green]")
    
    return correlation_df

# Step 3: Correlate predicted and actual T scores for each parameter combination
def analyze_correlation(df):
    """Detailed correlation analysis and visualization"""
    console.print("[bold cyan]Running correlation analysis...[/bold cyan]")
    
    if df is None or len(df) == 0:
        console.print("[yellow]No data available for correlation analysis.[/yellow]")
        return
        
    if len(df) < 3:
        console.print("[yellow]Not enough data for correlation analysis (need at least 3 samples).[/yellow]")
        return
    
    # Create detailed analysis
    analysis_dir = analyze_scores()
    
    if not analysis_dir:
        console.print("[yellow]Analysis couldn't be completed.[/yellow]")
        return
    
    console.print(f"[bold green]Correlation analysis complete![/bold green]")
    console.print(f"[green]Detailed reports saved to: {analysis_dir}[/green]")

def show_live_stats():
    """Display live statistics during processing"""
    if not results_tracker["results"]:
        return
        
    # Calculate completion percentage
    processed = results_tracker["processed_combinations"]
    total = results_tracker["total_combinations"]
    percent_complete = (processed / total) * 100 if total > 0 else 0
    
    # Calculate timing
    elapsed = time.time() - results_tracker["start_time"]
    per_combination = elapsed / processed if processed > 0 else 0
    remaining = (total - processed) * per_combination if processed > 0 else 0
    
    # Prepare metrics if ground truth is available
    metrics_table = None
    results_df = pd.DataFrame(results_tracker["results"])
    
    if "true_score" in results_df.columns and not results_df["true_score"].isna().all():
        valid_results = results_df.dropna(subset=["true_score"])
        
        if len(valid_results) > 0:
            # Calculate metrics
            mae = mean_absolute_error(valid_results["true_score"], valid_results["predicted_score"])
            mse = mean_squared_error(valid_results["true_score"], valid_results["predicted_score"])
            rmse = np.sqrt(mse)
            r2 = r2_score(valid_results["true_score"], valid_results["predicted_score"]) if len(valid_results) > 1 else 0
            
            # T-score accuracy
            accuracy = np.mean(valid_results["correct"]) if "correct" in valid_results.columns else 0
            
            # Create metrics table
            metrics_table = Table(title="Performance Metrics")
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            
            metrics_table.add_row("MAE", f"{mae:.4f}")
            metrics_table.add_row("RMSE", f"{rmse:.4f}")
            metrics_table.add_row("R²", f"{r2:.4f}")
            metrics_table.add_row("Accuracy", f"{accuracy:.2%}")
    
    # Create progress table
    progress_table = Table(title="Processing Progress")
    progress_table.add_column("Metric", style="cyan")
    progress_table.add_column("Value", style="yellow")
    
    progress_table.add_row("WSIs Processed", f"{len(results_tracker['processed_wsis'])}")
    progress_table.add_row("Combinations Completed", f"{processed}/{total} ({percent_complete:.1f}%)")
    progress_table.add_row("Time Elapsed", f"{elapsed/60:.1f} minutes")
    progress_table.add_row("Estimated Remaining", f"{remaining/60:.1f} minutes")
    
    # Create T-score distribution table
    t_table = Table(title="T-Score Distribution")
    t_table.add_column("T-Score", style="cyan")
    t_table.add_column("Count", style="green")
    t_table.add_column("Accuracy", style="yellow")
    
    for t_score in sorted(results_tracker["t_scores"].keys()):
        count = results_tracker["t_scores"][t_score]
        total = results_tracker["t_score_total"][t_score]
        correct = results_tracker["t_score_correct"][t_score]
        
        accuracy = f"{correct/total:.1%}" if total > 0 else "N/A"
        t_table.add_row(f"T{t_score}", f"{count}", accuracy)

    # Display tables
    console.print("\n")
    console.print(progress_table)
    
    if metrics_table:
        console.print(metrics_table)
        
    console.print(t_table)
    console.print("\n")

def analyze_scores():
    """Analyze the prediction results and create comprehensive reports"""
    if not results_tracker["results"]:
        console.print("[yellow]No results to analyze[/yellow]")
        return
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results_tracker["results"])
    
    # Create output directory for detailed analysis
    analysis_dir = SUMMARY_DIR / "detailed_analysis"
    analysis_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save raw results
    results_df.to_csv(analysis_dir / f"all_predictions_{timestamp}.csv", index=False)
    
    # Create a confusion matrix if we have ground truth
    if "true_score" in results_df.columns and not results_df["true_score"].isna().all():
        valid_results = results_df.dropna(subset=["true_score"])
        
        # Count unique WSIs being analyzed
        unique_wsis = valid_results["wsi_name"].unique()
        console.print(f"[bold cyan]Analyzing {len(unique_wsis)} unique WSIs with T scores[/bold cyan]")
        
        # Check if we're missing any T scores
        if len(ground_truth) > len(unique_wsis):
            missing_count = len(ground_truth) - len(unique_wsis)
            console.print(f"[yellow]Note: {missing_count} WSIs with ground truth T scores were not analyzed[/yellow]")
            
            # Identify missing WSIs for debugging
            analyzed_wsis = set(unique_wsis)
            all_truth_wsis = set(ground_truth.keys())
            missing_wsis = all_truth_wsis - analyzed_wsis
            
            if len(missing_wsis) <= 10:
                console.print(f"[yellow]Missing WSIs: {', '.join(missing_wsis)}[/yellow]")
            else:
                console.print(f"[yellow]First 10 missing WSIs: {', '.join(list(missing_wsis)[:10])}...[/yellow]")
        
        # Report T-score distribution
        t_score_counts = valid_results.groupby("true_score").size().to_dict()
        console.print("[cyan]T-score distribution in analysis data:[/cyan]")
        for score, count in sorted(t_score_counts.items()):
            console.print(f"[cyan]  T{int(score) if score.is_integer() else score}: {count} samples[/cyan]")
        
        if len(valid_results) > 0:
            # Round scores to integers for confusion matrix
            valid_results["true_category"] = valid_results["true_score"].round().astype(int)
            
            # Calculate overall metrics
            mae = mean_absolute_error(valid_results["true_score"], valid_results["predicted_score"])
            mse = mean_squared_error(valid_results["true_score"], valid_results["predicted_score"])
            rmse = np.sqrt(mse)
            r2 = r2_score(valid_results["true_score"], valid_results["predicted_score"]) if len(valid_results) > 1 else 0
            
            # Create confusion matrix
            conf_matrix = pd.crosstab(
                valid_results["true_category"], 
                valid_results["predicted_category"],
                rownames=["True"],
                colnames=["Predicted"],
                margins=True
            )
            
            # Save confusion matrix
            conf_matrix.to_csv(analysis_dir / f"confusion_matrix_{timestamp}.csv")
            
            # Create confusion matrix heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix.iloc[:-1, :-1], annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(analysis_dir / f"confusion_matrix_{timestamp}.png")
            plt.close()
            
            # Calculate accuracy, precision, recall for each class
            class_metrics = []
            for t_class in sorted(valid_results["true_category"].unique()):
                t_class = int(t_class)
                true_pos = conf_matrix.loc[t_class, t_class] if t_class in conf_matrix.index and t_class in conf_matrix.columns else 0
                false_pos = conf_matrix.loc["All", t_class] - true_pos if t_class in conf_matrix.columns else 0
                false_neg = conf_matrix.loc[t_class, "All"] - true_pos if t_class in conf_matrix.index else 0
                
                precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
                recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                class_metrics.append({
                    "class": f"T{t_class}",
                    "true_positives": true_pos,
                    "false_positives": false_pos,
                    "false_negatives": false_neg,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })
            
            # Save class metrics
            pd.DataFrame(class_metrics).to_csv(analysis_dir / f"class_metrics_{timestamp}.csv", index=False)
            
            # Find best parameter combination
            param_metrics = valid_results.groupby(["prob_thres", "foci_dist"]).apply(
                lambda x: pd.Series({
                    "mae": mean_absolute_error(x["true_score"], x["predicted_score"]),
                    "rmse": np.sqrt(mean_squared_error(x["true_score"], x["predicted_score"])),
                    "r2": r2_score(x["true_score"], x["predicted_score"]) if len(x) > 1 else 0,
                    "count": len(x)
                })
            ).reset_index()
            
            # Sort by MAE (lower is better)
            param_metrics = param_metrics.sort_values("mae")
            param_metrics.to_csv(analysis_dir / f"parameter_performance_{timestamp}.csv", index=False)
            
            # Plot parameter performance
            plt.figure(figsize=(12, 8))
            pivot_mae = param_metrics.pivot_table(
                index="prob_thres", 
                columns="foci_dist", 
                values="mae"
            )
            sns.heatmap(pivot_mae, annot=True, cmap="YlOrRd_r", fmt=".3f")
            plt.title("Mean Absolute Error by Parameter Combination")
            plt.savefig(analysis_dir / f"parameter_mae_{timestamp}.png")
            plt.close()
            
            # Write comprehensive report
            with open(analysis_dir / f"analysis_report_{timestamp}.txt", "w") as f:
                f.write(f"KidneyGrader Analysis Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Overview\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total WSIs processed: {len(results_tracker['processed_wsis'])}\n")
                f.write(f"Total predictions: {len(results_df)}\n")
                f.write(f"Predictions with ground truth: {len(valid_results)}\n\n")
                
                f.write("Overall Metrics\n")
                f.write("-" * 50 + "\n")
                f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
                f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
                f.write(f"R² Score: {r2:.4f}\n\n")
                
                f.write("T-Score Distribution\n")
                f.write("-" * 50 + "\n")
                for t_score in sorted(results_tracker["t_scores"].keys()):
                    count = results_tracker["t_scores"][t_score]
                    total = results_tracker["t_score_total"][t_score]
                    correct = results_tracker["t_score_correct"][t_score]
                    accuracy = f"{correct/total:.1%}" if total > 0 else "N/A"
                    f.write(f"T{t_score}: {count} predictions, Accuracy: {accuracy}\n")
                f.write("\n")
                
                f.write("Best Parameter Combinations\n")
                f.write("-" * 50 + "\n")
                for i, row in param_metrics.head(3).iterrows():
                    f.write(f"{i+1}. prob_thres={row['prob_thres']}, foci_dist={row['foci_dist']}\n")
                    f.write(f"   MAE: {row['mae']:.4f}, RMSE: {row['rmse']:.4f}, R²: {row['r2']:.4f}, Count: {row['count']}\n")
                f.write("\n")
                
                f.write("Class-Specific Metrics\n")
                f.write("-" * 50 + "\n")
                for metrics in class_metrics:
                    f.write(f"Class {metrics['class']}:\n")
                    f.write(f"  Precision: {metrics['precision']:.4f}\n")
                    f.write(f"  Recall: {metrics['recall']:.4f}\n")
                    f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
                    f.write(f"  True Positives: {metrics['true_positives']}\n")
                    f.write(f"  False Positives: {metrics['false_positives']}\n")
                    f.write(f"  False Negatives: {metrics['false_negatives']}\n\n")
            
            # Display summary to console
            # Get total number of WSIs with T scores available
            total_t_scores = len(ground_truth)
            used_t_scores = len(unique_wsis)
            
            console.print(Panel(f"""
[bold green]Analysis Complete![/bold green]

[cyan]WSI Stats:[/cyan]
  Total WSIs with T scores: {total_t_scores}/103
  WSIs included in analysis: {used_t_scores} ({used_t_scores/total_t_scores:.1%})

[cyan]Overall Metrics:[/cyan]
  MAE: {mae:.4f}
  RMSE: {rmse:.4f}
  R²: {r2:.4f}

[cyan]Best Parameter Combination:[/cyan]
  prob_thres={param_metrics.iloc[0]['prob_thres']}, foci_dist={param_metrics.iloc[0]['foci_dist']}
  MAE: {param_metrics.iloc[0]['mae']:.4f}

[cyan]Detailed analysis saved to:[/cyan] {analysis_dir}
            """, title="Analysis Results"))
        
        else:
            console.print("[yellow]No valid ground truth values for comparison[/yellow]")
    else:
        console.print("[yellow]No ground truth available for analysis[/yellow]")
        
    # Return the location of reports
    return analysis_dir

# Main function to run the entire pipeline
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="KidneyGrader Pipeline - Process WSIs with tubulitis scoring")
    parser.add_argument("--force-rerun", action="store_true", help="Force rerun all WSIs, ignoring checkpoint")
    parser.add_argument("--clear-checkpoint", action="store_true", help="Clear the checkpoint file before starting")
    parser.add_argument("--summary-only", action="store_true", help="Only regenerate summary, skip WSI processing")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations for tubules and inflammatory cells")
    parser.add_argument("--output_dir", type=str, help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    args = parser.parse_args()
    
    console.print(Panel("[bold cyan]KidneyGrader Pipeline[/bold cyan]", subtitle="Automatic Banff Scoring for Kidney Biopsies"))
    
    # Set up global directories based on arguments
    global OUTPUT_DIR, SUMMARY_DIR, CHECKPOINT_FILE
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = DEFAULT_OUTPUT_DIR
    
    # Set up dependent directories
    SUMMARY_DIR = OUTPUT_DIR / "summary"
    CHECKPOINT_FILE = OUTPUT_DIR / "pipeline_checkpoint.json"
    
    # Create necessary directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if input directory exists
    if not INPUT_DIR.exists():
        console.print(f"[bold red]Error: Input directory not found: {INPUT_DIR}[/bold red]")
        return
        
    # Handle checkpoint clearing if requested
    if args.clear_checkpoint:
        console.print("[yellow]Clearing checkpoint as requested...[/yellow]")
        clear_checkpoint()
    
    # Ask for confirmation if we're doing a full run
    if not args.summary_only:
        if args.force_rerun:
            console.print(f"[yellow]This will reprocess ALL WSIs in {INPUT_DIR}, ignoring any previous results.[/yellow]")
        else:
            console.print(f"[yellow]This will process WSIs in {INPUT_DIR} that haven't been processed yet.[/yellow]")
        console.print(f"[yellow]Results will be saved to {OUTPUT_DIR}[/yellow]")
    else:
        console.print(f"[yellow]Regenerating summary only, skipping WSI processing.[/yellow]")
    
    # Start timer
    start_time = time.time()
    
    # Run the pipeline if not summary-only mode
    pipeline_success = True
    if not args.summary_only:
        pipeline_success = run_pipeline(force_rerun=args.force_rerun, visualize=args.visualize)
    
    if not pipeline_success:
        console.print("[bold red]Pipeline execution failed.[/bold red]")
        return
    
    # Aggregate results
    console.print("[bold cyan]Aggregating results...[/bold cyan]")
    aggregated_df = aggregate_scores()
    
    # Correlation analysis
    if aggregated_df is not None and not aggregated_df.empty:
        analyze_correlation(aggregated_df)
    
    # Final report
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Count T scores available and used
    t_scores_available = len(ground_truth)
    t_scores_used = 0
    if aggregated_df is not None and "true_score" in aggregated_df.columns:
        # Count unique WSIs with ground truth
        t_scores_used = len(aggregated_df.dropna(subset=["true_score"])["wsi_name"].unique())
    
    # Calculate coverage percentage
    coverage_pct = (t_scores_used / 103) * 100 if t_scores_used > 0 else 0
    
    # Get checkpoint details
    checkpoint_info = "No checkpoint available"
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint = json.load(f)
            processed_count = sum(len(params) for params in checkpoint.get("processed_wsis", {}).values())
            processed_wsis = len(checkpoint.get("processed_wsis", {}))
            last_updated = checkpoint.get("last_updated", "unknown")
            checkpoint_info = f"Last updated: {last_updated}, {processed_wsis} WSIs, {processed_count} combinations"
        except:
            pass
    
    console.print(Panel(f"""
[bold green]Pipeline Complete![/bold green]

[cyan]Processing Stats:[/cyan]
  WSIs Processed: {len(results_tracker['processed_wsis'])}
  Parameter Combinations: {len(PROB_THRESHOLDS) * len(FOCI_DISTANCES)}
  Total Executions: {results_tracker['processed_combinations']}
  
[cyan]T-Score Analysis:[/cyan]
  T scores found in CSV: {t_scores_available} (expected: 103)
  T scores used in analysis: {t_scores_used} ({coverage_pct:.1f}% coverage)
  {f"[bold yellow]Missing {103 - t_scores_available} WSIs with T scores![/bold yellow]" if t_scores_available < 103 else "[green]All 103 WSIs with T scores found[/green]"}
  
[cyan]Checkpoint Status:[/cyan]
  {checkpoint_info}
  
[cyan]Total runtime:[/cyan] {int(hours)}h {int(minutes)}m {int(seconds)}s

[cyan]Results saved to:[/cyan] {OUTPUT_DIR}
[cyan]Analysis saved to:[/cyan] {SUMMARY_DIR}
    """, title="KidneyGrader Summary"))

if __name__ == "__main__":
    main()
