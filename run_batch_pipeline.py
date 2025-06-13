# usage: python run_batch_pipeline.py --input_dir /path/to/wsi/files --output_dir batch_results
# only grading with custom params: python run_batch_pipeline.py --input_dir /path/to/wsi/files --stage grade --prob_thres 0.75

import os
import argparse
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

console = Console()

WSI_EXTENSIONS = {'.svs', '.tif', '.tiff'}

def setup_logging(log_file: Path):
    """Setup logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def find_wsi_files(input_dir: Path) -> list[Path]:
    wsi_files = []
    for ext in WSI_EXTENSIONS:
        wsi_files.extend(input_dir.glob(f"*{ext}"))
        wsi_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    return sorted(wsi_files)

def run_single_wsi(wsi_path: Path, output_dir: Path, **kwargs) -> dict:
    wsi_name = wsi_path.stem
    start_time = time.time()
    
    cmd = [
        "python", "main.py",
        "--input_path", str(wsi_path),
        "--output_dir", str(output_dir),
        "--stage", kwargs.get("stage", "all")
    ]
    
    if kwargs.get("force"):
        cmd.append("--force")
    if kwargs.get("visualise"):
        cmd.append("--visualise")
    if kwargs.get("update_summary"):
        cmd.append("--update_summary")
    if kwargs.get("prob_thres") is not None:
        cmd.extend(["--prob_thres", str(kwargs["prob_thres"])])
    if kwargs.get("model_path"):
        cmd.extend(["--model_path", kwargs["model_path"]])
    
    # handle precomputed detections directory
    detection_json = kwargs.get("detection_json")
    if kwargs.get("precomputed_detections_dir") and not detection_json:
        # Auto-locate detection file for this WSI
        precomputed_dir = Path(kwargs["precomputed_detections_dir"])
        wsi_detection_dir = precomputed_dir / wsi_name
        detection_file = wsi_detection_dir / "detected-inflammatory-cells.json"
        if detection_file.exists():
            detection_json = str(detection_file)
        else:
            logging.warning(f"No precomputed detection found for {wsi_name} at {detection_file}")
    
    if detection_json:
        cmd.extend(["--detection_json", detection_json])
    if kwargs.get("instance_mask_class1"):
        cmd.extend(["--instance_mask_class1", kwargs["instance_mask_class1"]])
    
    try:
        if kwargs.get("verbose"):
            result = subprocess.run(
                cmd,
                text=True,
                timeout=kwargs.get("timeout", 10800)
            )
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "wsi_name": wsi_name,
                    "status": "success",
                    "duration": duration,
                    "stdout": "",
                    "stderr": ""
                }
            else:
                return {
                    "wsi_name": wsi_name,
                    "status": "failed",
                    "duration": duration,
                    "stdout": "",
                    "stderr": f"Process failed with return code {result.returncode}",
                    "returncode": result.returncode
                }
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 10800)
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                return {
                    "wsi_name": wsi_name,
                    "status": "success",
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                return {
                    "wsi_name": wsi_name,
                    "status": "failed",
                    "duration": duration,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                }
            
    except subprocess.TimeoutExpired:
        return {
            "wsi_name": wsi_name,
            "status": "timeout",
            "duration": time.time() - start_time,
            "error": f"Timeout after {kwargs.get('timeout', 10800)} seconds"
        }
    except Exception as e:
        return {
            "wsi_name": wsi_name,
            "status": "error",
            "duration": time.time() - start_time,
            "error": str(e)
        }

def run_parallel_batch(wsi_files: list[Path], output_dir: Path, max_workers: int = 1, **kwargs):
    results = []
    failed_wsis = []
    
    with Progress() as progress:
        main_task = progress.add_task("Processing WSIs", total=len(wsi_files))
        
        if max_workers == 1:
            for wsi_path in wsi_files:
                console.print(f"\n[bold blue]Processing:[/bold blue] {wsi_path.name}")
                result = run_single_wsi(wsi_path, output_dir, **kwargs)
                results.append(result)
                
                if result["status"] == "success":
                    console.print(f"[green]✓[/green] {result['wsi_name']} completed in {result['duration']:.1f}s")
                else:
                    console.print(f"[red]✗[/red] {result['wsi_name']} {result['status']}")
                    failed_wsis.append(result)
                    error_msg = result.get('error') or result.get('stderr') or 'Unknown error'
                    logging.error(f"Failed to process {result['wsi_name']}: {error_msg}")
                    if result.get('stdout'):
                        logging.info(f"stdout for {result['wsi_name']}: {result['stdout'][-500:]}")  # last 500 chars
                
                progress.update(main_task, advance=1)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_wsi = {
                    executor.submit(run_single_wsi, wsi_path, output_dir, **kwargs): wsi_path
                    for wsi_path in wsi_files
                }
                
                for future in as_completed(future_to_wsi):
                    wsi_path = future_to_wsi[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result["status"] == "success":
                            console.print(f"[green]✓[/green] {result['wsi_name']} completed in {result['duration']:.1f}s")
                        else:
                            console.print(f"[red]✗[/red] {result['wsi_name']} {result['status']}")
                            failed_wsis.append(result)
                            error_msg = result.get('error') or result.get('stderr') or 'Unknown error'
                            logging.error(f"Failed to process {result['wsi_name']}: {error_msg}")
                            if result.get('stdout'):
                                logging.info(f"stdout for {result['wsi_name']}: {result['stdout'][-500:]}")  # last 500 chars
                    except Exception as e:
                        console.print(f"[red]✗[/red] {wsi_path.name} exception: {e}")
                        failed_wsis.append({
                            "wsi_name": wsi_path.stem,
                            "status": "exception",
                            "error": str(e)
                        })
                    
                    progress.update(main_task, advance=1)
    
    return results, failed_wsis

def print_summary(results: list[dict], failed_wsis: list[dict]):
    console.print("\n[bold cyan]Processing Summary[/bold cyan]")
    
    status_counts = {}
    total_duration = 0
    
    for result in results:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        total_duration += result.get("duration", 0)
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total WSIs", str(len(results)))
    table.add_row("Successful", str(status_counts.get("success", 0)))
    table.add_row("Failed", str(status_counts.get("failed", 0)))
    table.add_row("Timeout", str(status_counts.get("timeout", 0)))
    table.add_row("Errors", str(status_counts.get("error", 0)))
    table.add_row("Total Duration", f"{total_duration:.1f}s")
    table.add_row("Average Duration", f"{total_duration/len(results):.1f}s" if results else "0s")
    
    console.print(table)
    
    if failed_wsis:
        console.print(f"\n[bold red]Failed WSIs ({len(failed_wsis)}):[/bold red]")
        for failed in failed_wsis:
            console.print(f"  • {failed['wsi_name']}: {failed['status']} - {failed.get('error', 'See logs')}")

def main():
    parser = argparse.ArgumentParser(description="Batch process WSI files using KidneyGrader pipeline")
    
    parser.add_argument("--input_dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--output_dir", default="batch_results", help="Output directory for all results")
    
    parser.add_argument("--stage", default="all", 
                       help="Which stage(s) to run (default: all). Options: 1, 2, 3, all, segment, detect, grade")
    parser.add_argument("--force", action="store_true", help="Recompute all stages even if outputs exist")
    parser.add_argument("--visualise", action="store_true", help="Generate visualizations")
    parser.add_argument("--update_summary", action="store_true", help="Update summary files after each WSI")
    
    parser.add_argument("--model_path", default="checkpoints/segmentation/kidney_grader_unet.pth",
                       help="Path to segmentation model checkpoint")
    parser.add_argument("--prob_thres", type=float, default=0.50,
                       help="Probability threshold for inflammatory cell filtering")
    
    parser.add_argument("--detection_json", help="Path to custom detection JSON file")
    parser.add_argument("--instance_mask_class1", help="Path to custom tubule instance mask")
    parser.add_argument("--precomputed_detections_dir", help="Directory containing precomputed detection folders organized by WSI name")
    
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Number of parallel workers (default: 1 for sequential)")
    parser.add_argument("--timeout", type=int, default=10800,
                       help="Timeout per WSI in seconds (default: 10800)")
    
    parser.add_argument("--pattern", help="Only process files matching this pattern (e.g., '*case1*')")
    parser.add_argument("--limit", type=int, help="Limit number of files to process (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Show real-time output from each WSI processing")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory {input_dir} does not exist")
        return 1
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "batch_processing.log"
    setup_logging(log_file)
    
    wsi_files = find_wsi_files(input_dir)
    
    if args.pattern:
        wsi_files = [f for f in wsi_files if f.match(args.pattern)]
    
    if args.limit:
        wsi_files = wsi_files[:args.limit]
    
    if not wsi_files:
        console.print(f"[red]Error:[/red] No WSI files found in {input_dir}")
        console.print(f"Looking for extensions: {', '.join(WSI_EXTENSIONS)}")
        return 1
    
    console.print(f"[green]Found {len(wsi_files)} WSI files to process[/green]")
    console.print(f"[green]Output directory:[/green] {output_dir}")
    console.print(f"[green]Max workers:[/green] {args.max_workers}")
    console.print(f"[green]Stage:[/green] {args.stage}")

    kwargs = {
        "stage": args.stage,
        "force": args.force,
        "visualise": args.visualise,
        "update_summary": args.update_summary,
        "prob_thres": args.prob_thres,
        "model_path": args.model_path,
        "detection_json": args.detection_json,
        "instance_mask_class1": args.instance_mask_class1,
        "precomputed_detections_dir": args.precomputed_detections_dir,
        "timeout": args.timeout,
        "verbose": args.verbose
    }
    
    start_time = time.time()
    console.print(f"\n[bold cyan]Starting batch processing...[/bold cyan]")
    
    results, failed_wsis = run_parallel_batch(wsi_files, output_dir, args.max_workers, **kwargs)
    
    total_time = time.time() - start_time
    
    print_summary(results, failed_wsis)
    console.print(f"\n[bold green]Batch processing completed in {total_time:.1f} seconds[/bold green]")
    
    if args.update_summary and any(r["status"] == "success" for r in results):
        console.print("[yellow]Running final summary update...[/yellow]")
        subprocess.run([
            "python", "main.py",
            "--input_path", "dummy",
            "--output_dir", str(output_dir),
            "--summary_only"
        ])
    
    import json
    results_file = output_dir / "batch_results.json"
    with open(results_file, "w") as f:
        json.dump({"results": results, "failed": failed_wsis, "total_time": total_time}, f, indent=2)
    
    console.print(f"[green]Detailed results saved to:[/green] {results_file}")
    
    return 0 if not failed_wsis else 1

if __name__ == "__main__":
    exit(main()) 