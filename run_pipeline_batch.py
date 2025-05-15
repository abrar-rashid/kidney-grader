import os
import subprocess
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
import json

console = Console()

def extract_predicted_t_score(report_path):
    try:
        # Check if the file is a JSON or a TXT file
        if report_path.suffix == '.json':
            with open(report_path, 'r') as f:
                data = json.load(f)
                print(data)
                return data.get("Tubulitis Grade", "N/A")
        else:  # Fallback to reading as a TXT file
            with open(report_path, 'r') as f:
                for line in f:
                    if "Tubulitis Grade:" in line:
                        return line.split(":")[-1].strip()
    except Exception as e:
        console.print(f"[bold red]Error reading grading report:[/bold red] {report_path} - {e}")
    return "N/A"

def run_pipeline_for_all_slides(input_dir, output_root, model_path, prob_thres=0.50, foci_dist=200):
    # Collect all the slides to be processed
    slides_to_process = []

    # Loop over all subdirectories (t0, g0, ti0, etc.)
    for score_folder in sorted(Path(input_dir).glob("*")):
        if not score_folder.is_dir():
            continue
        
        score_name = score_folder.stem  # Extract score name from folder name (e.g., t0, g1, ti0)
        output_dir = f"{output_root}_{score_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Loop over all slides in the subdirectory
        for slide_path in sorted(score_folder.glob("*.svs")):
            slide_name = slide_path.stem
            slides_to_process.append((score_name, slide_name, slide_path, output_dir))

    # Display the list of slides to be processed using Rich table
    table = Table(title="Slides to be Processed", title_style="bold green")
    table.add_column("Folder", style="cyan")
    table.add_column("Slide Name", style="magenta")
    table.add_column("Path", style="yellow")

    for score_name, slide_name, slide_path, output_dir in slides_to_process:
        table.add_row(score_name, slide_name, str(slide_path))

    console.print(table)

    # Confirm to proceed
    proceed = Confirm.ask("\n[bold blue]Do you want to proceed with running the pipeline on these slides?[/bold blue]")
    if not proceed:
        console.print("[bold red]Aborting the process.[/bold red]")
        return

    # Store the results for summary
    results = []

    # Process each slide
    for score_name, slide_name, slide_path, output_dir in slides_to_process:
        console.print(f"\n[bold green][INFO] Running pipeline for:[/bold green] [cyan]{slide_name}[/cyan] in [blue]{score_name}[/blue] -> Output: [yellow]{output_dir}[/yellow]")

        # Construct the command to run the pipeline
        command = [
            "python", "main.py",
            "--input_path", str(slide_path),
            "--output_dir", output_dir,
            "--stage", "grade",
            "--model_path", model_path,
            "--prob_thres", str(prob_thres),
            "--foci_dist", str(foci_dist),
            "--force"
        ]

        # Execute the pipeline command
        try:
            subprocess.run(command, check=True)
            # Path to the grading report
            prob_tag = f"p{prob_thres:.2f}".replace(".", "")
            grading_report_path = Path(output_dir) / "grading" / f"{slide_name}_{prob_tag}_d{foci_dist}" / "grading_report.json"
            predicted_t_score = extract_predicted_t_score(grading_report_path)
            true_t_score = score_name

            # Store the result
            results.append((slide_name, predicted_t_score, true_t_score))
            console.print(f"[bold green][SUCCESS] Completed for slide:[/bold green] {slide_name}")

        except subprocess.CalledProcessError as e:
            console.print(f"[bold red][ERROR] Pipeline failed for slide:[/bold red] {slide_name} - Error: {e}")
            results.append((slide_name, "Error", score_name))

    # Print the summary table at the end
    result_table = Table(title="Grading Results Summary", title_style="bold yellow")
    result_table.add_column("WSI Name", style="cyan")
    result_table.add_column("Predicted T Score", style="green")
    result_table.add_column("True T Score", style="red")

    for slide_name, predicted_t_score, true_t_score in results:
        result_table.add_row(slide_name, predicted_t_score, true_t_score)

    console.print(result_table)

if __name__ == "__main__":
    input_dir = "data/samples"  # Root directory containing score subdirectories (t0, g0, ti0, etc.)
    output_root = "outputs_current/outputs_for_correlation/output"      # Base output directory (e.g., output_t0, output_g1)
    model_path = "checkpoints/best_current_model.pth"  # Path to the model

    # Set the probability threshold and foci distance
    prob_thres = 0.80
    foci_dist = 500

    run_pipeline_for_all_slides(input_dir, output_root, model_path, prob_thres, foci_dist)
