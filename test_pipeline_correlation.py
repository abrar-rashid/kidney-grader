import os
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.panel import Panel
from datetime import datetime
import shutil

from main import KidneyGraderPipeline

def collect_wsi_paths(input_dir: Path):
    wsi_paths = []
    t_scores = []
    for subdir in sorted(input_dir.glob("t*")):
        if not subdir.is_dir():
            continue
        try:
            t_score = int(subdir.name[1:])
        except ValueError:
            continue
        for wsi in subdir.glob("*.svs"):
            wsi_paths.append(wsi)
            t_scores.append(t_score)
    return wsi_paths, t_scores

def main():
    parser = argparse.ArgumentParser(description="Run full pipeline and compare predicted T scores to ground truth")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="checkpoints/improved_unet_best.pth")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths, t_scores_gt = collect_wsi_paths(input_dir)

    console = Console()
    console.print(f"[bold cyan]Found {len(wsi_paths)} WSIs to process:[/bold cyan]")
    for wsi, score in zip(wsi_paths, t_scores_gt):
        console.print(f"[green] - {wsi.name}[/green] (T-score: [bold magenta]{score}[/bold magenta])")

    predicted_scores = []
    inflam_counts = []

    pipeline = KidneyGraderPipeline(output_dir=output_dir, model_path=args.model_path)

    for wsi_path, true_score in zip(wsi_paths, t_scores_gt):
        wsi_name = wsi_path.stem
        console.print()
        console.print(
            Panel.fit(
                f"[bold white]Analyzing WSI:[/bold white] [cyan]{wsi_name}[/cyan]\n"
                f"[bold white]Ground Truth T Score:[/bold white] [magenta]T{true_score}[/magenta]\n"
                f"[grey50]{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/grey50]",
                title="[bold green]⏳ WSI Analysis Started[/bold green]",
                border_style="bright_yellow",
            )
        )

        try:
            result = pipeline.run_pipeline(str(wsi_path), force=args.force, visualise=False)
            predicted_score = int(result["tubulitis_score"][-1])
            predicted_scores.append(predicted_score)

            # Update quantification JSON with GT T-score
            quant_base = output_dir / "quantification" / f"{wsi_name}_quantification.json"
            if quant_base.exists():
                with open(quant_base) as f:
                    data = json.load(f)
                data["ground_truth_t_score"] = true_score
                with open(quant_base, "w") as f:
                    json.dump(data, f, indent=2)
                inflam_counts.append(data["total_inflam_cells"])

                # Move quantification to t-score subdir
                quant_dest = output_dir / "quantification" / f"t{true_score}" / wsi_name
                quant_dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(quant_base), quant_dest / quant_base.name)

            # Move grading report to t-score subdir and append GT T-score
            grading_base = output_dir / "grading" / f"{wsi_name}_grading_report.txt"
            if grading_base.exists():
                with open(grading_base, "a") as f:
                    f.write(f"\nGround Truth T Score: T{true_score}\n")
                grading_dest = output_dir / "grading" / f"t{true_score}" / wsi_name
                grading_dest.mkdir(parents=True, exist_ok=True)
                shutil.move(str(grading_base), grading_dest / grading_base.name)
            else:
                inflam_counts.append(None)

            console.print(f"[bold green]✓ Done:[/bold green] {wsi_name} → Predicted T{predicted_score} with {inflam_counts[-1]} inflam cells")

        except Exception as e:
            console.print(f"[bold red]✗ Error on {wsi_name}:[/bold red] {str(e)}")
            predicted_scores.append(None)
            inflam_counts.append(None)

    # Save summary
    results_path = output_dir / "grading_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "wsi_names": [p.stem for p in wsi_paths],
            "true_scores": t_scores_gt,
            "predicted_scores": predicted_scores,
            "inflam_counts": inflam_counts
        }, f, indent=2)
    console.print(f"[bold green]✓ Saved results to:[/bold green] {results_path}")

    # Analysis & plots
    valid_idx = [i for i, s in enumerate(predicted_scores) if s is not None]
    y_true = np.array([t_scores_gt[i] for i in valid_idx])
    y_pred = np.array([predicted_scores[i] for i in valid_idx])
    cell_counts = np.array([inflam_counts[i] for i in valid_idx])

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, y_pred, color="purple", alpha=0.7)
    plt.xlabel("Ground Truth T Score")
    plt.ylabel("Predicted T Score")
    plt.title("Predicted vs Ground Truth T Scores")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_gt_tscore.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true, cell_counts, color="crimson", alpha=0.7)
    plt.xlabel("Ground Truth T Score")
    plt.ylabel("Inflammatory Cell Count")
    plt.title("Inflam Cells vs T Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "inflam_vs_gt_tscore.png")

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    console.print("\n[bold]Confusion Matrix:[/bold]")
    console.print(cm)
    console.print("\n[bold]Classification Report:[/bold]")
    console.print(classification_report(y_true, y_pred, digits=3))

    mismatch_counter = Counter()
    misclassified_cases = defaultdict(list)
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true != pred:
            wsi_name = wsi_paths[valid_idx[i]].stem
            mismatch_counter[(true, pred)] += 1
            misclassified_cases[(true, pred)].append(wsi_name)

    report_path = output_dir / "misclassification_report.txt"
    with open(report_path, "w") as f:
        f.write("Misclassification Report\n========================\n\n")
        for (gt, pred), count in sorted(mismatch_counter.items()):
            f.write(f"T{gt} misclassified as T{pred}: {count} case(s)\n")
            for wsi in misclassified_cases[(gt, pred)]:
                f.write(f"  - {wsi}\n")
            f.write("\n")

    console.print(f"\n[bold green]✓ Misclassification breakdown saved to:[/bold green] {report_path}")

if __name__ == "__main__":
    main()
