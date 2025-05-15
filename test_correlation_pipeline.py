import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import spearmanr
from rich.console import Console

console = Console()

def collect_results(output_root):
    """Collects grading and quantification results from run_pipeline_batch outputs."""
    results = []
    console.print(f"[blue]Looking for result folders in: {output_root}[/blue]")

    for folder in Path(output_root).glob("output_*"):
        console.print(f"[cyan]Inspecting folder: {folder}[/cyan]")

        # Iterate over all grading and quantification subfolders
        grading_subdirs = list((folder / "grading").glob("*/grading_report.json"))
        quant_subdirs = list((folder / "quantification").glob("*/anon_*_quantification.json"))

        for grading_report in grading_subdirs:
            # Get the corresponding quantification directory based on the subfolder name
            subfolder_name = grading_report.parent.name
            quant_subdir = folder / "quantification" / subfolder_name
            
            quant_jsons = list(quant_subdir.glob("anon_*_quantification.json"))
            summary_stats_csvs = list(quant_subdir.glob("summary_stats.csv"))

            if not quant_jsons:
                console.print(f"[red]No quantification JSON found in {quant_subdir}[/red]")
                continue

            quant_json = quant_jsons[0]  # Take the first match
            summary_stats_csv = summary_stats_csvs[0] if summary_stats_csvs else None

            console.print(f"[magenta]Looking for grading report at: {grading_report}[/magenta]")
            console.print(f"[magenta]Looking for quantification JSON at: {quant_json}[/magenta]")
            console.print(f"[magenta]Looking for summary stats at: {summary_stats_csv}[/magenta]")

            try:
                with open(grading_report, "r") as f:
                    grading_data = json.load(f)
                with open(quant_json, "r") as f:
                    quant_data = json.load(f)

                # Extract data from JSON or CSV
                mean_cells_per_tubule = np.nan
                max_cells_in_tubule = np.nan
                total_cells_per_foci = np.nan
                mean_cells_per_foci = np.nan
                max_cells_per_foci = np.nan

                # Try to read summary statistics from JSON
                if "summary_stats" in quant_data:
                    stats = quant_data["summary_stats"]
                    mean_cells_per_tubule = stats.get("mean_cells_per_tubule", np.nan)
                    max_cells_in_tubule = stats.get("max_cells_in_tubule", np.nan)
                    total_cells_per_foci = stats.get("total_cells", np.nan)

                    # Check if focus stats exist
                    focus_stats = stats.get("focus_stats", [])
                    if focus_stats:
                        mean_cells_per_foci = focus_stats[0].get("mean_cells_per_tubule", np.nan)
                        max_cells_per_foci = focus_stats[0].get("max_cells_in_tubule", np.nan)
                    console.print("[green]Using summary stats from JSON.[/green]")
                # If JSON stats are not available, read from CSV
                elif summary_stats_csv and summary_stats_csv.stat().st_size > 0:
                    summary_df = pd.read_csv(summary_stats_csv)
                    mean_cells_per_tubule = summary_df["mean_cells_per_tubule"].iloc[0]
                    max_cells_in_tubule = summary_df["max_cells_in_tubule"].iloc[0]
                    total_cells_per_foci = summary_df["total_cells"].iloc[0]
                    focus_stats = eval(summary_df["focus_stats"].iloc[0])
                    if focus_stats:
                        mean_cells_per_foci = focus_stats[0].get('mean_cells_per_tubule', np.nan)
                        max_cells_per_foci = focus_stats[0].get('max_cells_in_tubule', np.nan)
                    console.print("[green]Using summary stats from CSV.[/green]")

                # Add to results
                results.append({
                    "wsi_name": subfolder_name,
                    "true_score": folder.name.split('_')[-1][1:],
                    "predicted_score": grading_data.get("Tubulitis Grade", "N/A").replace("t", ""),
                    "total_inflam_cells": quant_data.get("total_inflam_cells", np.nan),
                    "mean_cells_per_tubule": mean_cells_per_tubule,
                    "max_cells_in_tubule": max_cells_in_tubule,
                    "total_cells_per_foci": total_cells_per_foci,
                    "mean_cells_per_foci": mean_cells_per_foci,
                    "max_cells_per_foci": max_cells_per_foci
                })
                console.print(f"[green]Found valid results for: {subfolder_name}[/green]")

            except Exception as e:
                console.print(f"[red]Error processing {subfolder_name}: {e}[/red]")
    
    console.print(f"[yellow]Total valid result sets found: {len(results)}[/yellow]")
    return pd.DataFrame(results)


def save_metrics(metrics_df, output_dir):
    metrics_path = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    console.print(f"[green]Metrics saved to:[/green] {metrics_path}")

def plot_scatter(x, y, xlabel, ylabel, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_correlation(results_df, output_dir):
    plot_scatter(results_df["true_score"], results_df["predicted_score"], 
                 "True T Score", "Predicted T Score", 
                 "Predicted vs True T Score", 
                 output_dir / "pred_vs_true_tscore.png")
    
    metrics = ["total_inflam_cells", "mean_cells_per_tubule", "max_cells_in_tubule",
               "total_cells_per_foci", "mean_cells_per_foci", "max_cells_per_foci"]
    
    for metric in metrics:
        plot_scatter(results_df["true_score"], results_df[metric], 
                     "True T Score", metric.replace("_", " ").title(), 
                     f"{metric.replace('_', ' ').title()} vs True T Score", 
                     output_dir / f"{metric}_vs_tscore.png")
        pearson_corr = np.corrcoef(results_df["true_score"], results_df[metric])[0, 1]
        spearman_corr = spearmanr(results_df["true_score"], results_df[metric]).correlation
        console.print(f"[blue]{metric.replace('_', ' ').title()} Correlation:[/blue] Pearson {pearson_corr:.2f}, Spearman {spearman_corr:.2f}")

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    console.print(f"[green]Confusion matrix saved to:[/green] {output_path}")

def generate_classification_report(y_true, y_pred, output_path):
    report = classification_report(y_true, y_pred, digits=3)
    with open(output_path, "w") as f:
        f.write(report)
    console.print(f"[green]Classification report saved to:[/green] {output_path}")

def main():
    output_root = "outputs_for_correlation"
    output_dir = Path(output_root) / "correlation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df = collect_results(output_root)

    console.print(f"[yellow]Results DataFrame structure:[/yellow]\n{results_df.head()}")
    console.print(f"[yellow]DataFrame Columns:[/yellow] {results_df.columns.tolist()}")

    # Convert columns to numeric
    results_df["true_score"] = pd.to_numeric(results_df["true_score"], errors='coerce')
    results_df["predicted_score"] = pd.to_numeric(results_df["predicted_score"], errors='coerce')

    # Drop rows with missing data
    results_df = results_df.dropna()

    # Save metrics summary
    save_metrics(results_df, output_dir)

    # Generate correlation plots
    plot_correlation(results_df, output_dir)

    # Plot confusion matrix
    y_true = results_df["true_score"].astype(int)
    y_pred = results_df["predicted_score"].astype(int)
    plot_confusion_matrix(y_true, y_pred, output_dir / "confusion_matrix.png")

    # Generate classification report
    generate_classification_report(y_true, y_pred, output_dir / "classification_report.txt")

if __name__ == "__main__":
    main()
