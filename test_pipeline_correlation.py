# """
# Run the full KidneyGrader pipeline on a folder tree like

#     └── <input_dir>
#         ├── t0
#         │   ├── case‑001.svs
#         │   └── …
#         ├── t1
#         │   └── …
#         └── t3
#             └── …

# and save per‑WSI outputs in

#     <output_dir>/t0/segmentation/…
#     <output_dir>/t0/detection/…
#     <output_dir>/t0/quantification/<wsi>_quantification.json
#     <output_dir>/t0/grading/<wsi>_grading_report.txt
#     …

# while global plots / summaries stay in <output_dir>.
# """

# import argparse
# import json
# import shutil
# from collections import Counter, defaultdict
# from datetime import datetime
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# from rich.console import Console
# from rich.panel import Panel
# from scipy.stats import spearmanr
# from sklearn.metrics import confusion_matrix, classification_report

# from main import KidneyGraderPipeline


# # ─────────────────────────── helpers ────────────────────────────
# def collect_wsi_paths(root: Path):
#     wsi_paths, t_scores = [], []
#     for subdir in sorted(root.glob("t*")):
#         if subdir.is_dir() and subdir.name[1:].isdigit():
#             t_score = int(subdir.name[1:])
#             for wsi in subdir.glob("*.svs"):
#                 wsi_paths.append(wsi)
#                 t_scores.append(t_score)
#     return wsi_paths, t_scores


# def calculate_metrics(summary_json):
#     s = summary_json["summary_stats"]
#     metrics = {
#         "total_inflam_cells": summary_json["total_inflam_cells"],
#         "mean_cells_per_tubule": s["mean_cells_per_tubule"],
#         "max_cells_in_tubule": s["max_cells_in_tubule"],
#         "total_cells_per_foci": sum(f["total_cells"] for f in s["focus_stats"]),
#         "mean_cells_per_foci": np.mean([f["mean_cells_per_tubule"] for f in s["focus_stats"]]),
#         "max_cells_per_foci": max(f["max_cells_in_tubule"] for f in s["focus_stats"]),
#     }
#     return metrics


# # ─────────────────────────── main ───────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(
#         description="Run full pipeline and correlate predictions with ground‑truth T scores"
#     )
#     parser.add_argument("--input_dir", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--model_path", default="checkpoints/best_current_model.pth")
#     parser.add_argument("--force", action="store_true")
#     # allow one or many thresholds, e.g. "0.1,0.2,0.5"
#     parser.add_argument(
#         "--prob_thres_list",
#         default="0.5,0.6,0.7,0.8,0.9",
#         help="Comma‑separated probability thresholds used in detection (default 0.70)",
#     )
#     parser.add_argument(
#         "--foci_dist_list",
#         default="100,200,300,400,500,1000",
#         help="Comma-separated list of foci distances (default: 100)",
#     )
#     args = parser.parse_args()

#     input_dir = Path(args.input_dir)
#     root_out = Path(args.output_dir)
#     root_out.mkdir(parents=True, exist_ok=True)
#     thr_list = [float(x) for x in args.prob_thres_list.split(",")]
#     foci_dist_list = [int(x) for x in args.foci_dist_list.split(",")]

#     wsi_paths, t_scores_gt = collect_wsi_paths(input_dir)

#     console = Console()
#     console.print(f"[bold cyan]Found {len(wsi_paths)} WSIs to process:[/bold cyan]")
#     for wsi, s in zip(wsi_paths, t_scores_gt):
#         console.print(f"[green] - {wsi.name}[/green] (T‑score: [magenta]T{s}[/magenta])")

#     # containers for analysis
#     predicted_scores, inflam_counts = [], []
#     avg_cells_per_tubule, max_cells_per_tubule = [], []
#     total_cells_per_foci, mean_cells_per_foci, max_cells_per_foci = [], [], []

#     # ───────────── iterate over cases ─────────────

#     for prob_thres in thr_list:
#         console.rule(f"[bold yellow]Probability threshold p ≥ {prob_thres}")
#         for foci_dist in foci_dist_list:
#             console.rule(f"[bold blue]Probability p ≥ {prob_thres}, Foci Distance: {foci_dist}")
#             output_dir = root_out / f"p{str(prob_thres).replace('.', '')}_d{foci_dist}"
#             output_dir.mkdir(parents=True, exist_ok=True)

#             # reset metric accumulators here (inflam_counts, etc.)
#             predicted_scores, inflam_counts = [], []
#             avg_cells_per_tubule = []
#             max_cells_per_tubule = []
#             total_cells_per_foci = []
#             mean_cells_per_foci = []
#             max_cells_per_foci = []
#             for wsi_path, true_score in zip(wsi_paths, t_scores_gt):
#                 wsi_name = wsi_path.stem
#                 console.print()
#                 console.print(
#                     Panel.fit(
#                         f"[white]Analyzing WSI:[/white] [cyan]{wsi_name}[/cyan]\n"
#                         f"[white]Ground‑Truth T Score:[/white] [magenta]T{true_score}[/magenta]\n"
#                         f"[grey50]{datetime.now():%Y-%m-%d %H:%M:%S}[/grey50]",
#                         title="[bold green]⏳  Analysis Started[/bold green]",
#                         border_style="bright_yellow",
#                     )
#                 )

#                 # create /tX/ sub‑folder and run the pipeline there
#                 t_dir = root_out / f"t{true_score}"
#                 t_dir.mkdir(parents=True, exist_ok=True)
#                 pipeline = KidneyGraderPipeline(output_dir=t_dir, model_path=args.model_path, prob_thres=prob_thres)

#                 try:
#                     result = pipeline.run_pipeline(str(wsi_path), force=args.force, visualise=False)
#                     predicted_score = int(result["tubulitis_score"][-1])
#                     predicted_scores.append(predicted_score)

#                     # ---------- quantification JSON ----------------------------------
#                     quant_path = t_dir / "quantification" / f"{wsi_name}_quantification.json"
#                     if quant_path.exists():
#                         with quant_path.open() as f:
#                             data = json.load(f)

#                         # add GT score back into JSON for reference
#                         data["ground_truth_t_score"] = true_score
#                         with quant_path.open("w") as f:
#                             json.dump(data, f, indent=2)

#                         # metrics
#                         m = calculate_metrics(data)
#                         inflam_counts.append(m["total_inflam_cells"])
#                         avg_cells_per_tubule.append(m["mean_cells_per_tubule"])
#                         max_cells_per_tubule.append(m["max_cells_in_tubule"])
#                         total_cells_per_foci.append(m["total_cells_per_foci"])
#                         mean_cells_per_foci.append(m["mean_cells_per_foci"])
#                         max_cells_per_foci.append(m["max_cells_per_foci"])
#                     else:
#                         # keep lengths aligned
#                         inflam_counts.append(None)
#                         avg_cells_per_tubule.append(np.nan)
#                         max_cells_per_tubule.append(np.nan)
#                         total_cells_per_foci.append(np.nan)
#                         mean_cells_per_foci.append(np.nan)
#                         max_cells_per_foci.append(np.nan)

#                     # ---------- append GT score to grading report --------------------
#                     gr_path = Path(result["grading_report"])
#                     if gr_path.exists():
#                         with gr_path.open("a") as f:
#                             f.write(f"\nGround‑Truth T Score: T{true_score}\n")

#                     console.print(
#                         f"[bold green]✓ Done:[/bold green] {wsi_name} → "
#                         f"Predicted T{predicted_score} with {inflam_counts[-1]} inflam cells"
#                     )

#                 except Exception as e:
#                     console.print(f"[bold red]✗ Error on {wsi_name}:[/bold red] {e}")
#                     predicted_scores.append(None)
#                     inflam_counts.append(None)
#                     # pad metric lists
#                     avg_cells_per_tubule.append(np.nan)
#                     max_cells_per_tubule.append(np.nan)
#                     total_cells_per_foci.append(np.nan)
#                     mean_cells_per_foci.append(np.nan)
#                     max_cells_per_foci.append(np.nan)

#     # ───────────── global summary artefacts ─────────────
#     summary = {
#         "wsi_names": [p.stem for p in wsi_paths],
#         "true_scores": t_scores_gt,
#         "predicted_scores": predicted_scores,
#         "inflam_counts": inflam_counts,
#         "avg_cells_per_tubule": avg_cells_per_tubule,
#         "max_cells_per_tubule": max_cells_per_tubule,
#         "total_cells_per_foci": total_cells_per_foci,
#         "mean_cells_per_foci": mean_cells_per_foci,
#         "max_cells_per_foci": max_cells_per_foci,
#     }
#     results_path = root_out / "grading_results.json"
#     with results_path.open("w") as f:
#         json.dump(summary, f, indent=2)
#     console.print(f"[bold green]✓ Saved results to:[/bold green] {results_path}")

#     # ---------- plots -------------------------------------------------------
#     valid = [i for i, p in enumerate(predicted_scores) if p is not None]
#     y_true, y_pred = np.array(summary["true_scores"])[valid], np.array(summary["predicted_scores"])[valid]
#     cell_counts = np.array(summary["inflam_counts"])[valid]

#     plt.figure(figsize=(6, 4))
#     plt.scatter(y_true, y_pred, alpha=0.7, color="purple")
#     plt.xlabel("Ground‑Truth T Score")
#     plt.ylabel("Predicted T Score")
#     plt.title("Predicted vs Ground‑Truth T Scores")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(root_out / "pred_vs_gt_tscore.png")

#     plt.figure(figsize=(6, 4))
#     plt.scatter(y_true, cell_counts, alpha=0.7, color="crimson")
#     plt.xlabel("Ground‑Truth T Score")
#     plt.ylabel("Inflammatory Cell Count")
#     plt.title("Inflam Cells vs T Score")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(root_out / "inflam_vs_gt_tscore.png")

#     # extra metrics
#     metrics_map = {
#         "Average Cells per Tubule": avg_cells_per_tubule,
#         "Maximum Cells per Tubule": max_cells_per_tubule,
#         "Total Cells per Foci": total_cells_per_foci,
#         "Mean Cells per Foci": mean_cells_per_foci,
#         "Maximum Cells per Foci": max_cells_per_foci,
#     }
#     for title, vals in metrics_map.items():
#         plt.figure(figsize=(6, 4))
#         plt.scatter(y_true, vals, alpha=0.7)
#         plt.xlabel("Ground‑Truth T Score")
#         plt.ylabel(title)
#         plt.title(f"{title} vs T Score")
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(root_out / f"{title.replace(' ', '_').lower()}_vs_tscore.png")

#         pearson = np.corrcoef(y_true, vals)[0, 1]
#         spearman = spearmanr(y_true, vals).correlation
#         console.print(f"[blue]{title} Correlation:[/blue] Pearson {pearson:.2f}, Spearman {spearman:.2f}")

#     # confusion matrix & report
#     cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
#     console.print("\n[bold]Confusion Matrix:[/bold]")
#     console.print(cm)
#     console.print("\n[bold]Classification Report:[/bold]")
#     console.print(classification_report(y_true, y_pred, digits=3))

#     # misclassification breakdown
#     mismatches = Counter()
#     cases = defaultdict(list)
#     for idx, (gt, pred) in enumerate(zip(y_true, y_pred)):
#         if gt != pred:
#             name = wsi_paths[valid[idx]].stem
#             mismatches[(gt, pred)] += 1
#             cases[(gt, pred)].append(name)

#     report = root_out / "misclassification_report.txt"
#     with report.open("w") as f:
#         f.write("Misclassification Report\n========================\n\n")
#         for (gt, pred), count in sorted(mismatches.items()):
#             f.write(f"T{gt} misclassified as T{pred}: {count} case(s)\n")
#             for name in cases[(gt, pred)]:
#                 f.write(f"  - {name}\n")
#             f.write("\n")
#     console.print(f"[green]✓ Misclassification breakdown saved to:[/green] {report}")


# if __name__ == "__main__":
#     main()
