import os
import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .detect import run_inflammatory_cell_detection

def collect_wsi_paths(input_dir: Path):
    t_scores = []
    wsi_paths = []
    for subdir in sorted(input_dir.glob("t*")):
        if not subdir.is_dir():
            continue
        try:
            t_score = int(subdir.name[1:]) 
        except ValueError:
            continue
        for wsi_file in subdir.glob("*.svs"):
            wsi_paths.append(wsi_file)
            t_scores.append(t_score)
    return wsi_paths, t_scores

def main():
    parser = argparse.ArgumentParser(description="Run detection and correlate with T-score")
    parser.add_argument("--input_dir", required=True, help="Directory with t0/, t1/, ... subfolders of WSIs")
    parser.add_argument("--output_dir", required=True, help="Where to store detection results")
    parser.add_argument("--model_path", default="detection/models/", help="Path to detection model directory")
    parser.add_argument("--force", action="store_true", help="Force re-run detection even if coords exist")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths, t_scores = collect_wsi_paths(input_dir)
    print(f"\n[INFO] Found {len(wsi_paths)} WSI(s) to process:")
    for wsi_path, t_score in zip(wsi_paths, t_scores):
        print(f" - {wsi_path.name} (T-score: {t_score}, Folder: t{t_score})")
    print()
    inflam_counts = []

    total_start = time.time()

    for wsi_path, t_score in zip(wsi_paths, t_scores):
        start_time = time.time()
        wsi_name = wsi_path.stem
        output_subdir = output_dir / "detection" / wsi_name
        output_subdir.mkdir(parents=True, exist_ok=True)

        coords_path = output_subdir / "inflam_cell_mm_coords.npy"
        try:
            if coords_path.exists() and not args.force:
                mm_coords = np.load(coords_path)
            else:
                mm_coords = run_inflammatory_cell_detection(
                    wsi_path=str(wsi_path),
                    output_dir=output_subdir,
                    model_path=args.model_path,
                    visualise=False
                )
                np.save(coords_path, mm_coords)

            count = len(mm_coords)
            inflam_counts.append(count)
            duration = time.time() - start_time
            print(f"[✓] {wsi_name}: {count} cells (T-score: {t_score}) — took {duration:.2f} sec")
        except Exception as e:
            inflam_counts.append(None)
            print(f"[✗] {wsi_name}: Detection failed (T-score: {t_score}) — {str(e)}")

    total_time = time.time() - total_start
    print(f"\n[✓] Processed {len(wsi_paths)} WSIs in {total_time:.2f} seconds.")

    plt.figure(figsize=(6, 4))
    plt.scatter(t_scores, inflam_counts, color="crimson", alpha=0.7)
    plt.xlabel("T Score")
    plt.ylabel("Inflammatory Cell Count")
    plt.title("Inflammatory Cell Count vs T Score")
    plt.grid(True)

    if len(set(t_scores)) > 1:
        coef = np.polyfit(t_scores, inflam_counts, 1)
        xs = np.array([min(t_scores), max(t_scores)])
        ys = coef[0] * xs + coef[1]
        plt.plot(xs, ys, linestyle="--", color="blue", label="Linear Fit")
        plt.legend()

    plot_path = output_dir / "inflam_vs_t_score.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"[✓] Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
