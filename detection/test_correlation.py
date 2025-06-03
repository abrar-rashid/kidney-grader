import os
import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .detect import run_inflammatory_cell_detection


def collect_wsi_paths(input_dir: Path):
    wsi_paths = []
    scores = []
    score_types = []

    for subdir in sorted(input_dir.glob("*")):
        if not subdir.is_dir():
            continue
        name = subdir.name.lower()
        if name.startswith("t") and name[1:].isdigit():
            score = int(name[1:])
            score_type = "t"
        elif name.startswith("ti") and name[2:].isdigit():
            score = int(name[2:])
            score_type = "ti"
        else:
            continue

        for wsi_file in subdir.glob("*.svs"):
            wsi_paths.append(wsi_file)
            scores.append(score)
            score_types.append(score_type)

    return wsi_paths, scores, score_types


def plot_score_correlation(scores, counts, score_type, output_dir):
    if not scores:
        return

    plt.figure(figsize=(6, 4))
    plt.scatter(scores, counts, alpha=0.7,
                color="crimson" if score_type == "t" else "teal")
    plt.xlabel(f"{score_type.upper()} Score")
    plt.ylabel("Inflammatory Cell Count")
    plt.title(f"Inflammatory Cell Count vs {score_type.upper()} Score")
    plt.grid(True)

    if len(set(scores)) > 1:
        coef = np.polyfit(scores, counts, 1)
        xs = np.array([min(scores), max(scores)])
        ys = coef[0] * xs + coef[1]
        plt.plot(xs, ys, linestyle="--",
                 color="blue" if score_type == "t" else "orange", label="Linear Fit")
        plt.legend()

    path = output_dir / f"inflam_vs_{score_type}_score.png"
    plt.tight_layout()
    plt.savefig(path)
    print(f"[✓] Saved {score_type.upper()}-score plot to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run detection and correlate with T/ TI-score")
    parser.add_argument("--input_dir", required=True, help="Directory with t0/, ti0/, etc.")
    parser.add_argument("--output_dir", required=True, help="Output folder for results")
    parser.add_argument("--model_path", default="checkpoints/detection/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wsi_paths, scores, score_types = collect_wsi_paths(input_dir)
    print(f"\n[INFO] Found {len(wsi_paths)} WSI(s):")
    for wsi, s, t in zip(wsi_paths, scores, score_types):
        print(f" - {wsi.name} ({t.upper()} score: {s})")

    inflam_counts = []

    total_start = time.time()
    for wsi_path, score, score_type in zip(wsi_paths, scores, score_types):
        start = time.time()
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
            duration = time.time() - start
            print(f"[✓] {wsi_name}: {count} cells ({score_type.upper()} score: {score}) — {duration:.2f}s")
        except Exception as e:
            inflam_counts.append(None)
            print(f"[✗] {wsi_name} failed — {e}")

    print(f"\n[✓] Finished {len(wsi_paths)} WSIs in {time.time() - total_start:.2f} sec.")

    # Split by type for plotting
    for stype in ["t", "ti"]:
        stype_scores = [s for s, t in zip(scores, score_types) if t == stype]
        stype_counts = [c for c, t in zip(inflam_counts, score_types) if t == stype]
        plot_score_correlation(stype_scores, stype_counts, stype, output_dir)


if __name__ == "__main__":
    main()
