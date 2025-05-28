import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Folders to search
base_dirs = ['results_gpu0', 'results_gpu1', 'results_gpu2']
output_dir = 'multi_stage_pipeline_results'
os.makedirs(output_dir, exist_ok=True)

data = []

for base in base_dirs:
    individual_reports_path = os.path.join(base, 'individual_reports')
    if not os.path.isdir(individual_reports_path):
        continue

    for wsi_folder in os.listdir(individual_reports_path):
        wsi_path = os.path.join(individual_reports_path, wsi_folder)
        if not os.path.isdir(wsi_path):
            continue

        for param_tag in os.listdir(wsi_path):
            if not param_tag.startswith('p060_d200'):
                continue

            param_path = os.path.join(wsi_path, param_tag)
            grading_path = os.path.join(param_path, 'grading_report.json')
            summary_csv = os.path.join(param_path, 'quantification', 'summary_stats.csv')

            if not (os.path.exists(grading_path) and os.path.exists(summary_csv)):
                continue

            try:
                with open(grading_path) as f:
                    grading = json.load(f)
                if grading.get('prob_thres') != 0.6 or grading.get('foci_dist') != 200:
                    continue
                if grading['tubulitis_score_ground_truth'].startswith('t'):
                    gt_score = int(grading['tubulitis_score_ground_truth'][1:])
                else:
                    continue

                summary = pd.read_csv(summary_csv)
                mean_cells = float(summary['mean_cells_per_tubule'].iloc[0])

                data.append({
                    'wsi_name': grading['wsi_name'],
                    'mean_cells_per_tubule': mean_cells,
                    'ground_truth_score': gt_score
                })

            except Exception as e:
                print(f"Failed to process {param_path}: {e}")

# Deduplicate by wsi_name
df = pd.DataFrame(data).drop_duplicates(subset='wsi_name')

# Calculate correlation
corr, _ = pearsonr(df['mean_cells_per_tubule'], df['ground_truth_score'])
print(f"Correlation: {corr:.4f}")

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(df['mean_cells_per_tubule'], df['ground_truth_score'], alpha=0.7)
plt.title(f'Mean Cells per Tubule vs Ground Truth Score\nPearson r = {corr:.2f}')
plt.xlabel('Mean Cells per Tubule')
plt.ylabel('Ground Truth Tubulitis Score')
plt.grid(True)

plot_path = os.path.join(output_dir, 'mean_cells_vs_gt_score.png')
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to {plot_path}")
