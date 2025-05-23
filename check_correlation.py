import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
files = ['results_gpu0/summary/detailed_analysis/all_predictions_20250520_121243.csv', 'results_gpu1/summary/detailed_analysis/all_predictions_20250521_014541.csv', 'results_gpu2/summary/detailed_analysis/all_predictions_20250522_032243.csv']
dfs = [pd.read_csv(f) for f in files]
df = pd.concat(dfs, ignore_index=True)

output_dir = "multi_stage_pipeline_results"
os.makedirs(output_dir, exist_ok=True)

# Filter for prob_thres == 0.6 and foci_dist == 200
filtered_df = df[(df['prob_thres'] == 0.6) & (df['foci_dist'] == 200)]

# Ensure one row per WSI by keeping the first occurrence
unique_wsis = filtered_df.drop_duplicates(subset='wsi_name')

print(f"Number of unique WSIs: {unique_wsis['wsi_name'].nunique()}")
print("List of WSI names:")
print(unique_wsis['wsi_name'].tolist())

# Compute correlation
correlation = unique_wsis['predicted_score'].corr(unique_wsis['true_score'])
print(f"Correlation between predicted and true score: {correlation:.4f}")

# Scatter plot
plt.figure()
sns.scatterplot(data=unique_wsis, x='true_score', y='predicted_score')
plt.title('Predicted vs True Score (prob_thres=0.6, foci_dist=200)')
plt.xlabel('True Score')
plt.ylabel('Predicted Score')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/scatter_pred_vs_true.png")

# Histogram of differences
plt.figure()
sns.histplot(unique_wsis['predicted_score'] - unique_wsis['true_score'], bins=10, kde=True)
plt.title('Prediction Error Distribution')
plt.xlabel('Predicted Score - True Score')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/prediction_error_histogram.png")

# Optional heatmap if you'd like:
plt.figure()
heatmap_data = pd.crosstab(unique_wsis['true_score'], unique_wsis['predicted_score'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='Blues')
plt.title('True vs Predicted Score Heatmap')
plt.xlabel('Predicted Score')
plt.ylabel('True Score')
plt.tight_layout()
plt.savefig(f"{output_dir}/heatmap_true_vs_predicted.png")
