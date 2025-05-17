#!/bin/bash
# Script to combine results from parallel GPU runs to achieve identical results to a single run

# Base directories
BASE_DIR="/data/ar2221/KidneyGrader"
GPU0_OUTPUT="${BASE_DIR}/results_gpu0"
GPU1_OUTPUT="${BASE_DIR}/results_gpu1"
GPU2_OUTPUT="${BASE_DIR}/results_gpu2"
COMBINED_OUTPUT="${BASE_DIR}/results_combined"

# Create combined output directory
mkdir -p "${COMBINED_OUTPUT}"
mkdir -p "${COMBINED_OUTPUT}/individual_reports"
mkdir -p "${COMBINED_OUTPUT}/summary"

echo "Combining results from:"
echo "  - ${GPU0_OUTPUT}"
echo "  - ${GPU1_OUTPUT}"
echo "  - ${GPU2_OUTPUT}"
echo "Into: ${COMBINED_OUTPUT}"

# Step 1: Copy individual reports from each GPU run
echo "Copying individual reports..."
for gpu_dir in "${GPU0_OUTPUT}" "${GPU1_OUTPUT}" "${GPU2_OUTPUT}"; do
  if [ -d "${gpu_dir}/individual_reports" ]; then
    # Find all WSI directories in individual_reports and copy them
    for wsi_dir in "${gpu_dir}/individual_reports"/*; do
      if [ -d "${wsi_dir}" ]; then
        wsi_name=$(basename "${wsi_dir}")
        echo "  Copying ${wsi_name} from $(basename "${gpu_dir}")"
        cp -r "${wsi_dir}" "${COMBINED_OUTPUT}/individual_reports/"
      fi
    done
  else
    echo "Warning: No individual_reports directory found in ${gpu_dir}"
  fi
done

# Step 2: Merge checkpoint files to create a combined checkpoint
echo "Merging checkpoint data..."
COMBINED_CHECKPOINT="${COMBINED_OUTPUT}/pipeline_checkpoint.json"
echo "{\"processed_wsis\": {}, \"last_updated\": \"$(date +'%Y-%m-%d %H:%M:%S')\"}" > "${COMBINED_CHECKPOINT}"

# Function to merge checkpoint data
merge_checkpoint() {
  local gpu_checkpoint="$1"
  if [ -f "${gpu_checkpoint}" ]; then
    echo "  Processing checkpoint: ${gpu_checkpoint}"
    # Extract WSIs from this checkpoint and add to combined checkpoint
    python3 -c "
import json
import sys

try:
    # Load the combined checkpoint
    with open('${COMBINED_CHECKPOINT}', 'r') as f:
        combined = json.load(f)
    
    # Load the GPU checkpoint
    with open('${gpu_checkpoint}', 'r') as f:
        gpu_data = json.load(f)
    
    # Merge processed WSIs
    for wsi, params in gpu_data.get('processed_wsis', {}).items():
        if wsi not in combined['processed_wsis']:
            combined['processed_wsis'][wsi] = []
        
        # Add all parameter combinations
        for param in params:
            if param not in combined['processed_wsis'][wsi]:
                combined['processed_wsis'][wsi].append(param)
    
    # Save updated combined checkpoint
    with open('${COMBINED_CHECKPOINT}', 'w') as f:
        json.dump(combined, f, indent=2)
    
    print(f'Added {len(gpu_data.get(\"processed_wsis\", {}))} WSIs from {sys.argv[1]}')
except Exception as e:
    print(f'Error merging checkpoint {sys.argv[1]}: {e}')
" "${gpu_checkpoint}"
  else
    echo "  Warning: Checkpoint file not found: ${gpu_checkpoint}"
  fi
}

# Merge all checkpoints
for gpu_dir in "${GPU0_OUTPUT}" "${GPU1_OUTPUT}" "${GPU2_OUTPUT}"; do
  merge_checkpoint "${gpu_dir}/pipeline_checkpoint.json"
done

# Step 3: Copy any other necessary files (correlation_analysis.csv, etc.)
echo "Copying additional files..."
for file in "correlation_analysis.csv" "banff_scores.csv"; do
  for gpu_dir in "${GPU0_OUTPUT}" "${GPU1_OUTPUT}" "${GPU2_OUTPUT}"; do
    if [ -f "${gpu_dir}/${file}" ]; then
      echo "  Found ${file} in ${gpu_dir}"
      cp "${gpu_dir}/${file}" "${COMBINED_OUTPUT}/"
      break
    fi
  done
done

# Step 4: Copy any summary files that might be useful for reference
echo "Copying summary files for reference..."
for gpu_dir in "${GPU0_OUTPUT}" "${GPU1_OUTPUT}" "${GPU2_OUTPUT}"; do
  if [ -d "${gpu_dir}/summary" ]; then
    mkdir -p "${COMBINED_OUTPUT}/summary/from_$(basename ${gpu_dir})"
    cp -r "${gpu_dir}/summary"/* "${COMBINED_OUTPUT}/summary/from_$(basename ${gpu_dir})/"
  fi
done

echo "Individual reports combined successfully."

# Step 5: Make a temporary copy of the run_pipeline_on_wsis.py to avoid conflicts with running processes
echo "Creating a clean environment for summary generation..."
TMP_SCRIPT="${COMBINED_OUTPUT}/run_pipeline_on_wsis_for_summary.py"
cp "${BASE_DIR}/run_pipeline_on_wsis.py" "${TMP_SCRIPT}"

# Modify the temporary script to use the combined directory
# This ensures aggregation considers all WSIs across all directories
sed -i "s|INPUT_DIR = Path(\".*\")|INPUT_DIR = Path(\"/data/ar2221/all_wsis\")|g" "${TMP_SCRIPT}"
sed -i "s|DEFAULT_OUTPUT_DIR = Path(\".*\")|DEFAULT_OUTPUT_DIR = Path(\"${COMBINED_OUTPUT}\")|g" "${TMP_SCRIPT}"
sed -i "s|OUTPUT_DIR = .*|OUTPUT_DIR = Path(\"${COMBINED_OUTPUT}\")|g" "${TMP_SCRIPT}"

echo "Now generating combined summary..."

# Step 6: Run the aggregation and analysis part only
cd "${COMBINED_OUTPUT}"
python3 "${TMP_SCRIPT}" --summary-only

# Cleanup the temporary script
echo "Cleaning up temporary files..."
rm -f "${TMP_SCRIPT}"

echo "Process complete! Combined results and summary available at: ${COMBINED_OUTPUT}"
echo ""
echo "To verify the combined results match what would happen in a single run:"
echo "1. Check that evaluation_metrics.csv includes data from all WSI sets"
echo "2. Verify the best_parameters.csv file contains analysis across all WSIs"
echo "3. Make sure predicted_vs_ground_truth.png shows all data points" 