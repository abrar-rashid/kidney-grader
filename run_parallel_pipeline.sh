#!/bin/bash
# Script to run KidneyGrader pipeline in parallel on 3 GPUs

# Base directories
BASE_DIR="/data/ar2221/KidneyGrader"

# Define the input and output directories for each GPU
GPU0_INPUT="/data/ar2221/all_wsis/wsi_set1"
GPU1_INPUT="/data/ar2221/all_wsis/wsi_set2"
GPU2_INPUT="/data/ar2221/all_wsis/wsi_set3"

GPU0_OUTPUT="${BASE_DIR}/results_gpu0"
GPU1_OUTPUT="${BASE_DIR}/results_gpu1"
GPU2_OUTPUT="${BASE_DIR}/results_gpu2"

# Create output directories and copy necessary files
for output_dir in "${GPU0_OUTPUT}" "${GPU1_OUTPUT}" "${GPU2_OUTPUT}"; do
  mkdir -p "${output_dir}"
  
  # Copy banff_scores.csv to each output directory to ensure consistent ground truth
  if [ -f "${BASE_DIR}/banff_scores.csv" ]; then
    echo "Copying banff_scores.csv to ${output_dir}"
    cp "${BASE_DIR}/banff_scores.csv" "${output_dir}/"
  else
    echo "Warning: banff_scores.csv not found in ${BASE_DIR}"
  fi
done

# Create separate copies of the pipeline script for each GPU to avoid race conditions
echo "Creating separate copies of run_pipeline_on_wsis.py for each GPU..."

cp "${BASE_DIR}/run_pipeline_on_wsis.py" "${BASE_DIR}/run_pipeline_on_wsis_gpu0.py"
cp "${BASE_DIR}/run_pipeline_on_wsis.py" "${BASE_DIR}/run_pipeline_on_wsis_gpu1.py"
cp "${BASE_DIR}/run_pipeline_on_wsis.py" "${BASE_DIR}/run_pipeline_on_wsis_gpu2.py"

# Modify each copy to use the correct input directory
sed -i "s|INPUT_DIR = Path(\".*\")|INPUT_DIR = Path(\"${GPU0_INPUT}\")|g" "${BASE_DIR}/run_pipeline_on_wsis_gpu0.py"
sed -i "s|INPUT_DIR = Path(\".*\")|INPUT_DIR = Path(\"${GPU1_INPUT}\")|g" "${BASE_DIR}/run_pipeline_on_wsis_gpu1.py"
sed -i "s|INPUT_DIR = Path(\".*\")|INPUT_DIR = Path(\"${GPU2_INPUT}\")|g" "${BASE_DIR}/run_pipeline_on_wsis_gpu2.py"

# Also modify the DEFAULT_OUTPUT_DIR to match our output directories
sed -i "s|DEFAULT_OUTPUT_DIR = Path(\".*\")|DEFAULT_OUTPUT_DIR = Path(\"${GPU0_OUTPUT}\")|g" "${BASE_DIR}/run_pipeline_on_wsis_gpu0.py"
sed -i "s|DEFAULT_OUTPUT_DIR = Path(\".*\")|DEFAULT_OUTPUT_DIR = Path(\"${GPU1_OUTPUT}\")|g" "${BASE_DIR}/run_pipeline_on_wsis_gpu1.py"
sed -i "s|DEFAULT_OUTPUT_DIR = Path(\".*\")|DEFAULT_OUTPUT_DIR = Path(\"${GPU2_OUTPUT}\")|g" "${BASE_DIR}/run_pipeline_on_wsis_gpu2.py"

# Create tmux sessions using simpler command approach
echo "Starting tmux sessions..."

# GPU 0
tmux new-session -d -s kidney_gpu0
tmux send-keys -t kidney_gpu0 "cd ${BASE_DIR}" C-m
tmux send-keys -t kidney_gpu0 "echo 'Starting GPU 0 processing on ${GPU0_INPUT}'" C-m
tmux send-keys -t kidney_gpu0 "CUDA_VISIBLE_DEVICES=0 python run_pipeline_on_wsis_gpu0.py --output_dir ${GPU0_OUTPUT} --clear-checkpoint" C-m

# GPU 1
tmux new-session -d -s kidney_gpu1
tmux send-keys -t kidney_gpu1 "cd ${BASE_DIR}" C-m
tmux send-keys -t kidney_gpu1 "echo 'Starting GPU 1 processing on ${GPU1_INPUT}'" C-m
tmux send-keys -t kidney_gpu1 "CUDA_VISIBLE_DEVICES=1 python run_pipeline_on_wsis_gpu1.py --output_dir ${GPU1_OUTPUT} --clear-checkpoint" C-m

# GPU 2
tmux new-session -d -s kidney_gpu2
tmux send-keys -t kidney_gpu2 "cd ${BASE_DIR}" C-m
tmux send-keys -t kidney_gpu2 "echo 'Starting GPU 2 processing on ${GPU2_INPUT}'" C-m
tmux send-keys -t kidney_gpu2 "CUDA_VISIBLE_DEVICES=2 python run_pipeline_on_wsis_gpu2.py --output_dir ${GPU2_OUTPUT} --clear-checkpoint" C-m

echo "Started processing on 3 GPUs in tmux sessions:"
echo "  - kidney_gpu0: GPU 0, input=${GPU0_INPUT}, output=${GPU0_OUTPUT}"
echo "  - kidney_gpu1: GPU 1, input=${GPU1_INPUT}, output=${GPU1_OUTPUT}"
echo "  - kidney_gpu2: GPU 2, input=${GPU2_INPUT}, output=${GPU2_OUTPUT}"
echo ""
echo "Monitor sessions with:"
echo "  tmux attach-session -t kidney_gpu0"
echo "  tmux attach-session -t kidney_gpu1"
echo "  tmux attach-session -t kidney_gpu2"
echo ""
echo "Detach from a session with Ctrl+B followed by D"
echo ""
echo "IMPORTANT: After all processes complete, run the combine_results.sh script:"
echo "  ./combine_results.sh"
echo ""
echo "This will properly merge all results to create an equivalent output to running on a single machine." 