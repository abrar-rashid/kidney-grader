#!/bin/bash
# Script to clean up parallel processing setup

echo "Cleaning up parallel processing setup..."

# Kill tmux sessions if they exist
for session in kidney_gpu0 kidney_gpu1 kidney_gpu2; do
  if tmux has-session -t $session 2>/dev/null; then
    echo "Killing tmux session: $session"
    tmux kill-session -t $session
  else
    echo "Tmux session $session not found"
  fi
done

# Remove temporary script files
echo "Removing temporary script files..."
rm -f /data/ar2221/KidneyGrader/run_pipeline_on_wsis_gpu*.py

# Optionally, remove output directories (uncomment if needed)
# echo "WARNING: This will remove all result data! Ctrl+C to cancel (5 seconds)..."
# sleep 5
# rm -rf /data/ar2221/KidneyGrader/results_gpu0
# rm -rf /data/ar2221/KidneyGrader/results_gpu1
# rm -rf /data/ar2221/KidneyGrader/results_gpu2
# rm -rf /data/ar2221/KidneyGrader/results_combined

echo "Cleanup complete. You can now restart the parallel processing." 