#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-train_runs}"
MODE="${2:-start}"   # start | resume

ARCHS=("deeplabv3" "fpn" "swin_unetr" "unet")
DATASETS=("munich" "lombardia")
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p logs/tmux

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n "launcher"

job_idx=0

for dataset in "${DATASETS[@]}"; do
  for arch in "${ARCHS[@]}"; do
    gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"
    window_name="${dataset}_${arch}"
    log_file="logs/tmux/${window_name}.log"

    if [[ "$MODE" == "resume" ]]; then
      ckpt="./logs/${arch}/${dataset}/train/checkpoints/last.pt"
      cmd="CUDA_VISIBLE_DEVICES=${gpu} python train.py --arch ${arch} --dataset ${dataset} --ckpt_path ${ckpt} 2>&1 | tee ${log_file}"
    else
      cmd="CUDA_VISIBLE_DEVICES=${gpu} python train.py --arch ${arch} --dataset ${dataset} 2>&1 | tee ${log_file}"
    fi

    tmux new-window -t "$SESSION_NAME" -n "$window_name"
    tmux send-keys -t "${SESSION_NAME}:${window_name}" "$cmd" C-m

    echo "[train] ${window_name} -> GPU ${gpu}"
    job_idx=$((job_idx + 1))
  done
done

tmux kill-window -t "${SESSION_NAME}:launcher"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "List windows: tmux list-windows -t $SESSION_NAME"