#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-train_runs}"
MODE="${2:-start}"          # start | resume

ARCHS=("deeplabv3" "fpn" "swin_unetr" "unet")

MUNICH_GPU=0
LOMBARDIA_GPU=1

mkdir -p logs/tmux

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n "munich_queue"

build_cmd() {
  local arch="$1"
  local dataset="$2"
  local gpu="$3"
  local log_file="$4"

  if [[ "$MODE" == "resume" ]]; then
    local ckpt="./logs/${arch}/${dataset}/train/checkpoints/last.pt"
    echo "CUDA_VISIBLE_DEVICES=${gpu} python train.py --arch ${arch} --dataset ${dataset} --ckpt_path ${ckpt} 2>&1 | tee ${log_file}"
  else
    echo "CUDA_VISIBLE_DEVICES=${gpu} python train.py --arch ${arch} --dataset ${dataset} 2>&1 | tee ${log_file}"
  fi
}

# Window 1: Munich queue on GPU 0
tmux send-keys -t "${SESSION_NAME}:munich_queue" "set -euo pipefail" C-m
for arch in "${ARCHS[@]}"; do
  window_name="munich_${arch}"
  log_file="logs/tmux/${window_name}.log"
  cmd="$(build_cmd "$arch" "munich" "$MUNICH_GPU" "$log_file")"

  tmux send-keys -t "${SESSION_NAME}:munich_queue" "echo '[train] ${window_name} -> GPU ${MUNICH_GPU}'" C-m
  tmux send-keys -t "${SESSION_NAME}:munich_queue" "$cmd" C-m
done
tmux send-keys -t "${SESSION_NAME}:munich_queue" "echo '[done] munich queue finished'" C-m

# Window 2: Lombardia queue on GPU 1
tmux new-window -t "$SESSION_NAME" -n "lombardia_queue"
tmux send-keys -t "${SESSION_NAME}:lombardia_queue" "set -euo pipefail" C-m
for arch in "${ARCHS[@]}"; do
  window_name="lombardia_${arch}"
  log_file="logs/tmux/${window_name}.log"
  cmd="$(build_cmd "$arch" "lombardia" "$LOMBARDIA_GPU" "$log_file")"

  tmux send-keys -t "${SESSION_NAME}:lombardia_queue" "echo '[train] ${window_name} -> GPU ${LOMBARDIA_GPU}'" C-m
  tmux send-keys -t "${SESSION_NAME}:lombardia_queue" "$cmd" C-m
done
tmux send-keys -t "${SESSION_NAME}:lombardia_queue" "echo '[done] lombardia queue finished'" C-m

echo "Started tmux session: $SESSION_NAME"
echo "Munich queue   -> GPU ${MUNICH_GPU}"
echo "Lombardia queue -> GPU ${LOMBARDIA_GPU}"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "List windows: tmux list-windows -t $SESSION_NAME"