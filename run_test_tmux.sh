#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-test_runs}"
DELAY="${60:-2}"   # seconds between job launches

ARCHS=("deeplabv3" "fpn" "swin_unetr" "unet")
GPUS=(0 1 2 3 4 5 6 7)

mkdir -p logs/tmux

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n "launcher"

declare -a JOBS=(
  "munich::deeplabv3::./pretrained/deeplab_munich.ckpt"
  "munich::fpn::./pretrained/fpn_munich.ckpt"
  "munich::swin_unetr::./pretrained/swin_munich.ckpt"
  "munich::unet::./pretrained/unet_munich.ckpt"

  "lombardia:A:deeplabv3:./pretrained/deeplab_lombardia.ckpt"
  "lombardia:A:fpn:./pretrained/fpn_lombardia.ckpt"
  "lombardia:A:swin_unetr:./pretrained/swin_lombardia.ckpt"
  "lombardia:A:unet:./pretrained/unet_lombardia.ckpt"

  "lombardia:Y:deeplabv3:./pretrained/deeplab_lombardia.ckpt"
  "lombardia:Y:fpn:./pretrained/fpn_lombardia.ckpt"
  "lombardia:Y:swin_unetr:./pretrained/swin_lombardia.ckpt"
  "lombardia:Y:unet:./pretrained/unet_lombardia.ckpt"
)

job_idx=0

for job in "${JOBS[@]}"; do
  IFS=":" read -r dataset maybe_test_id arch ckpt <<< "$job"

  gpu="${GPUS[$((job_idx % ${#GPUS[@]}))]}"

  if [[ "$dataset" == "munich" ]]; then
    IFS=":" read -r dataset arch ckpt <<< "$job"

    window_name="${dataset}_${arch}"
    log_file="logs/tmux/${window_name}_test.log"
    cmd="CUDA_VISIBLE_DEVICES=${gpu} python test.py --arch ${arch} --dataset ${dataset} --ckpt_path ${ckpt} 2>&1 | tee ${log_file}"
  else
    test_id="$maybe_test_id"
    window_name="${dataset}_${test_id}_${arch}"
    log_file="logs/tmux/${window_name}_test.log"
    cmd="CUDA_VISIBLE_DEVICES=${gpu} python test.py --arch ${arch} --dataset ${dataset} --test_id ${test_id} --ckpt_path ${ckpt} 2>&1 | tee ${log_file}"
  fi

  tmux new-window -t "$SESSION_NAME" -n "$window_name"
  tmux send-keys -t "${SESSION_NAME}:${window_name}" "$cmd" C-m

  echo "[test] ${window_name} -> GPU ${gpu}"

  job_idx=$((job_idx + 1))

  # ⏱️ delay to avoid spikes
  sleep "$DELAY"
done

tmux kill-window -t "${SESSION_NAME}:launcher"

echo "Started tmux session: $SESSION_NAME"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "List windows: tmux list-windows -t $SESSION_NAME"