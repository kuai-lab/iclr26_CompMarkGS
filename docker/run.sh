#!/bin/bash
# Run the CompMarkGS 48bit container.
#   bash docker/run.sh check                     # are the CUDA extensions working on these GPUs?
#   bash docker/run.sh infer                     # evaluate the included checkpoints (no training)
#   bash docker/run.sh train 57494               # train + evaluate all 25 scenes with message seed 57494
#   bash docker/run.sh train 1234 5678           # several message seeds, one full set after another
#   bash docker/run.sh shell                     # interactive shell
# Env: GPU_IDS=0,1,2,3 (host GPUs, default all)  ONLY=nerf_synthetic/lego,llff/fern  MAX_PER_GPU=1
#      DATA=./dataset  OUT=./outputs  IMAGE=compmarkgs:48bit
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-compmarkgs:48bit}; DATA=${DATA:-$HERE/dataset}; OUT=${OUT:-$HERE/outputs}
mkdir -p "$OUT"
GPUS=${GPU_IDS:+\"device=$GPU_IDS\"}; GPUS=${GPUS:-all}
cmd=$1; shift || true
case "$cmd" in
  check) C=(python scripts/check_gpu.py) ;;
  infer) C=(bash scripts/infer_checkpoints.sh) ;;
  train) [ $# -ge 1 ] || { echo "usage: bash docker/run.sh train MSG_SEED [MSG_SEED ...]"; exit 1; }; C=(bash scripts/train.sh "$@") ;;
  shell) C=(bash) ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
IT=; [ "$cmd" = shell ] && IT=-it
docker run --rm $IT --gpus "$GPUS" --shm-size 16g --user "$(id -u):$(id -g)" \
  -v "$(readlink -f "$DATA")":/workspace/dataset:ro -v "$(readlink -f "$OUT")":/workspace/outputs \
  -e ONLY -e MAX_PER_GPU "$IMAGE" "${C[@]}"
