#!/usr/bin/env bash
set -euo pipefail

############################################
# 公共参数
############################################
DATA_DIR="/j-jepa-vol/l1-jet-id/data/jetid/processed"
DATASET="hls4ml"
SORT_BY="kt"

############################################
# 仅列出 save_dir（末尾保留 / 可省心）
############################################
save_dirs=(
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/2/150/kt/trial-5/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/2/150/kt/trial-7/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/2/150/kt/trial-8/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/1/150/kt/trial-5/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/1/150/kt/trial-7/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/1/150/kt/trial-8/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/8/150/kt/trial-6/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/8/150/kt/trial-7/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/8/150/kt/trial-8/"
    "/j-jepa-vol/linformer4HEP/runs/num_partitions/16/150/kt/trial-6/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/16/150/kt/trial-7/"
  "/j-jepa-vol/linformer4HEP/runs/num_partitions/16/150/kt/trial-8/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_all/150/kt/trial-0/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_all/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_all/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_234/150/kt/trial-0/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_234/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_234/150/kt/trial-2/"
)

############################################
# 主循环
############################################
for SAVE_DIR in "${save_dirs[@]}"; do
  # 去掉可能多余的尾部斜杠再拼接模型文件名
  TEST_MODEL="${SAVE_DIR%/}/best.weights.h5"

  echo "Running test with:"
  echo "  save_dir   = $SAVE_DIR"
  echo "  test_model = $TEST_MODEL"
  echo "----------------------------------------"

  python test.py \
    --data_dir   "$DATA_DIR" \
    --dataset    "$DATASET" \
    --sort_by    "$SORT_BY" \
    --save_dir   "$SAVE_DIR" \
    --test_model "$TEST_MODEL"

  echo "Finished run for $SAVE_DIR"
  echo
done
