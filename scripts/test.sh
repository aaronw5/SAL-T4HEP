#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/j-jepa-vol/l1-jet-id/data/jetid/processed"
DATASET="hls4ml"
SORT_BY="kt"

save_dirs=(
  # ======  num_partitions  ======
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

  # ====== shuffle ======
  "/j-jepa-vol/linformer4HEP/runs/shuffle_all/150/kt/trial-0/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_all/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_all/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_234/150/kt/trial-0/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_234/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_234/150/kt/trial-2/"

  # ====== conv_filter （trial-1,2,3） ======
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_3_5_7/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_3_5_7/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_3_5_7/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_3_5_7_9/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_3_5_7_9/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_3_5_7_9/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_5_7/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_5_7/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/1_5_7/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_3_3/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_3_3/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_3_3/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_5_7/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_5_7/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_5_7/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_5_7_9/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_5_7_9/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/3_5_7_9/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/5_5_5/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/5_5_5/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/5_5_5/150/kt/trial-3/"

  "/j-jepa-vol/linformer4HEP/runs/conv_filter/7_7_7/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/7_7_7/150/kt/trial-2/"
  "/j-jepa-vol/linformer4HEP/runs/conv_filter/7_7_7/150/kt/trial-3/"
)


for SAVE_DIR in "${save_dirs[@]}"; do
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
