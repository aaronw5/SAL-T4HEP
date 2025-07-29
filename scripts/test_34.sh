#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/j-jepa-vol/l1-jet-id/data/jetid/processed"
DATASET="hls4ml"
SORT_BY="kt"

save_dirs=(
  # ====== shuffle ======
  "/j-jepa-vol/linformer4HEP/runs/shuffle_34/150/kt/trial-0/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_34/150/kt/trial-1/"
  "/j-jepa-vol/linformer4HEP/runs/shuffle_34/150/kt/trial-2/"
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
