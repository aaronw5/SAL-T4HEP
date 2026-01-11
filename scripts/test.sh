#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="/j-jepa-vol/l1-jet-id/data/jetid/processed"
DATASET="hls4ml"
SORT_BY="pt"

save_dirs=(
  # ======  sal-t  ======
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/conv/cluster_both/150/kt/trial-0"
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/conv/cluster_both/150/kt/trial-1"
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/conv/cluster_both/150/kt/trial-2"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/conv/cluster_both/150/pt/trial-0"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/conv/cluster_both/150/pt/trial-1"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/conv/cluster_both/150/pt/trial-2"

  # ====== linformer ======
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/vanilla/150/kt/trial-0"
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/vanilla/150/kt/trial-1"
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/vanilla/150/kt/trial-2"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/vanilla/150/pt/trial-0"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/vanilla/150/pt/trial-1"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/vanilla/150/pt/trial-2"

  # ====== transformer ======
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/transformer_w_test/150/kt/trial-0"
  # "/j-jepa-vol/linformer4HEP/runs/double_batch_size/transformer_w_test/150/kt/trial-2"
  "/j-jepa-vol/linformer4HEP/runs/double_batch_size/transformer_w_test/150/pt/trial-0"
)


for SAVE_DIR in "${save_dirs[@]}"; do
  TEST_MODEL="${SAVE_DIR%/}/best.weights.h5"

  echo "Running test with:"
  echo "  save_dir   = $SAVE_DIR"
  echo "  test_model = $TEST_MODEL"
  echo "----------------------------------------"

  python test_bkd_rej.py \
    --data_dir   "$DATA_DIR" \
    --sort_by    "$SORT_BY" \
    --save_dir   "$SAVE_DIR" \
    --test_model "$TEST_MODEL"

  echo "Finished run for $SAVE_DIR"
  echo
done
