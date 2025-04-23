#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: $0 \
    --data_dir PATH \
    --save_dir PATH \
    [--cluster_E] [--cluster_F] [--convolution] \
    [--conv_filter_heights H1,H2,...] [--vertical_stride N] \
    --batch_size N \
    --num_epochs N \
    --d_model N \
    --d_ff N \
    --num_heads N \
    --proj_dim N \
    --"
  exit 1
}

# Default flags
CLUSTER_E_FLAG=""
CLUSTER_F_FLAG=""
CONVOLUTION_FLAG=""
CONV_FILTER_HEIGHTS_FLAG=""
VERTICAL_STRIDE_FLAG=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)             DATA_DIR="$2"; shift 2;;
    --save_dir)             SAVE_DIR="$2"; shift 2;;
    --cluster_E)            CLUSTER_E_FLAG="--cluster_E"; shift;;
    --cluster_F)            CLUSTER_F_FLAG="--cluster_F"; shift;;
    --convolution)          CONVOLUTION_FLAG="--convolution"; shift;;
    --conv_filter_heights)  CONV_FILTER_HEIGHTS_FLAG="--conv_filter_heights $2"; shift 2;;
    --vertical_stride)      VERTICAL_STRIDE_FLAG="--vertical_stride $2"; shift 2;;
    --batch_size)           BATCH_SIZE="$2"; shift 2;;
    --num_epochs)           NUM_EPOCHS="$2"; shift 2;;
    --d_model)              D_MODEL="$2"; shift 2;;
    --d_ff)                 D_FF="$2"; shift 2;;
    --num_heads)            NUM_HEADS="$2"; shift 2;;
    --proj_dim)             PROJ_DIM="$2"; shift 2;;
    *) echo "Unknown argument: $1"; usage;;
  esac
done

# Validate required
if [[ -z "$DATA_DIR" || -z "$SAVE_DIR" || -z "$BATCH_SIZE" || -z "$NUM_EPOCHS" \
      || -z "$D_MODEL" || -z "$D_FF" || -z "$NUM_HEADS" || -z "$PROJ_DIM" ]]; then
  echo "Missing required arguments."; usage
fi

# Loop over particles, sort modes, and repeat 5 trials each
for NP in 16 32 150; do
  for SORT in cluster pt delta_R kt; do
      echo "=========================================="
      echo "num_particles=$NP, sort_by=$SORT"
      echo "=========================================="
      ./train_linformer.py \
        --data_dir             "${DATA_DIR}" \
        --save_dir             "${SAVE_DIR}" \
        ${CLUSTER_E_FLAG} \
        ${CLUSTER_F_FLAG} \
        ${CONVOLUTION_FLAG} \
        ${CONV_FILTER_HEIGHTS_FLAG} \
        ${VERTICAL_STRIDE_FLAG} \
        --batch_size           "${BATCH_SIZE}" \
        --num_epochs           "${NUM_EPOCHS}" \
        --d_model              "${D_MODEL}" \
        --d_ff                 "${D_FF}" \
        --num_heads            "${NUM_HEADS}" \
        --proj_dim             "${PROJ_DIM}" \
        --num_particles        "${NP}" \
        --sort_by              "${SORT}"
  done
done
