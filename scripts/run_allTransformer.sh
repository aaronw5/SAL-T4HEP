#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: $0 \
    --data_dir PATH \
    --save_dir PATH \
     [--convolution] \
    [--conv_filter_heights H1,H2,...] [--vertical_stride N] \
    --batch_size N \
    --num_epochs N \
    --d_model N \
    --d_ff N \
    --num_heads N \
    --num_particles N \
    --sort_by pt|eta|phi|delta_R|kt|cluster"
  exit 1
}

# Default values
CLUSTER_E_FLAG=""
CLUSTER_F_FLAG=""
SHARE_EF_FLAG=""
CONVOLUTION_FLAG=""
CONV_FILTER_HEIGHTS_FLAG=""
VERTICAL_STRIDE_FLAG=""

# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir) DATA_DIR="$2"; shift 2;;
    --save_dir) SAVE_DIR="$2"; shift 2;;

    --convolution) CONVOLUTION_FLAG="--convolution"; shift;;
    --conv_filter_heights) CONV_FILTER_HEIGHTS_FLAG="--conv_filter_heights $2"; shift 2;;
    --vertical_stride) VERTICAL_STRIDE_FLAG="--vertical_stride $2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --num_epochs) NUM_EPOCHS="$2"; shift 2;;
    --d_model) D_MODEL="$2"; shift 2;;
    --d_ff) D_FF="$2"; shift 2;;
    --num_heads) NUM_HEADS="$2"; shift 2;;
    --num_particles) NP_LIST="$2"; shift 2;;
    --sort_by) SORT_MODES="$2"; shift 2;;
    *) echo "Unknown argument: $1"; usage;;
  esac
done

if [ -z "$DATA_DIR" ] || [ -z "$SAVE_DIR" ] || [ -z "$BATCH_SIZE" ] || \
   [ -z "$NUM_EPOCHS" ] || [ -z "$D_MODEL" ] || [ -z "$D_FF" ] || [ -z "$NUM_HEADS" ]; then
  echo "Missing required arguments."
  usage
fi


# Loop over particles & sort
# Allow list of particles and sorts via comma-separated
IFS=',' read -ra PARTICLES <<< "${NP_LIST:-16,32,150}"
IFS=',' read -ra SORTS <<< "${SORT_MODES:-pt,delta_R,kt}"
for NP in "${PARTICLES[@]}"; do
  for SORT in "${SORTS[@]}"; do
    echo "=========================================="
    echo "Running with num_particles=$NP, sort_by=$SORT"
    echo "=========================================="
    ./train_transformer.py \
      --data_dir             "${DATA_DIR}" \
      --save_dir             "${SAVE_DIR}" \
      ${CLUSTER_E_FLAG} \
      ${CLUSTER_F_FLAG} \
      ${SHARE_EF_FLAG} \
      ${CONVOLUTION_FLAG} \
      ${CONV_FILTER_HEIGHTS_FLAG} \
      ${VERTICAL_STRIDE_FLAG} \
      --batch_size           "${BATCH_SIZE}" \
      --num_epochs           "${NUM_EPOCHS}" \
      --d_model              "${D_MODEL}" \
      --d_ff                 "${D_FF}" \
      --num_heads            "${NUM_HEADS}" \
      --num_particles        "${NP}" \
      --sort_by              "${SORT}"
  done
done
