#!/usr/bin/env bash
set -e

usage() {
  cat <<EOF
Usage: $0 \
  --data_dir PATH \
  --save_dir PATH \
  --dataset hls4ml|top|jetclass|QG \
  [--batch_size N] \
  [--val_split FLOAT] \
  [--d_model N] \
  [--d_ff N] \
  [--num_heads N] \
  [--proj_dim N] \
  [--num_particles N1,N2,...] \
  [--sort_by pt,eta,phi,delta_R,kt,cluster] \
  [--cluster_E] [--cluster_F] [--share_EF] [--convolution] \
  [--conv_filter_heights N1 N2 ...] \
  [--num_layers N] \
  [--shuffle_all N]      ### NEW
  [--shuffle_234 N]      ### NEW
EOF
  exit 1
}

# Default flags
CLUSTER_E_FLAG=""
CLUSTER_F_FLAG=""
SHARE_EF_FLAG=""
CONV_FLAG=""
BATCH_SIZE_FLAG=""
VAL_SPLIT_FLAG=""
D_MODEL_FLAG=""
D_FF_FLAG=""
HEADS_FLAG=""
PROJ_DIM_FLAG=""
SHUFFLE_ALL_FLAG=""     ### NEW
SHUFFLE_234_FLAG=""     ### NEW
NP_LIST=""
SORT_MODES=""
CONV_FILTER_HEIGHTS=""
# Parse args
while [[ $# -gt 0 ]]; do
  case $1 in
    --data_dir)
      DATA_DIR="$2"; shift 2;;
    --save_dir)
      SAVE_DIR="$2"; shift 2;;
    --dataset)
      DATASET="$2"; shift 2;;
    --batch_size)
      BATCH_SIZE_FLAG="--batch_size $2"; shift 2;;
    --val_split)
      VAL_SPLIT_FLAG="--val_split $2"; shift 2;;
    --d_model)
      D_MODEL_FLAG="--d_model $2"; shift 2;;
    --d_ff)
      D_FF_FLAG="--d_ff $2"; shift 2;;
    --num_heads)
      HEADS_FLAG="--num_heads $2"; shift 2;;
    --proj_dim)
      PROJ_DIM_FLAG="--proj_dim $2"; shift 2;;
    --num_particles)
      NP_LIST="$2"; shift 2;;
    --sort_by)
      SORT_MODES="$2"; shift 2;;
    --cluster_E)
      CLUSTER_E_FLAG="--cluster_E"; shift;;
    --cluster_F)
      CLUSTER_F_FLAG="--cluster_F"; shift;;
    --share_EF)
      SHARE_EF_FLAG="--share_EF"; shift;;
    --convolution)
      CONV_FLAG="--convolution"; shift;;
    --conv_filter_heights)
      # Collect all subsequent non-flag tokens as filter heights; accept forms like "1 3 5" or "[1, 3, 5]"
      shift
      CONV_FILTER_HEIGHTS_FLAG="--conv_filter_heights"
      while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
        tok="$1"
        # strip commas and surrounding brackets
        tok="${tok//,/}"
        tok="${tok#\[}"
        tok="${tok%\]}"
        if [[ -n "$tok" ]]; then
          CONV_FILTER_HEIGHTS_FLAG+=" $tok"
        fi
        shift
      done
      ;;
    --num_layers)
      NUM_LAYERS_FLAG="--num_layers $2"; shift 2;;
    --shuffle_all)                         ### NEW
      SHUFFLE_ALL_FLAG="--shuffle_all $2"; shift 2;;   ### NEW
    --shuffle_234)                         ### NEW
      SHUFFLE_234_FLAG="--shuffle_234 $2"; shift 2;;   ### NEW
    *)
      echo "Unknown argument: $1"
      usage;;
  esac
done

# Validate required arguments
if [[ -z "$DATA_DIR" || -z "$SAVE_DIR" || -z "$DATASET" ]]; then
  echo "Missing required arguments."
  usage
fi

# Defaults for particle counts and sort modes
IFS=',' read -ra PARTICLES <<< "${NP_LIST:-16,32,150}"
IFS=',' read -ra SORTS     <<< "${SORT_MODES:-pt,delta_R,kt}"

# Loop over particle counts and sort modes
for NP in "${PARTICLES[@]}"; do
  for SORT in "${SORTS[@]}"; do
    echo "=========================================="
    echo "Running dataset=$DATASET, num_particles=$NP, sort_by=$SORT"
    echo "=========================================="
    args=(
      python3 train_linformer.py
      --data_dir "$DATA_DIR"
      --save_dir "$SAVE_DIR"
      --dataset "$DATASET"
      --num_particles "$NP"
      --sort_by "$SORT"
    )
    # Append optional key/value flags (stored as two+ tokens)
    [[ -n "$BATCH_SIZE_FLAG" ]] && args+=($BATCH_SIZE_FLAG)
    [[ -n "$VAL_SPLIT_FLAG"  ]] && args+=($VAL_SPLIT_FLAG)
    [[ -n "$D_MODEL_FLAG"    ]] && args+=($D_MODEL_FLAG)
    [[ -n "$D_FF_FLAG"       ]] && args+=($D_FF_FLAG)
    [[ -n "$HEADS_FLAG"      ]] && args+=($HEADS_FLAG)
    [[ -n "$PROJ_DIM_FLAG"   ]] && args+=($PROJ_DIM_FLAG)
    [[ -n "$NUM_LAYERS_FLAG" ]] && args+=($NUM_LAYERS_FLAG)
    [[ -n "$SHUFFLE_ALL_FLAG" ]] && args+=($SHUFFLE_ALL_FLAG)    ### NEW
    [[ -n "$SHUFFLE_234_FLAG" ]] && args+=($SHUFFLE_234_FLAG)    ### NEW
    # Append boolean flags (single token when set)
    [[ -n "$CLUSTER_E_FLAG"  ]] && args+=("$CLUSTER_E_FLAG")
    [[ -n "$CLUSTER_F_FLAG"  ]] && args+=("$CLUSTER_F_FLAG")
    [[ -n "$SHARE_EF_FLAG"   ]] && args+=("$SHARE_EF_FLAG")
    [[ -n "$CONV_FLAG"       ]] && args+=("$CONV_FLAG")
    # Append conv filter heights (multiple tokens)
    [[ -n "$CONV_FILTER_HEIGHTS_FLAG" ]] && args+=($CONV_FILTER_HEIGHTS_FLAG)

    "${args[@]}"
  done
done
