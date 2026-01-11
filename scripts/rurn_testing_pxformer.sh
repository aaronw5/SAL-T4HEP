#!/usr/bin/env bash
set -euo pipefail
python test_point_transformer.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/point_transformer_grid_0p4/matched/150/kt/trial-3/ \
  --sort_by kt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \
  --model_size matched \
  --serialize_by kt \
  --grid_size 0.4

python test_point_transformer.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/point_transformer_grid_0p4/matched/150/kt/trial-4/ \
  --sort_by kt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \
  --model_size matched \
  --serialize_by kt \
  --grid_size 0.4

# pt
# python test_point_transformer.py \
#   --save_dir /j-jepa-vol/linformer4HEP/runs/point_transformer_grid_0p4/matched/150/pt/trial-0/ \
#   --sort_by pt \
#   --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
#   --dataset hls4ml \
#   --model_size matched \
#   --grid_size 0.4

# python test_point_transformer.py \
#   --save_dir /j-jepa-vol/linformer4HEP/runs/point_transformer_grid_0p4/matched/150/pt/trial-1/ \
#   --sort_by pt \
#   --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
#   --dataset hls4ml \
#   --model_size matched \
#   --grid_size 0.4 

# python test_point_transformer.py \
#   --save_dir /j-jepa-vol/linformer4HEP/runs/point_transformer_grid_0p4/matched/150/pt/trial-2/ \
#   --sort_by pt \
#   --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
#   --dataset hls4ml \
#   --model_size matched \
#   --grid_size 0.4

