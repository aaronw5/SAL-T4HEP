#!/usr/bin/env bash
set -euo pipefail

# kt
python test_transolver.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/transolver/150/25x6/kt/trial-0/ \
  --sort_by kt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \

python test_transolver.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/transolver/150/25x6/kt/trial-1/ \
  --sort_by kt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \

python test_transolver.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/transolver/150/25x6/kt/trial-2/ \
  --sort_by kt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \

# pt
python test_transolver.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/transolver/150/25x6/pt/trial-0/ \
  --sort_by pt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \

python test_transolver.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/transolver/150/25x6/pt/trial-1/ \
  --sort_by pt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \

python test_transolver.py \
  --save_dir /j-jepa-vol/linformer4HEP/runs/transolver/150/25x6/pt/trial-2/ \
  --sort_by pt \
  --data_dir /j-jepa-vol/l1-jet-id/data/jetid/processed \
  --dataset hls4ml \