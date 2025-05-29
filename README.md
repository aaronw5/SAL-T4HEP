# SAL-T: Spatially Aware Linear Transformer for Jet Tagging

This repository contains code and scripts to preprocess jet data and train efficient transformer-based models for jet tagging on various datasets.

---

## 🛠️ Setup

### 1. Clone the helper Repository

```bash
git clone https://github.com/JavierZhao/l1-jet-id-hls4ml
cd l1-jet-id-hls4ml
```

### 2. Install Dependencies
```bash
pip install -e .
```

## 📦 Dataset Preparation
We use the [hls4ml dataset](https://zenodo.org/records/3602260) for training. To download and process the data:
```bash
python l1-jet-id-hls4ml/fast_jetclass/data/prepare_hls4ml_data.py \
  --root PATH-TO-DATA-DIR \
  --nconst [16|32|150] \
  --feats ptetaphi \
  --norm standard \
  --seed 42 \
  --kfolds 5
```

## 🚀 Running Training
To train our Linformer-based jet classifier:
```bash
chmod +x linformer4HEP/scripts/run_all.sh

./linformer4HEP/scripts/run_all.sh \
  --data_dir /j-jepa-vol/linformer_data/TopTagging/200 \
  --dataset top \
  --save_dir /j-jepa-vol/linformer4HEP/runs/top/200/1layer/conv/cluster_both \
  --cluster_E \
  --cluster_F \
  --convolution \
  --batch_size 4096 \
  --d_model 16 \
  --d_ff 16 \
  --num_heads 4 \
  --proj_dim 4 \
  --num_particles 200 \
  --sort_by [kt|deltaR|pt]
```
**Arguments:**

- `--data_dir`: Path to preprocessed data  
- `--dataset`: Dataset type (`top`, `hls4ml`, `QG`)  
- `--save_dir`: Output directory for logs and checkpoints  
- `--cluster_E`, `--cluster_F`: Enable spatial partitioning on keys/values  
- `--convolution`: Enable convolution layer in attention
- `--batch_size`: Training batch size  
- `--d_model`, `--d_ff`, `--num_heads`, `--proj_dim`: Model hyperparameters  
- `--num_particles`: Number of input particles per jet  
- `--sort_by`: Sorting strategy (`kt`, `pt`, `deltaR`.)  
