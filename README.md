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
To download and process the [hls4ml dataset](https://zenodo.org/records/3602260) :
```bash
python l1-jet-id-hls4ml/fast_jetclass/data/prepare_hls4ml_data.py \
  --root PATH-TO-DATA-DIR \
  --nconst [16|32|150] \
  --feats ptetaphi \
  --norm standard \
  --seed 42 \
  --kfolds 5
```

To download and process the [Top Quark Tagging Reference Dataset](https://zenodo.org/records/2603256):
```bash
python SAL-T4HEP/scripts/python process_top.py \
--input_dir PATH-TO-STORE-RAW-DATA \
--output_dir PATH-TO-STORE-PROCESSED-DATA \
```

To download and process the [Quark Gluon Dataset](https://zenodo.org/records/3164691):
```bash
python SAL-T4HEP/scripts/python process_qg.py \
--input_dir PATH-TO-STORE-RAW-DATA \
--output_dir PATH-TO-STORE-PROCESSED-DATA \
```
At this point, for the purpose of running our code, you no longer need the raw data, so you can safely delete that directory.

## 🚀 Running Training
To train our Linformer-based jet classifier:
```bash
chmod +x SAL-T4HEP/scripts/run_all.sh

./SAL-T4HEP/scripts/run_all.sh \
  --data_dir PATH-TO-DATA \
  --dataset [top|QG|hls4ml] \
  --save_dir PATH-TO-SAVE-RESULTS \
  --cluster_E \
  --cluster_F \
  --convolution \
  --batch_size 4096 \
  --d_model 16 \
  --d_ff 16 \
  --num_heads 4 \
  --proj_dim 4 \
  --num_particles [150|200] \
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
