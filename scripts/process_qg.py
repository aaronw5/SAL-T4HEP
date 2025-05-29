import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from glob import glob
import logging
from pathlib import Path
import argparse
import os
import numpy as np
import pandas as pd
import awkward as ak
import vector
import shutil
import subprocess
from dataset_utils import get_file, extract_archive

datasets = {
    'QuarkGluon': {
        # converted from https://zenodo.org/record/3164691
        '../': [
            ('https://hqu.web.cern.ch/datasets/QuarkGluon/QuarkGluon.tar', 'd8dd7f71a7aaaf9f1d2ee3cddef998f9'),
        ],
    },
}


def download_dataset(basedir, force_download=False):
    dataset = 'QuarkGluon'
    info = datasets[dataset]
    datadir = os.path.join(basedir, dataset)
    if force_download:
        if os.path.exists(datadir):
            print(f'Removing existing dir {datadir}')
            shutil.rmtree(datadir)
    for subdir, flist in info.items():
        for url, md5 in flist:
            fpath, download = get_file(url, datadir=datadir, file_hash=md5, force_download=force_download)
            if download:
                extract_archive(fpath, path=os.path.join(datadir, subdir))

def main():
    parser = argparse.ArgumentParser(
        description="Convert Quark Gluon dataset → .npy particle-level arrays"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing train.h5, val.h5, test.h5",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save feature .npy files",
    )
    parser.add_argument(
        "--maxlen", type=int, default=150, help="Max particles per jet (padding)"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=1.0,
        help="Fraction of jets to process (1.0 = all, <1 for subsample)",
    )
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    # download_dataset(args.input_dir)
    for tag in ['train', 'test']:
        input_dir = args.input_dir + "/QuarkGluon"
        output_dir = args.output_dir + f"/{tag}"
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
        # Load all .parquet files recursively
        if tag == 'train':
            logger.info(f"Processing training and validation data")
            all_files = glob(os.path.join(input_dir, '**', 'train_file_*.parquet'), recursive=True)
        else:
            logger.info(f"Processing testing data")
            all_files = glob(os.path.join(input_dir, '**', 'test_file_*.parquet'), recursive=True)

        features_list = []
        labels_list = []
        
        for i, file in enumerate(all_files):
            print(i)
            df = pd.read_parquet(file, engine='pyarrow')
        
            for idx, row in df.iterrows():
                # Extract per-particle features stored as lists
                px   = np.array(row['part_px'], dtype=np.float32)
                py   = np.array(row['part_py'], dtype=np.float32)
                deta = np.array(row['part_deta'], dtype=np.float32)
                dphi = np.array(row['part_dphi'], dtype=np.float32)
        
                # Skip if arrays are inconsistent in length
                if not (len(px) == len(py) == len(deta) == len(dphi)):
                    continue
        
                part_pT = np.sqrt(px**2 + py**2)
        
                # NEW ORDER: [pT, deta, dphi]
                part_features = np.stack([part_pT, deta, dphi], axis=1)  # shape: (N, 3)
                features_list.append(part_features)
        
                # Append label
                labels_list.append(row['label'])
        
        # Fixed max length
        max_particles = 150
        
        # Truncate or pad each jet
        features_padded = np.stack([
            np.pad(jet[:max_particles], ((0, max_particles - min(len(jet), max_particles)), (0, 0)), mode='constant')
            for jet in features_list
        ], axis=0).astype(np.float32)
        
        labels_array = np.array(labels_list)
        
        # Shuffle
        indices = np.random.permutation(len(features_padded))
        features_shuffled = features_padded[indices]
        labels_shuffled = labels_array[indices]
        
        # Normalize pT (index 0), using mask on pT != 0
        mask = features_shuffled[:, :, 0] != 0
        mean = 7.0
        std = 5.56
        features_shuffled[:, :, 0] = np.where(
            mask,
            (features_shuffled[:, :, 0] - mean) / std,
            features_shuffled[:, :, 0]
        )
        
        # Save
        if tag == 'test':
            np.save(os.path.join(output_dir, 'features.npy'), features_shuffled)
            np.save(os.path.join(output_dir, 'labels.npy'), labels_shuffled)
            print(f"Saved features.npy with shape {features_shuffled.shape}")
            print(f"Saved labels.npy with shape {labels_shuffled.shape}")
        else:
            feats = features_shuffled
            labs = labels_shuffled
            # Split 80% train / 20% val
            num_total = len(feats)
            num_val = int(0.2 * num_total)
            num_train = num_total - num_val
            
            features_train = feats[:num_train]
            labels_train = labs[:num_train]
            
            features_val = feats[num_train:]
            labels_val = labs[num_train:]
            
            # Save to disk
            train_out_dir = args.output_dir + f"/train"
            val_out_dir = args.output_dir + f"/val"
            os.makedirs(train_out_dir, exist_ok=True)
            os.makedirs(val_out_dir, exist_ok=True)
            np.save(os.path.join(train_out_dir, 'features.npy'), features_train)
            np.save(os.path.join(train_out_dir, 'labels.npy'), labels_train)
            print(f"Saved features.npy with shape {features_train.shape}")
            print(f"Saved labels.npy with shape {labels_train.shape}")
            np.save(os.path.join(val_out_dir, 'features.npy'), features_val)
            np.save(os.path.join(val_out_dir, 'labels.npy'), labels_val)
            print(f"Saved features.npy with shape {features_val.shape}")
            print(f"Saved labels.npy with shape {labels_val.shape}")
            
if __name__ == "__main__":
    main()

        