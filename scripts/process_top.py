# -*- coding: utf-8 -*-
"""
process_h5_to_npy.py

Reads HDF5 files from Top Tagging, computes relative eta, relative phi, and normalized pT
for each particle, pads each jet to `maxlen` constituents, and writes out NumPy (.npy) files
for features and labels.
Expects optionally `train.h5`, `val.h5`, and `test.h5` in the input directory.
Supports subsampling a fraction of events.
"""
import logging
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import awkward as ak
import vector

# register vector with awkward for .eta, .phi, .pt
vector.register_awkward()


def pad_to_fixed_length(
    arr: np.ndarray, maxlen: int, pad_value: float = 0.0
) -> np.ndarray:
    """
    Pads or truncates a 2D array arr of shape (n_particles, n_features)
    to (maxlen, n_features).
    """
    n, f = arr.shape
    if n >= maxlen:
        return arr[:maxlen]
    padded = np.full((maxlen, f), pad_value, dtype=arr.dtype)
    padded[:n] = arr
    return padded


def build_features_and_labels(h5path: Path, maxlen: int = 150, frac: float = 1.0):
    """
    Reads one HDF5 file and computes:
      - particle-level features: (Δη, Δφ, pT/jet_pT)
      - optional labels: `is_signal_new`
    Can subsample a fraction `frac` of jets (0 < frac <= 1).

    Returns:
      pf_features: np.ndarray, shape (n_jets, maxlen, 3)
      labels:      np.ndarray of shape (n_jets,) or None
    """
    # load raw table
    df = pd.read_hdf(str(h5path), key="table")
    # subsample if requested
    if frac < 1.0:
        original = len(df)
        df = df.sample(frac=frac, random_state=42)
        df = df.reset_index(drop=True)
        logging.info(f"Subsampled {original} → {len(df)} jets ({frac*100:.1f}%)")
    n_jets = len(df)

    # detect particle columns dynamically
    px_cols = sorted([c for c in df.columns if c.startswith("PX_")])
    max_particles = len(px_cols)
    py_cols = [c.replace("PX_", "PY_") for c in px_cols]
    pz_cols = [c.replace("PX_", "PZ_") for c in px_cols]
    e_cols = [c.replace("PX_", "E_") for c in px_cols]

    # extract raw arrays
    px = df[px_cols].values
    py = df[py_cols].values
    pz = df[pz_cols].values
    E = df[e_cols].values

    # mask zero-energy slots (padding)
    mask = E > 0
    n_particles = np.sum(mask, axis=1)

    # build awkward 4-vectors for masked entries
    px_ak = ak.unflatten(px[mask], n_particles)
    py_ak = ak.unflatten(py[mask], n_particles)
    pz_ak = ak.unflatten(pz[mask], n_particles)
    E_ak = ak.unflatten(E[mask], n_particles)

    p4 = ak.zip(
        {"px": px_ak, "py": py_ak, "pz": pz_ak, "energy": E_ak},
        with_name="Momentum4D",
    )
    jet_p4 = ak.sum(p4, axis=-1)

    # compute per-particle features
    deta = p4.eta - jet_p4.eta  # (n_jets, n_particles)
    # wrap phi difference to [-π,π]
    dphi = (p4.phi - jet_p4.phi + np.pi) % (2 * np.pi) - np.pi
    pt_norm = p4.pt / jet_p4.pt

    # stack and pad per event
    features = []
    for ev_deta, ev_dphi, ev_pt in zip(deta, dphi, pt_norm):
        arr = np.stack(
            [ev_deta.to_numpy(), ev_dphi.to_numpy(), ev_pt.to_numpy()], axis=1
        )
        features.append(pad_to_fixed_length(arr, maxlen))

    pf_features = np.stack(features, axis=0)

    # optional labels
    labels = None
    if "is_signal_new" in df.columns:
        labels = df["is_signal_new"].astype(int).values

    return pf_features, labels


def main():
    parser = argparse.ArgumentParser(
        description="Convert Top-Tagging .h5 → .npy particle-level arrays"
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

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # process the requested splits
    for split in ["train", "val", "test"]:
        h5file = in_dir / f"{split}.h5"
        if not h5file.exists():
            logger.warning(f"Missing {h5file}, skipping")
            continue

        feats, labs = build_features_and_labels(
            h5file, maxlen=args.maxlen, frac=args.frac
        )
        out_f = out_dir / split / "features.npy"
        np.save(out_f, feats)
        logger.info(f"Saved features: {out_f} shape={feats.shape}")

        if labs is not None:
            out_l = out_dir / split / "labels.npy"
            np.save(out_l, labs)
            logger.info(f"Saved labels:   {out_l} shape={labs.shape}")


if __name__ == "__main__":
    main()
