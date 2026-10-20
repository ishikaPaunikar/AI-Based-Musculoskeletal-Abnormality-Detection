"""Data loading, patient-level splitting, preprocessing and tf.data pipelines.

Leakage control
---------------
* MURA's official `train` split is divided into train/validation by PATIENT
  (GroupShuffleSplit), so no patient (and therefore no study/image) appears in both.
* MURA's official `valid` split is patient-disjoint from `train` by construction and is
  used here as the held-out TEST set (the official test set is not public).
  It is never used for early stopping, threshold selection or hyper-parameter search.

Pixel scaling
-------------
The pipeline outputs float32 images in [0, 255] with 3 identical channels.
Normalisation happens INSIDE each model (Rescaling(1/255) for the baseline, EfficientNet's
built-in preprocessing for EfficientNetB0), so the same inputs work for training,
evaluation, Grad-CAM and the web app.
"""
import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from tensorflow import keras

from . import config

AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------- dataframes
def load_split_dataframe(split, root=None):
    """Parse MURA's `<split>_image_paths.csv` into a dataframe with labels and IDs.

    Label rule (from MURA's folder naming): study folder ending in 'positive' = abnormal (1).
    """
    root = Path(root or config.DATA_ROOT)
    csv = root / f"{split}_image_paths.csv"
    if csv.exists():
        paths = pd.read_csv(csv, header=None)[0].tolist()
    else:  # fall back to the file manifest (data/mura_v1_1.csv)
        from .manifest import manifest_paths
        paths = manifest_paths(split, root)
    rows = []
    for p in paths:
        parts = Path(p).parts
        i = next(k for k, x in enumerate(parts) if x in ("train", "valid"))
        body, patient, study = parts[i + 1], parts[i + 2], parts[i + 3]
        rows.append(
            dict(
                path=str(root / Path(*parts[i:])),
                split=split,
                body_part=body.replace("XR_", "").title(),
                patient_id=f"{split}_{patient}",             # patient-level group key
                study_id=f"{split}_{body}_{patient}_{study}",
                label=int(study.endswith("positive")),
            )
        )
    df = pd.DataFrame(rows)
    missing = [p for p in df.path if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(df)} {split} images are missing under {root}, e.g. {missing[0]}. "
            "Run `python scripts/prepare_data.py` for a full report.")
    return df


def subsample_patients(df, frac, seed):
    """Keep a random fraction of PATIENTS (never individual images)."""
    if frac >= 1.0:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(seed)
    pats = df.patient_id.unique()
    keep = rng.choice(pats, size=max(2, int(len(pats) * frac)), replace=False)
    return df[df.patient_id.isin(keep)].reset_index(drop=True)


def make_splits(root=None, val_fraction=None, seed=None, subset=1.0, load_test=True):
    """Return (train_df, val_df, test_df). Test = official MURA `valid` split."""
    val_fraction = config.VAL_FRACTION if val_fraction is None else val_fraction
    seed = config.SEED if seed is None else seed
    full = load_split_dataframe("train", root)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    tr_idx, va_idx = next(gss.split(full, groups=full.patient_id))
    train, val = full.iloc[tr_idx].reset_index(drop=True), full.iloc[va_idx].reset_index(drop=True)
    overlap = set(train.patient_id) & set(val.patient_id)
    assert not overlap, f"Patient leakage between train and val: {len(overlap)} patients"
    train, val = subsample_patients(train, subset, seed), subsample_patients(val, subset, seed)
    test = load_split_dataframe("valid", root) if load_test else None
    if test is not None and set(train.patient_id.str.split("_", n=1).str[1]) & set(
        test.patient_id.str.split("_", n=1).str[1]
    ):
        warnings.warn("Patient numbers overlap between official train and valid splits - check your dataset copy.")
    return train, val, test


def summarize_split(df, name):
    return (
        f"{name:<6} images={len(df):>6}  studies={df.study_id.nunique():>5}  "
        f"patients={df.patient_id.nunique():>5}  abnormal={df.label.mean():.1%}"
    )


def class_weights(df):
    n, n1 = len(df), int(df.label.sum())
    n0 = n - n1
    return {0: n / (2.0 * max(n0, 1)), 1: n / (2.0 * max(n1, 1))}


# ----------------------------------------------------------------------------- preprocessing
def preprocess_array(gray, size=config.IMG_SIZE, clahe=False):
    """uint8 grayscale (H, W) -> float32 (size, size, 3) in [0, 255].

    Steps: optional CLAHE contrast enhancement (OpenCV) -> aspect-preserving resize ->
    black padding to a square -> replicate to 3 channels (needed by ImageNet weights).
    """
    if gray.ndim == 3:
        gray = gray[..., 0]
    gray = gray.astype(np.uint8)
    if clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return np.repeat(canvas[..., None], 3, axis=2).astype(np.float32)


def load_image(path, size=config.IMG_SIZE, clahe=False):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise IOError(f"Could not read image: {path}")
    return preprocess_array(gray, size, clahe)


def build_augmentation():
    """Light, anatomically plausible augmentation (training only)."""
    L = keras.layers
    return keras.Sequential(
        [
            L.RandomFlip("horizontal"),
            L.RandomRotation(0.04, fill_mode="constant", fill_value=0.0),
            L.RandomZoom(0.1, fill_mode="constant", fill_value=0.0),
            L.RandomTranslation(0.05, 0.05, fill_mode="constant", fill_value=0.0),
            L.RandomBrightness(0.1, value_range=(0, 255)),
            L.RandomContrast(0.15),
        ],
        name="augmentation",
    )


def _tf_load(path, label, size, clahe):
    def _load(p):
        p = p.decode("utf-8") if isinstance(p, bytes) else str(p)
        return load_image(p, size, clahe)

    img = tf.numpy_function(_load, [path], tf.float32)
    img.set_shape((size, size, 3))
    return img, tf.cast(label, tf.float32)


def build_dataset(df, batch_size=32, training=False, size=config.IMG_SIZE, clahe=False, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("float32")))
    if training:
        ds = ds.shuffle(min(len(df), 5000), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: _tf_load(p, l, size, clahe), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if augment:
        aug = build_augmentation()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
