"""Create a tiny SYNTHETIC dataset with MURA's folder layout - for smoke-testing the code only.

    python scripts/make_dummy_data.py --out data/dummy/MURA-v1.1
    MURA_ROOT=data/dummy/MURA-v1.1 python -m src.train --model baseline --weights none --epochs 2

The images are random noise with a bright blob in "abnormal" studies. Any metric computed on
this data says NOTHING about real-world performance.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

PARTS = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]


def make_img(abnormal, rng):
    h, w = int(rng.integers(180, 300)), int(rng.integers(180, 300))
    img = rng.normal(60, 20, (h, w)).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (w // 3, 10), (2 * w // 3, h - 10), 140, -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    if abnormal:
        cv2.circle(img, (int(rng.integers(w // 3, 2 * w // 3)), int(rng.integers(30, h - 30))), 14, 235, -1)
    return img


def build(root, split, n_patients, start_id, rng):
    rels = []
    for k in range(n_patients):
        pid, part = start_id + k, PARTS[int(rng.integers(len(PARTS)))]
        for s in range(1, int(rng.integers(1, 3)) + 1):
            abnormal = bool(rng.random() < 0.45)
            for i in range(1, int(rng.integers(2, 4)) + 1):
                rel = Path(split) / f"XR_{part}" / f"patient{pid:05d}" / f"study{s}_{'positive' if abnormal else 'negative'}" / f"image{i}.png"
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(root / rel), make_img(abnormal, rng))
                rels.append(f"MURA-v1.1/{rel.as_posix()}")
    (root / f"{split}_image_paths.csv").write_text("\n".join(rels) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dummy/MURA-v1.1")
    ap.add_argument("--train_patients", type=int, default=80)
    ap.add_argument("--valid_patients", type=int, default=24)
    a = ap.parse_args()
    rng, root = np.random.default_rng(0), Path(a.out)
    build(root, "train", a.train_patients, 1, rng)
    build(root, "valid", a.valid_patients, 20000, rng)
    print(f"Synthetic data written to {root}  (export MURA_ROOT={root})")
