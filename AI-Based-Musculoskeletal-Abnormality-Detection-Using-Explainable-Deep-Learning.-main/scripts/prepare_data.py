"""Check / prepare the MURA data folder using the manifest.

    python scripts/prepare_data.py                       # report only
    python scripts/prepare_data.py --verify_md5          # also check every file's MD5 (slow, ~3.4 GB)
    python scripts/prepare_data.py --data_root /content/data/MURA-v1.1

* Writes train_image_paths.csv / valid_image_paths.csv into the data folder if they are missing.
* Reports class counts and how many manifest images are present on disk.
* Exits with code 1 if images are missing/corrupt, so you notice before training.
"""
import argparse
import base64
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from src import config
from src.manifest import IMG_EXT, load_manifest


def md5_b64(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root")
    ap.add_argument("--manifest")
    ap.add_argument("--verify_md5", action="store_true")
    a = ap.parse_args()
    root = Path(a.data_root or config.DATA_ROOT)
    if a.manifest:
        config.MANIFEST = Path(a.manifest)
    man = load_manifest(root)
    man = man[man.file_name.str.lower().str.endswith(IMG_EXT)].copy()
    man["split"] = man.file_name.str.split("/").str[0]
    man["study"] = man.file_name.str.split("/").str[3]
    man["label"] = man.study.str.endswith("positive").astype(int)
    man["body_part"] = man.file_name.str.split("/").str[1].str.replace("XR_", "")

    root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "valid"]:
        csv = root / f"{split}_image_paths.csv"
        if not csv.exists():
            paths = sorted("MURA-v1.1/" + man[man.split == split].file_name)
            csv.write_text("\n".join(paths) + "\n")
            print(f"wrote {csv} ({len(paths)} paths)")

    print("\nManifest summary")
    for split, g in man.groupby("split"):
        print(f"  {split:<5} images={len(g):>6} studies={g.file_name.str.rsplit('/', n=1).str[0].nunique():>5} "
              f"patients={g.file_name.str.split('/').str[2].nunique():>5} abnormal={g.label.mean():.1%}")
        print("        " + ", ".join(f"{k}:{v}" for k, v in g.body_part.value_counts().items()))

    man["exists"] = [(root / f).exists() for f in man.file_name]
    miss = man[~man.exists]
    print(f"\nImages on disk: {int(man.exists.sum())}/{len(man)} found under {root}")
    bad = pd.DataFrame()
    if a.verify_md5 and man.exists.any():
        present = man[man.exists]
        ok = [md5_b64(root / f) == h for f, h in zip(present.file_name, present.md5_hash)]
        bad = present[[not x for x in ok]]
        print(f"MD5 check: {len(present) - len(bad)} OK, {len(bad)} corrupt/mismatched")
    if len(miss):
        print(f"\nMISSING {len(miss)} images (first: {miss.file_name.iloc[0]}).")
        print("The manifest lists files only - the X-ray PNGs must come from the official MURA download.")
    if len(miss) or len(bad):
        sys.exit(1)
    print("\nData folder is complete. Next: python -m src.train --model efficientnet --run_name efficientnet")


if __name__ == "__main__":
    main()
