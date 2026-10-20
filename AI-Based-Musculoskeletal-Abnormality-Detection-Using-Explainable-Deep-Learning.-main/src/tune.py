"""Hyper-parameter optimisation by random search (scikit-learn ParameterSampler).

    python -m src.tune --model efficientnet --trials 12 --subset 0.25 --trial_epochs 5
    python -m src.tune --model baseline     --trials 12 --subset 0.25 --trial_epochs 8

Every trial trains on a patient-level subset and is scored by validation AUC on the
patient-held-out validation split. The official valid/test split is never touched.
Both models get the same search budget so the comparison is fair.
"""
import argparse
import gc
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import ParameterSampler
from tensorflow import keras

from . import config
from .train import run_training

SPACES = {
    "efficientnet": {
        "lr": loguniform(3e-4, 3e-3),
        "ft_lr": loguniform(1e-5, 1e-4),
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "batch_size": [16, 32],
        "unfreeze_layers": [20, 40, 80],
        "optimizer": ["adam", "adamw"],
    },
    "baseline": {
        "lr": loguniform(3e-4, 3e-3),
        "dropout": [0.2, 0.3, 0.4, 0.5],
        "batch_size": [16, 32],
        "optimizer": ["adam", "adamw"],
    },
}


def _py(v):
    return v.item() if hasattr(v, "item") else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "efficientnet"], required=True)
    ap.add_argument("--data_root")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--trial_epochs", type=int, default=5)
    ap.add_argument("--head_epochs", type=int, default=2)
    ap.add_argument("--subset", type=float, default=0.25)
    ap.add_argument("--weights", default="imagenet")
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    if args.data_root:
        config.set_data_root(args.data_root)

    out = config.OUTPUT_ROOT / f"tuning_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    samples = list(ParameterSampler(SPACES[args.model], n_iter=args.trials, random_state=args.seed))
    rows = []
    for i, params in enumerate(samples):
        params = {k: _py(v) for k, v in params.items()}
        print(f"\n=== Trial {i + 1}/{len(samples)}: {params}")
        trial_dir = out / f"trial_{i:02d}"
        try:
            cfg = run_training(args.model, run_dir=trial_dir, epochs=args.trial_epochs,
                               head_epochs=args.head_epochs, weights=args.weights, subset=args.subset,
                               clahe=args.clahe, seed=args.seed, verbose=1, **params)
            rows.append({**params, "val_auc": cfg["best_val_auc"], "best_epoch": cfg["best_epoch"]})
        except Exception as e:  # keep the search going if one configuration fails (e.g. OOM)
            print("Trial failed:", e)
            rows.append({**params, "val_auc": float("nan"), "best_epoch": -1})
        shutil.rmtree(trial_dir, ignore_errors=True)   # keep only the table, not the weights
        keras.backend.clear_session()
        gc.collect()
        pd.DataFrame(rows).to_csv(out / "trials.csv", index=False)

    df = pd.DataFrame(rows).dropna(subset=["val_auc"])
    if df.empty:
        raise SystemExit("All trials failed.")
    best = df.loc[df.val_auc.idxmax()]
    best_params = {k: _py(best[k]) for k in SPACES[args.model]}
    (out / "best_params.json").write_text(json.dumps(
        {"model": args.model, "best_params": best_params, "best_val_auc": float(best.val_auc),
         "n_trials": len(samples), "subset": args.subset, "trial_epochs": args.trial_epochs}, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(len(df)), df.val_auc.values)
    axes[0].set_xlabel("trial")
    axes[0].set_ylabel("validation AUC")
    axes[0].set_title("Validation AUC per trial")
    axes[0].set_ylim(max(0.0, df.val_auc.min() - 0.05), min(1.0, df.val_auc.max() + 0.03))
    sc = axes[1].scatter(df.lr, df.val_auc, c=df.dropout, cmap="viridis")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("validation AUC")
    axes[1].set_title("AUC vs learning rate (colour = dropout)")
    fig.colorbar(sc, ax=axes[1])
    fig.tight_layout()
    fig.savefig(out / "tuning_results.png", dpi=150)
    print("\nBest params:", best_params, f"(val AUC {best.val_auc:.4f})")
    print("Saved:", out / "best_params.json")


if __name__ == "__main__":
    main()
