"""Train the baseline CNN or EfficientNetB0.

    python -m src.train --model baseline     --run_name baseline
    python -m src.train --model efficientnet --run_name efficientnet [--params outputs/tuning_efficientnet/best_params.json]

EfficientNetB0 uses two phases: (1) train the new head with the backbone frozen,
(2) fine-tune the top backbone layers at a low learning rate.
Model selection uses validation AUC on the patient-held-out validation split only.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config
from .data import build_dataset, class_weights, make_splits, summarize_split
from .models import build_model, compile_model, set_backbone_trainable


def plot_history(hist, path, boundary=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (key, title) in zip(axes, [("loss", "Loss"), ("auc", "ROC-AUC"), ("accuracy", "Accuracy")]):
        ax.plot(hist.index + 1, hist[key], label="train")
        ax.plot(hist.index + 1, hist["val_" + key], label="validation")
        if boundary:
            ax.axvline(boundary + 0.5, color="gray", ls="--", label="start fine-tuning")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_training(model_type, run_name=None, run_dir=None, lr=1e-3, ft_lr=1e-5, batch_size=32, dropout=0.3,
                 epochs=15, head_epochs=3, unfreeze_layers=40, optimizer="adam", weights="imagenet",
                 subset=1.0, clahe=False, seed=config.SEED, verbose=1):
    keras.utils.set_random_seed(seed)
    run_dir = Path(run_dir or config.OUTPUT_ROOT / (run_name or model_type))
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _ = make_splits(subset=subset, seed=seed, load_test=False)
    if verbose:
        print(summarize_split(train_df, "train"))
        print(summarize_split(val_df, "val"))
    train_ds = build_dataset(train_df, batch_size, training=True, clahe=clahe, augment=True)
    val_ds = build_dataset(val_df, batch_size, training=False, clahe=clahe)
    cw = class_weights(train_df)

    model = build_model(model_type, dropout=dropout, weights=weights)
    ckpt = keras.callbacks.ModelCheckpoint(
        str(run_dir / "best_model.keras"), monitor="val_auc", mode="max", save_best_only=True
    )
    histories, boundary = [], None

    def fit_phase(n_epochs, patience, initial_epoch):
        cbs = [
            ckpt,
            keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=patience),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        ]
        h = model.fit(train_ds, validation_data=val_ds, epochs=initial_epoch + n_epochs,
                      initial_epoch=initial_epoch, class_weight=cw, callbacks=cbs, verbose=verbose)
        histories.append(pd.DataFrame(h.history))

    t0 = time.time()
    if model_type == "baseline":
        compile_model(model, lr, optimizer)
        fit_phase(epochs, patience=5, initial_epoch=0)
    elif unfreeze_layers <= 0 or epochs <= head_epochs:
        set_backbone_trainable(model, 0)
        compile_model(model, lr, optimizer)
        fit_phase(epochs, patience=4, initial_epoch=0)
    else:
        set_backbone_trainable(model, 0)
        compile_model(model, lr, optimizer)
        fit_phase(head_epochs, patience=head_epochs, initial_epoch=0)
        boundary = head_epochs
        set_backbone_trainable(model, unfreeze_layers)
        compile_model(model, ft_lr, optimizer)
        fit_phase(epochs - head_epochs, patience=4, initial_epoch=head_epochs)
    elapsed = time.time() - t0

    hist = pd.concat(histories, ignore_index=True)
    hist.to_csv(run_dir / "history.csv", index=False)
    plot_history(hist, run_dir / "training_curves.png", boundary)

    cfg = dict(
        model_type=model_type, run_name=run_dir.name, lr=lr, ft_lr=ft_lr, batch_size=batch_size,
        dropout=dropout, epochs=epochs, head_epochs=head_epochs, unfreeze_layers=unfreeze_layers,
        optimizer=optimizer, weights=str(weights), subset=subset, clahe=clahe, seed=seed,
        img_size=config.IMG_SIZE, val_fraction=config.VAL_FRACTION,
        best_val_auc=float(hist["val_auc"].max()), best_epoch=int(hist["val_auc"].idxmax()) + 1,
        train_images=len(train_df), val_images=len(val_df), train_time_sec=round(elapsed, 1),
        tf_version=tf.__version__, threshold=0.5,
    )
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    if verbose:
        print(f"Best validation AUC {cfg['best_val_auc']:.4f} at epoch {cfg['best_epoch']} -> {run_dir}")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "efficientnet"], required=True)
    ap.add_argument("--run_name")
    ap.add_argument("--data_root")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--head_epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft_lr", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--unfreeze_layers", type=int, default=40)
    ap.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    ap.add_argument("--weights", default="imagenet", help="'imagenet' or 'none'")
    ap.add_argument("--subset", type=float, default=1.0, help="fraction of patients to use (quick runs)")
    ap.add_argument("--clahe", action="store_true", help="apply CLAHE contrast enhancement")
    ap.add_argument("--params", help="best_params.json from src.tune; overrides hyper-parameters")
    args = ap.parse_args()
    if args.data_root:
        config.set_data_root(args.data_root)
    kw = {k: getattr(args, k) for k in ["lr", "ft_lr", "batch_size", "dropout", "unfreeze_layers", "optimizer"]}
    if args.params:
        kw.update(json.loads(Path(args.params).read_text())["best_params"])
        print("Using tuned hyper-parameters:", kw)
    run_training(args.model, args.run_name, epochs=args.epochs, head_epochs=args.head_epochs,
                 weights=args.weights, subset=args.subset, clahe=args.clahe, **kw)


if __name__ == "__main__":
    main()
