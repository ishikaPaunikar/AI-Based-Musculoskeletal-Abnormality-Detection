
"""Evaluate a trained MURA model on a limited or full test set.

Demo:
    python -m src.evaluate --run_name efficientnet --max_images 15 --n_boot 50

Full evaluation:
    python -m src.evaluate --run_name efficientnet

NOTE:
    Small samples are for software demonstration only.
    They are not reliable clinical performance estimates.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    auc,
    cohen_kappa_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

from . import config
from .data import make_splits, build_dataset, summarize_split


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def _div(a, b):
    return float(a / b) if b else float("nan")


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    both_classes = len(np.unique(y_true)) == 2

    return {
        "threshold": float(threshold),
        "accuracy": _div(tp + tn, len(y_true)),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "specificity": _div(tn, tn + fp),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if both_classes else float("nan")
        ),
        "cohen_kappa": (
            float(cohen_kappa_score(y_true, y_pred))
            if both_classes else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }


def youden_threshold(y_true, y_prob):
    """Choose a threshold using validation data only."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    threshold = thresholds[np.argmax(tpr - fpr)]

    if not np.isfinite(threshold):
        return 0.5

    return float(np.clip(threshold, 0.0, 1.0))


def bootstrap_auc_ci(df, n_boot=50, seed=0, prob_col="prob"):
    """Patient-level bootstrap confidence interval."""
    if df.empty or df["patient_id"].nunique() < 2:
        return [float("nan"), float("nan")]

    rng = np.random.RandomState(seed)
    groups = df.groupby("patient_id").indices
    keys = np.array(list(groups.keys()))

    y = df["label"].to_numpy()
    p = df[prob_col].to_numpy()
    aucs = []

    for _ in range(n_boot):
        sampled_keys = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[k] for k in sampled_keys])

        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))

    if not aucs:
        return [float("nan"), float("nan")]

    return [
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    ]


def study_level(df):
    if df.empty:
        return pd.DataFrame(
            columns=["study_id", "label", "prob", "patient_id", "body_part"]
        )

    return (
        df.groupby("study_id")
        .agg(
            label=("label", "first"),
            prob=("prob", "mean"),
            patient_id=("patient_id", "first"),
            body_part=("body_part", "first"),
        )
        .reset_index()
    )


def per_body_part(df, threshold):
    rows = []

    for body_part, group in df.groupby("body_part"):
        metrics = compute_metrics(
            group["label"], group["prob"], threshold
        )

        rows.append({
            "body_part": body_part,
            "n_images": len(group),
            "n_abnormal": int(group["label"].sum()),
            "auc": metrics["roc_auc"],
            "sensitivity": metrics["recall"],
            "specificity": metrics["specificity"],
            "accuracy": metrics["accuracy"],
        })

    columns = [
        "body_part", "n_images", "n_abnormal", "auc",
        "sensitivity", "specificity", "accuracy"
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values(
        "auc", ascending=False, na_position="last"
    )


# -------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------

def plot_confusion(metrics, path, title):
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ])

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            row_total = cm[i].sum()
            percentage = cm[i, j] / row_total if row_total else 0.0

            ax.text(
                j, i,
                f"{cm[i, j]}\n({percentage:.1%} of row)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    ax.set_xticks([0, 1], ["Normal", "Abnormal"])
    ax.set_yticks([0, 1], ["Normal", "Abnormal"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, threshold, path, title):
    if len(np.unique(y_true)) < 2:
        print(f"Skipping ROC plot: {title} has only one class.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    metrics = compute_metrics(y_true, y_prob, threshold)

    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)

    ax.scatter(
        [1 - metrics["specificity"]],
        [metrics["recall"]],
        c="red",
        zorder=5,
        label=f"Operating point (threshold={threshold:.2f})",
    )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------

def predict_probs(model, df, batch_size, clahe):
    if df.empty:
        return np.array([], dtype=float)

    dataset = build_dataset(
        df,
        batch_size,
        training=False,
        clahe=clahe,
    )

    return model.predict(dataset, verbose=1).ravel()


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate_run(
    run_name,
    batch_size=32,
    n_boot=50,
    max_images=None,
):
    run_dir = config.OUTPUT_ROOT / run_name
    config_path = run_dir / "config.json"
    model_path = run_dir / "best_model.keras"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing model configuration: {config_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained model: {model_path}")

    cfg = json.loads(config_path.read_text())
    model = keras.models.load_model(model_path, compile=False)

    # Preserve the project's existing patient-level data split.
    _, val_df, test_df = make_splits(
        subset=cfg["subset"],
        seed=cfg["seed"],
        val_fraction=cfg["val_fraction"],
    )

    print(summarize_split(val_df, "val"))
    print(summarize_split(test_df, "test"))

    # Limit the samples evaluated in each split for a quick demo.
    if max_images is not None:
        if max_images < 1:
            raise ValueError("--max_images must be at least 1")

        val_df = val_df.head(max_images).copy()
        test_df = test_df.head(max_images).copy()

    print(f"Validation images evaluated: {len(val_df)}")
    print(f"Test images evaluated: {len(test_df)}")

    val_df["prob"] = predict_probs(
        model, val_df, batch_size, cfg["clahe"]
    )
    test_df["prob"] = predict_probs(
        model, test_df, batch_size, cfg["clahe"]
    )

    val_df.to_csv(run_dir / "predictions_val.csv", index=False)
    test_df.to_csv(run_dir / "predictions_test.csv", index=False)

    # Use validation predictions for threshold selection.
    threshold = youden_threshold(val_df["label"], val_df["prob"])

    study_val = study_level(val_df)
    study_threshold = youden_threshold(
        study_val["label"], study_val["prob"]
    )

    study_test = study_level(test_df)

    metrics = {
        "run_name": run_name,
        "model_type": cfg["model_type"],
        "evaluation_mode": (
            f"Demo sample: up to {max_images} images per split"
            if max_images is not None else "Full split evaluation"
        ),
        "protocol": (
            "The official MURA valid split is used as the test split. "
            "Thresholds are selected from the internal validation split. "
            "When a small sample is requested, metrics are demonstration-only."
        ),
        "image_level_tuned_threshold": compute_metrics(
            test_df["label"], test_df["prob"], threshold
        ),
        "image_level_default_0.5": compute_metrics(
            test_df["label"], test_df["prob"], 0.5
        ),
        "study_level_tuned_threshold": compute_metrics(
            study_test["label"], study_test["prob"], study_threshold
        ),
        "validation_image_level_tuned_threshold": compute_metrics(
            val_df["label"], val_df["prob"], threshold
        ),
        "auc_95ci_patient_bootstrap": bootstrap_auc_ci(
            test_df, n_boot=n_boot
        ),
        "n_validation_images": len(val_df),
        "n_test_images": len(test_df),
        "n_test_studies": len(study_test),
        "n_test_patients": int(test_df["patient_id"].nunique()),
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True)
    )

    per_body_part(test_df, threshold).to_csv(
        run_dir / "per_body_part.csv", index=False
    )

    if not test_df.empty:
        plot_confusion(
            metrics["image_level_tuned_threshold"],
            run_dir / "confusion_matrix.png",
            f"{run_name}: image-level (threshold={threshold:.2f})",
        )

        plot_confusion(
            metrics["study_level_tuned_threshold"],
            run_dir / "confusion_matrix_study.png",
            f"{run_name}: study-level",
        )

        plot_roc(
            test_df["label"].to_numpy(),
            test_df["prob"].to_numpy(),
            threshold,
            run_dir / "roc_curve.png",
            f"{run_name}: ROC (test sample)",
        )

    # Store thresholds for the application.
    cfg["threshold"] = threshold
    cfg["study_threshold"] = study_threshold
    config_path.write_text(json.dumps(cfg, indent=2))

    result = metrics["image_level_tuned_threshold"]

    print(f"\nRun: {run_name}")
    print(f"Threshold: {threshold:.3f}")

    for key in [
        "accuracy", "precision", "recall",
        "specificity", "f1", "roc_auc"
    ]:
        print(f"{key}: {result[key]}")

    print("AUC 95% CI:", metrics["auc_95ci_patient_bootstrap"])
    print(f"Metrics saved to: {run_dir / 'metrics.json'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MURA model."
    )

    parser.add_argument("--run_name", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_boot", type=int, default=50)
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum images per split; e.g. 15 for a small demo.",
    )

    args = parser.parse_args()

    if args.data_root:
        config.set_data_root(args.data_root)

    evaluate_run(
        run_name=args.run_name,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()
