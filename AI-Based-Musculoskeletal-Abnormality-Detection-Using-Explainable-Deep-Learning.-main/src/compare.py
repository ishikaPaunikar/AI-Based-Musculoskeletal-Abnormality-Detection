"""Compare the baseline CNN with EfficientNetB0 using their saved evaluation outputs.

    python -m src.compare --baseline baseline --challenger efficientnet

Writes outputs/comparison/: comparison_table.{csv,md}, metric bar chart, overlaid ROC curves,
side-by-side confusion matrices and comparison.json (incl. paired patient-level bootstrap
for the AUC difference).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from . import config

METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]


def paired_auc_delta(a, b, n_boot=1000, seed=0):
    m = a[["path", "patient_id", "label", "prob"]].merge(b[["path", "prob"]], on="path", suffixes=("_a", "_b"))
    groups = m.groupby("patient_id").indices
    keys = np.array(list(groups.keys()))
    y, pa, pb = m.label.values, m.prob_a.values, m.prob_b.values
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        if len(np.unique(y[idx])) == 2:
            deltas.append(roc_auc_score(y[idx], pb[idx]) - roc_auc_score(y[idx], pa[idx]))
    point = roc_auc_score(y, pb) - roc_auc_score(y, pa)
    return float(point), [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--challenger", default="efficientnet")
    args = ap.parse_args()
    runs = [args.baseline, args.challenger]
    out = config.OUTPUT_ROOT / "comparison"
    out.mkdir(parents=True, exist_ok=True)

    M = {r: json.loads((config.OUTPUT_ROOT / r / "metrics.json").read_text()) for r in runs}
    P = {r: pd.read_csv(config.OUTPUT_ROOT / r / "predictions_test.csv") for r in runs}

    rows = []
    for r in runs:
        for level, key in [("image", "image_level_tuned_threshold"), ("study", "study_level_tuned_threshold")]:
            m = M[r][key]
            rows.append({"model": r, "level": level, "threshold": m["threshold"], **{k: m[k] for k in METRICS}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison_table.csv", index=False)
    (out / "comparison_table.md").write_text(md_table(table) + "\n")

    # grouped bar chart (image level)
    img = table[table.level == "image"].set_index("model")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.38
    for i, r in enumerate(runs):
        ax.bar(np.arange(len(METRICS)) + i * w, img.loc[r, METRICS].values, w, label=r)
    ax.set_xticks(np.arange(len(METRICS)) + w / 2, ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1)
    ax.set_title("Test-set performance (image level, validation-tuned threshold)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "metrics_comparison.png", dpi=150)
    plt.close(fig)

    # ROC overlay
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for r in runs:
        fpr, tpr, _ = roc_curve(P[r].label, P[r].prob)
        ax.plot(fpr, tpr, label=f"{r} (AUC {M[r]['image_level_tuned_threshold']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_comparison.png", dpi=150)
    plt.close(fig)

    # confusion matrices side by side
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, r in zip(axes, runs):
        m = M[r]["image_level_tuned_threshold"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Normal", "Abnormal"])
        ax.set_yticks([0, 1], ["Normal", "Abnormal"])
        ax.set_title(r)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    delta, ci = paired_auc_delta(P[args.baseline], P[args.challenger])
    summary = {"baseline": args.baseline, "challenger": args.challenger,
               "auc_delta_challenger_minus_baseline": delta, "auc_delta_95ci_patient_bootstrap": ci,
               "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    (out / "comparison.json").write_text(json.dumps(summary, indent=2))
    print(md_table(table))
    print(f"\nAUC delta ({args.challenger} - {args.baseline}) = {delta:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]")


if __name__ == "__main__":
    main()
