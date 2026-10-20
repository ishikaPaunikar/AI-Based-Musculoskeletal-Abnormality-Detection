"""Generate healthcare-oriented insights as Markdown from REAL evaluation outputs.

    python -m src.insights --baseline baseline --challenger efficientnet

Every number in outputs/insights.md is read from metrics.json / per_body_part.csv /
comparison.json. The text only interprets those numbers; it contains no hard-coded results.
"""
import argparse
import json

import pandas as pd

from . import config


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--challenger", default="efficientnet")
    args = ap.parse_args()
    runs = [args.baseline, args.challenger]
    M = {r: json.loads((config.OUTPUT_ROOT / r / "metrics.json").read_text()) for r in runs}
    comp = json.loads((config.OUTPUT_ROOT / "comparison" / "comparison.json").read_text())

    L = ["# Healthcare insights (auto-generated from evaluation outputs)", "",
         "> Educational research prototype. Not validated for clinical use.", ""]

    L += ["## 1. Model comparison", ""]
    for r in runs:
        m = M[r]["image_level_tuned_threshold"]
        lo, hi = M[r]["auc_95ci_patient_bootstrap"]
        L.append(f"- **{r}**: AUC {m['roc_auc']:.3f} (95% CI {lo:.3f}-{hi:.3f}), sensitivity {pct(m['recall'])}, "
                 f"specificity {pct(m['specificity'])}, precision {pct(m['precision'])}, F1 {m['f1']:.3f}, "
                 f"Cohen's kappa {m['cohen_kappa']:.3f}.")
    d, (lo, hi) = comp["auc_delta_challenger_minus_baseline"], comp["auc_delta_95ci_patient_bootstrap"]
    verdict = ("The 95% interval excludes zero, so the AUC difference is unlikely to be due to test-patient sampling alone."
               if comp["ci_excludes_zero"] else
               "The 95% interval includes zero, so this test set does not show a reliable AUC difference.")
    L += ["", f"Paired patient-level bootstrap: AUC({args.challenger}) - AUC({args.baseline}) = {d:+.3f} "
              f"(95% CI {lo:+.3f} to {hi:+.3f}). {verdict}", ""]

    L += ["## 2. Clinical error trade-off (image level, validation-tuned threshold)", ""]
    for r in runs:
        m = M[r]["image_level_tuned_threshold"]
        pos, neg = m["tp"] + m["fn"], m["tn"] + m["fp"]
        L.append(f"- **{r}** missed {m['fn']} of {pos} abnormal images (false negatives) and flagged "
                 f"{m['fp']} of {neg} normal images (false positives). In a screening workflow, false negatives "
                 f"delay care while false positives add radiologist workload; the threshold can be shifted to favour "
                 f"either, at the cost of the other.")
    s = {r: M[r]["study_level_tuned_threshold"] for r in runs}
    L += ["", "Study-level (average over a study's views) results: " + "; ".join(
        f"{r}: AUC {s[r]['roc_auc']:.3f}, sensitivity {pct(s[r]['recall'])}, specificity {pct(s[r]['specificity'])}"
        for r in runs) + ".", ""]

    best = max(runs, key=lambda r: M[r]["image_level_tuned_threshold"]["roc_auc"])
    bp = pd.read_csv(config.OUTPUT_ROOT / best / "per_body_part.csv").dropna(subset=["auc"])
    L += [f"## 3. Performance by body part ({best}, the higher-AUC model)", ""]
    if len(bp):
        top, low = bp.iloc[0], bp.iloc[-1]
        L.append(f"- Highest AUC: **{top.body_part}** ({top.auc:.3f}, n={int(top.n_images)} images); "
                 f"lowest: **{low.body_part}** ({low.auc:.3f}, n={int(low.n_images)} images).")
        weak = bp[bp.sensitivity < bp.sensitivity.median()]
        if len(weak):
            L.append("- Below-median sensitivity: " + ", ".join(f"{r.body_part} ({pct(r.sensitivity)})" for r in weak.itertuples()) + ".")
        L.append("- Per-region sample sizes are modest; treat differences between regions as indicative, not conclusive.")
    L += ["", "| Body part | Images | Abnormal | AUC | Sensitivity | Specificity |", "|---|---|---|---|---|---|"]
    for r in bp.itertuples():
        L.append(f"| {r.body_part} | {r.n_images} | {r.n_abnormal} | {r.auc:.3f} | {pct(r.sensitivity)} | {pct(r.specificity)} |")

    L += ["", "## 4. Explainability", "",
          "- Grad-CAM maps (`outputs/<run>/gradcam_examples.png`) should be inspected for true positives, false positives and "
          "false negatives. A useful model highlights bone/joint regions; maps concentrated on image borders, "
          "text markers or implants indicate shortcut learning.",
          "- Heat-maps are coarse and qualitative; they are not validated lesion localisation.", "",
          "## 5. Limitations that affect interpretation", "",
          "- Single-dataset, single-institution (Stanford) data; no external validation.",
          "- Labels are study-level; every image in an abnormal study is labelled abnormal even if the finding is visible in one view only.",
          "- Test set = MURA's public validation split (official test set is hidden); it was not used for training or tuning.",
          "- Predicted probabilities are not calibrated. Subgroup (age, sex, device) performance is unknown.",
          "- Not a medical device; outputs must not be used for diagnosis or treatment decisions."]
    path = config.OUTPUT_ROOT / "insights.md"
    path.write_text("\n".join(L) + "\n")
    print("Saved", path)


if __name__ == "__main__":
    main()
