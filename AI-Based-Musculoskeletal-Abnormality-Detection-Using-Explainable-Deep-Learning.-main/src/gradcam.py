"""Grad-CAM (Selvaraju et al., 2017) for the binary Normal/Abnormal models.

The heat-map explains the model's P(abnormal): warm regions push the score UP.
For a 'Normal' prediction the map therefore shows what little evidence, if any, pointed
towards abnormality. Grad-CAM is a coarse, qualitative aid - NOT a lesion localiser.

CLI (gallery of real TP / TN / FP / FN test examples):
    python -m src.gradcam --run_name efficientnet --n 4
"""
import argparse
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from . import config
from .data import load_image


def find_last_conv_layer(model):
    """Last layer with a 4-D (batch, H, W, C) output - the final conv feature map."""
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except Exception:
            continue
        if len(shape) == 4:
            return layer.name
    raise ValueError("No convolutional feature map found in model")


def build_grad_model(model, layer_name=None):
    layer_name = layer_name or find_last_conv_layer(model)
    return keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])


def compute_gradcam(grad_model, img_batch):
    """img_batch: (1, H, W, 3) float [0, 255]. Returns (heatmap in [0,1] at feature-map size, P(abnormal))."""
    x = tf.convert_to_tensor(img_batch, dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, 0]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                       # channel importance
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam).numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, float(preds[0, 0])


def make_gradcam_heatmap(model, img_batch, layer_name=None):
    return compute_gradcam(build_grad_model(model, layer_name), img_batch)


def overlay_heatmap(gray_u8, heatmap, alpha=0.4):
    """Blend a JET heat-map onto a grayscale uint8 image. Returns RGB uint8."""
    h, w = gray_u8.shape[:2]
    hm = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hm = cv2.applyColorMap(np.uint8(255 * np.clip(hm, 0, 1)), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    base = np.stack([gray_u8] * 3, axis=-1).astype(np.uint8)
    return cv2.addWeighted(base, 1 - alpha, hm, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--n", type=int, default=4, help="examples per outcome category")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = config.OUTPUT_ROOT / args.run_name
    cfg = json.loads((run_dir / "config.json").read_text())
    thr = cfg.get("threshold", 0.5)
    model = keras.models.load_model(run_dir / "best_model.keras", compile=False)
    grad_model = build_grad_model(model)
    df = pd.read_csv(run_dir / "predictions_test.csv")   # produced by src.evaluate
    df["pred"] = (df.prob >= thr).astype(int)
    cats = [("True positive", (df.label == 1) & (df.pred == 1)), ("True negative", (df.label == 0) & (df.pred == 0)),
            ("False positive", (df.label == 0) & (df.pred == 1)), ("False negative", (df.label == 1) & (df.pred == 0))]

    fig, axes = plt.subplots(len(cats), args.n, figsize=(3.6 * args.n, 2.2 * len(cats) + 0.6), squeeze=False)
    for r, (name, mask) in enumerate(cats):
        sub = df[mask]
        sub = sub.sample(min(args.n, len(sub)), random_state=args.seed) if len(sub) else sub
        for c in range(args.n):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(sub):
                continue
            row = sub.iloc[c]
            x = load_image(row.path, cfg["img_size"], cfg["clahe"])
            heat, prob = compute_gradcam(grad_model, x[None])
            gray = x[..., 0].astype(np.uint8)
            ax.imshow(np.hstack([np.stack([gray] * 3, -1), overlay_heatmap(gray, heat)]))
            ax.set_title(f"{name}\n{row.body_part}, P(abn)={prob:.2f}", fontsize=8)
    fig.suptitle(f"Grad-CAM examples - {args.run_name} (left: input, right: heat-map)", fontsize=10)
    fig.tight_layout()
    out = run_dir / "gradcam_examples.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)


if __name__ == "__main__":
    main()
