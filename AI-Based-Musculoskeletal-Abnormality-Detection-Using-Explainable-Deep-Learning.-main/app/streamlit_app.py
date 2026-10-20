"""Streamlit demo: upload musculoskeletal X-rays, view predictions and Grad-CAM.

Run from the project root with ``streamlit run app/streamlit_app.py``.
Educational research prototype only; not a clinical diagnostic tool.
"""
import html
import json
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from tensorflow import keras

from src import config
from src.data import preprocess_array
from src.gradcam import build_grad_model, compute_gradcam, overlay_heatmap

from app.styles import APP_CSS

st.set_page_config(
    page_title="MSK X-ray Abnormality Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{APP_CSS}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def load_run(run_name):
    run_dir = config.OUTPUT_ROOT / run_name
    cfg = json.loads((run_dir / "config.json").read_text())
    model = keras.models.load_model(run_dir / "best_model.keras", compile=False)
    return model, build_grad_model(model), cfg


def available_runs():
    return sorted(
        p.parent.name
        for p in config.OUTPUT_ROOT.glob("*/best_model.keras")
        if (p.parent / "config.json").exists()
    )


def analyse(grad_model, cfg, pil_img):
    gray = np.array(pil_img.convert("L"))
    x = preprocess_array(gray, cfg["img_size"], cfg.get("clahe", False))
    heat, prob = compute_gradcam(grad_model, x[None])
    return prob, x[..., 0].astype(np.uint8), heat


def probability_gauge(probability, threshold, dark_mode=False):
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "P(abnormal)", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": "#2563EB"},
                "bgcolor": "rgba(14, 165, 233, 0.08)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(14, 165, 233, 0.08)"},
                    {"range": [threshold * 100, 100], "color": "rgba(239, 68, 68, 0.08)"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(
        height=205,
        margin={"l": 24, "r": 24, "t": 46, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": "#CBD5E1" if dark_mode else "#334155"},
    )
    return figure


def make_report(results, model_name, threshold, study_probability, opacity):
    rows = max(1, len(results))
    figure, axes = plt.subplots(rows, 3, figsize=(12, 3.8 * rows), squeeze=False)
    figure.suptitle(
        f"Musculoskeletal X-ray analysis | {model_name} | "
        f"Study P(abnormal): {study_probability:.1%} | Threshold: {threshold:.2f}",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    for row, (name, probability, gray, heat) in enumerate(results):
        resized_heat = cv2.resize(heat, gray.shape[::-1], interpolation=cv2.INTER_CUBIC)
        axes[row, 0].imshow(gray, cmap="gray")
        axes[row, 0].set_title(f"{name} — preprocessed input\nP(abnormal): {probability:.1%}")
        axes[row, 1].imshow(gray, cmap="gray")
        axes[row, 1].imshow(np.clip(resized_heat, 0, 1), cmap="jet", alpha=0.48, vmin=0, vmax=1)
        axes[row, 1].set_title("Grad-CAM heat-map")
        axes[row, 2].imshow(overlay_heatmap(gray, heat, opacity))
        axes[row, 2].set_title("Grad-CAM overlay")
        for axis in axes[row]:
            axis.axis("off")
    figure.text(
        0.5,
        0.012,
        "Educational research output only. Not a medical device; not for diagnosis or treatment decisions.",
        ha="center",
        fontsize=9,
        color="#64748B",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    png = BytesIO()
    pdf = BytesIO()
    figure.savefig(png, format="png", dpi=150, bbox_inches="tight")
    figure.savefig(pdf, format="pdf", bbox_inches="tight")
    plt.close(figure)
    return png.getvalue(), pdf.getvalue()


def metric_card(label, value, note=None):
    with st.container(border=True):
        st.metric(label, value, help=note)


def render_performance(run_name, runs, dark_mode=False):
    run_dir = config.OUTPUT_ROOT / run_name
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        st.info(f"No evaluation metrics are available for **{run_name}** yet. Run `python -m src.evaluate --run_name {run_name}` to generate them.")
    else:
        metrics = json.loads(metrics_path.read_text())
        values = metrics["image_level_tuned_threshold"]
        st.subheader(f"{run_name} · held-out test performance")
        st.caption(
            f"Image-level metrics at validation-tuned threshold {values['threshold']:.3f}. "
            f"{metrics.get('n_test_images', 0):,} images · "
            f"{metrics.get('n_test_studies', 0):,} studies · "
            f"{metrics.get('n_test_patients', 0):,} patients"
        )
        metric_specs = [
            ("Accuracy", "accuracy"),
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("F1 score", "f1"),
            ("ROC-AUC", "roc_auc"),
        ]
        for columns, (label, key) in zip(st.columns(5), metric_specs):
            with columns.container(border=True):
                st.metric(label, f"{values[key]:.3f}")

        confusion = np.array([[values["tn"], values["fp"]], [values["fn"], values["tp"]]])
        chart_text_color = "#CBD5E1" if dark_mode else "#334155"
        chart_col, roc_col = st.columns(2)
        with chart_col.container(border=True):
            st.markdown("**Confusion matrix**")
            fig = go.Figure(
                go.Heatmap(
                    z=confusion,
                    x=["Predicted normal", "Predicted abnormal"],
                    y=["Actual normal", "Actual abnormal"],
                    text=confusion,
                    texttemplate="%{text}",
                    colorscale=[[0, "#E0F2FE"], [1, "#2563EB"]],
                    showscale=False,
                    hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
                )
            )
            fig.update_layout(
                height=340,
                margin={"l": 8, "r": 8, "t": 12, "b": 8},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color=chart_text_color,
                xaxis={"side": "bottom"},
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with roc_col.container(border=True):
            st.markdown("**ROC curve**")
            predictions = run_dir / "predictions_test.csv"
            if predictions.exists():
                frame = pd.read_csv(predictions)
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(frame["label"], frame["prob"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC {values['roc_auc']:.3f}",
                                         line={"color": "#0EA5E9", "width": 3}))
                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                                         line={"color": "#94A3B8", "dash": "dash"}))
                fig.update_layout(
                    height=340,
                    margin={"l": 12, "r": 12, "t": 12, "b": 12},
                    xaxis_title="False positive rate",
                    yaxis_title="True positive rate",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color=chart_text_color,
                    legend={"orientation": "h", "y": -0.25},
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("The ROC curve appears after test predictions are exported by the evaluation step.")

        ci = metrics.get("auc_95ci_patient_bootstrap")
        if ci:
            st.caption(f"Patient-bootstrap ROC-AUC 95% CI: {ci[0]:.3f}–{ci[1]:.3f}")
        body_parts = run_dir / "per_body_part.csv"
        if body_parts.exists():
            with st.expander("Performance by body part"):
                st.dataframe(pd.read_csv(body_parts), use_container_width=True, hide_index=True)
        for image_name, title in [
            ("training_curves.png", "Training history"),
            ("gradcam_examples.png", "Grad-CAM examples"),
        ]:
            image_path = run_dir / image_name
            if image_path.exists():
                with st.expander(title):
                    st.image(str(image_path), use_container_width=True)

    st.subheader("Baseline CNN vs EfficientNet")
    comparison_path = config.OUTPUT_ROOT / "comparison" / "comparison_table.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        image_level = comparison[comparison["level"] == "image"] if "level" in comparison else comparison
        st.dataframe(image_level, use_container_width=True, hide_index=True)
    else:
        comparison_rows = []
        for candidate in runs:
            candidate_metrics = config.OUTPUT_ROOT / candidate / "metrics.json"
            if candidate_metrics.exists():
                candidate_values = json.loads(candidate_metrics.read_text())["image_level_tuned_threshold"]
                comparison_rows.append(
                    {
                        "Model": candidate,
                        "Accuracy": candidate_values["accuracy"],
                        "Precision": candidate_values["precision"],
                        "Recall": candidate_values["recall"],
                        "F1": candidate_values["f1"],
                        "ROC-AUC": candidate_values["roc_auc"],
                    }
                )
        if comparison_rows:
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Evaluation outputs for the baseline and EfficientNet models will appear here when available.")

    comparison_dir = config.OUTPUT_ROOT / "comparison"
    if (comparison_dir / "roc_comparison.png").exists():
        chart_a, chart_b = st.columns(2)
        chart_a.image(str(comparison_dir / "metrics_comparison.png"), use_container_width=True)
        chart_b.image(str(comparison_dir / "roc_comparison.png"), use_container_width=True)


def render_about():
    sections = [
        ("🗂️", "Dataset", "MURA v1.1 is a public collection of musculoskeletal radiographs covering elbow, finger, forearm, hand, humerus, shoulder, and wrist. Labels are normal or abnormal at study level."),
        ("🧠", "Method", "The app can load a baseline CNN or EfficientNetB0 model. Grad-CAM highlights image regions associated with the model’s abnormality score; it is a qualitative visualization, not a lesion localizer."),
        ("⚠️", "Limitations", "This research prototype has not been clinically validated or cleared for medical use. Dataset labels, acquisition sites, and populations limit generalizability. Probabilities are not calibrated, and out-of-scope inputs can still produce a score."),
        ("🤝", "Ethics & safe use", "Use only for educational or research exploration. Never use the output to diagnose, treat, triage, or make decisions about a patient. Seek assessment from a qualified clinician."),
    ]
    for index in range(0, len(sections), 2):
        left, right = st.columns(2)
        for column, (icon, title, copy) in zip((left, right), sections[index:index + 2]):
            with column.container(border=True):
                st.markdown(f"### {icon} {title}")
                st.write(copy)


runs = available_runs()
with st.sidebar:
    st.markdown("## 🦴 MURA")
    st.caption("Clinical research interface")
    st.markdown("---")

    st.markdown("### 🧠 Model")
    if runs:
        default_run = runs.index("efficientnet") if "efficientnet" in runs else 0
        run = st.selectbox("Architecture", runs, index=default_run, help="Select a trained model checkpoint.")
        model, grad_model, cfg = load_run(run)
    else:
        run, cfg = None, None
        st.error("No trained model found in outputs/. Train a model first.")

    st.markdown("### 🎚️ Inference settings")
    if cfg:
        threshold = st.slider(
            "Decision threshold",
            0.05,
            0.95,
            min(0.95, max(0.05, float(cfg.get("threshold", 0.5)))),
            0.01,
            help="Scores at or above this value are classified as abnormal. Default is validation-tuned.",
        )
    else:
        threshold = 0.5
        st.caption("Threshold is available after a model is trained.")

    st.markdown("### 🎨 Visualization")
    opacity = st.slider(
        "Heat-map opacity",
        0.1,
        0.9,
        0.4,
        0.05,
        help="Adjust the Grad-CAM overlay intensity.",
    )

    st.markdown("### 🌓 Appearance")
    appearance = st.radio(
        "Color mode",
        ["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        help="Switch the interface between light and dark themes.",
    )

    with st.container(border=True):
        st.markdown("**📋 Model info**")
        if cfg:
            st.caption(
                f"Architecture · {cfg.get('model_type', run)}\n\n"
                f"Input size · {cfg.get('img_size', 224)} px\n\n"
                f"Preprocessing · CLAHE {'on' if cfg.get('clahe', False) else 'off'}"
            )
        else:
            st.caption("Model details will appear when a trained checkpoint is available.")

dark_mode = appearance == "Dark"
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
          --background-color: #0B1220;
          --secondary-background-color: #111C2E;
          --text-color: #E2E8F0;
          --clinical-page: #0B1220;
          --clinical-surface: #111C2E;
          --clinical-ink: #E2E8F0;
          --clinical-muted: #94A3B8;
          --clinical-border: #253449;
        }
        [data-testid="stFileUploader"] section { background: rgba(14,165,233,.07); }
        [data-testid="stDataFrame"] { background: #111C2E; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="hero">
      <div class="hero-mark">🦴</div>
      <div>
        <div class="eyebrow">MUSCULOSKELETAL IMAGING · RESEARCH</div>
        <h1>Musculoskeletal X-ray<br class="hero-break"> Abnormality Detection</h1>
        <p>AI-assisted image review with visual explanations for research and education.</p>
        <div class="hero-badges">
          <span>Model · {model}</span><span>Dataset · MURA</span><span>Input · 224 px</span>
        </div>
      </div>
    </section>
    """.format(model=html.escape(run or "No model loaded")),
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="clinical-notice"><span>ⓘ</span><div><b>Research prototype — not a medical device.</b> '
    "Not clinically validated. Never use for diagnosis or treatment decisions; consult a qualified clinician.</div></div>",
    unsafe_allow_html=True,
)

tab_prediction, tab_performance, tab_about = st.tabs(
    ["🔬  Prediction", "📊  Model performance", "ℹ️  About & limitations"]
)

with tab_prediction:
    if not runs:
        st.info("Train and save a model in `outputs/`, then reload this app to begin.")
    else:
        st.markdown("## Image review")
        st.caption("Upload one or more PNG or JPG radiographs from a single study.")
        uploaded_files = st.file_uploader(
            "Choose X-ray image(s)",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Multiple images are averaged into one study-level abnormality score.",
            label_visibility="collapsed",
        )
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} image{'s' if len(uploaded_files) != 1 else ''} selected**")
            preview_columns = st.columns(min(len(uploaded_files), 5))
            for column, uploaded in zip(preview_columns, uploaded_files[:5]):
                try:
                    with column:
                        st.image(Image.open(uploaded), caption=uploaded.name, use_container_width=True)
                except (OSError, ValueError) as error:
                    st.error(f"Could not preview {uploaded.name}: {error}")
            if len(uploaded_files) > 5:
                st.caption(f"+ {len(uploaded_files) - 5} more images")

            results = []
            with st.spinner(f"Analyzing {len(uploaded_files)} X-ray image(s)…"):
                for uploaded in uploaded_files:
                    try:
                        uploaded.seek(0)
                        image = Image.open(uploaded).copy()
                        results.append((uploaded.name, *analyse(grad_model, cfg, image)))
                    except Exception as error:
                        st.error(f"Could not process {uploaded.name}: {error}")

            if results:
                study_probability = float(np.mean([item[1] for item in results]))
                abnormal = study_probability >= threshold
                confidence = study_probability if abnormal else 1 - study_probability
                borderline = abs(study_probability - threshold) < 0.10
                result_label = "Abnormal" if abnormal else "Normal"

                st.markdown("## Study result" if len(results) > 1 else "## Prediction result")
                result_col, gauge_col, metric_col = st.columns([1.15, 1, 1])
                with result_col.container(border=True):
                    st.markdown(
                        f'<div class="result-summary {"abnormal" if abnormal else "normal"}">'
                        f'<span class="result-icon">{"⚠️" if abnormal else "✓"}</span>'
                        f'<div><div class="result-kicker">{"STUDY-LEVEL RESULT" if len(results) > 1 else "MODEL RESULT"}</div>'
                        f'<div class="result-label">{result_label}</div></div></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Model: {run} · Decision threshold: {threshold:.2f}")
                    if borderline:
                        st.markdown(
                            '<span class="status-chip">⚠ Borderline / uncertain</span>',
                            unsafe_allow_html=True,
                        )
                with gauge_col.container(border=True):
                    st.plotly_chart(
                        probability_gauge(study_probability, threshold, dark_mode),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.caption("Amber marker indicates the selected decision threshold.")
                with metric_col:
                    metric_card("Confidence in predicted label", f"{confidence:.1%}")
                    metric_card("P(abnormal)", f"{study_probability:.1%}")
                    metric_card("Decision margin", f"{abs(study_probability - threshold):.1%}")

                st.caption("Confidence is based on the model’s raw, uncalibrated probability.")
                if len(results) > 1:
                    summary = pd.DataFrame(
                        [
                            {
                                "Image": name,
                                "P(abnormal)": f"{probability:.1%}",
                                "Image result": "Abnormal" if probability >= threshold else "Normal",
                            }
                            for name, probability, _, _ in results
                        ]
                    )
                    st.markdown("### Study image summary")
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                st.markdown("## Grad-CAM visual explanation")
                st.caption(
                    "Warm colors indicate regions that increased P(abnormal). "
                    "This qualitative aid is not a lesion localizer."
                )
                for index, (name, probability, gray, heat) in enumerate(results):
                    with st.expander(f"{name} · P(abnormal) {probability:.1%}", expanded=len(results) == 1):
                        show_overlay = st.toggle(
                            "Show Grad-CAM overlay",
                            value=True,
                            key=f"overlay_{index}_{name}",
                        )
                        original_col, heat_col, overlay_col = st.columns(3)
                        with original_col.container(border=True):
                            st.markdown("**Original**")
                            st.image(gray, use_container_width=True)
                        resized_heat = cv2.resize(heat, gray.shape[::-1], interpolation=cv2.INTER_CUBIC)
                        heat_image = np.uint8(255 * np.clip(resized_heat, 0, 1))
                        with heat_col.container(border=True):
                            st.markdown("**Heat-map**")
                            st.image(heat_image, use_container_width=True, clamp=True)
                        with overlay_col.container(border=True):
                            st.markdown("**Overlay**")
                            overlay = overlay_heatmap(gray, heat, opacity) if show_overlay else gray
                            st.image(overlay, use_container_width=True)
                st.markdown(
                    '<div class="attention-legend"><span>Low attention</span>'
                    '<div class="attention-gradient"></div><span>High attention</span></div>',
                    unsafe_allow_html=True,
                )

                png_report, pdf_report = make_report(results, run, threshold, study_probability, opacity)
                download_png, download_pdf = st.columns(2)
                download_png.download_button(
                    "⬇️ Download report (PNG)",
                    data=png_report,
                    file_name="mura_xray_report.png",
                    mime="image/png",
                    use_container_width=True,
                )
                download_pdf.download_button(
                    "⬇️ Download report (PDF)",
                    data=pdf_report,
                    file_name="mura_xray_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.markdown(
                '<div class="empty-state"><div class="empty-icon">🩻</div>'
                '<h3>Your image review starts here</h3>'
                '<p>Drop PNG or JPG radiographs above to view a model estimate and Grad-CAM explanation.</p>'
                '<span>Supported regions: elbow · finger · forearm · hand · humerus · shoulder · wrist</span></div>',
                unsafe_allow_html=True,
            )

with tab_performance:
    render_performance(run, runs, dark_mode) if run else st.info("Train a model to view evaluation outputs.")

with tab_about:
    render_about()

st.markdown(
    '<footer class="app-footer">Musculoskeletal X-ray Abnormality Detection · '
    "Educational research prototype · Not a medical device</footer>",
    unsafe_allow_html=True,
)
