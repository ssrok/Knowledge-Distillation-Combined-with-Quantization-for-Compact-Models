"""
streamlit_app.py
----------------
Presentation dashboard for:
  "Knowledge Distillation Combined with Quantization for Compact Models"

Run with:
    cd knowledge-distillation-quantization
    streamlit run app/streamlit_app.py

Expected directory layout (relative to project root):
    kdq_results/
        models/
            m2_baseline/        (HF safetensors checkpoint)
            m3_distilled/       (HF safetensors checkpoint)
            m4_quantized_baseline.pt
            m5_quantized_distilled.pt
        results/
            model_comparison.csv
            model_comparison.png
            training_curves.png
            experiment_summary.json

M1 (BERT-base teacher) is loaded on-demand from the HuggingFace Hub
using the same model ID used during the Colab experiment.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# ── Resolve project root so the app works regardless of cwd ──────────────
APP_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
KDQ_DIR      = PROJECT_ROOT / "kdq_results"
MODELS_DIR   = KDQ_DIR / "models"
RESULTS_DIR  = KDQ_DIR / "results"

# Add project root to path so src.* imports work
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Page config (must be first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="KD + Quantization Demo",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model registry ───────────────────────────────────────────────────────
# Maps model ID → (display label, precision, checkpoint path or Hub ID,
#                  checkpoint type: 'hub' | 'hf_dir' | 'pt')
MODEL_REGISTRY = {
    "M1": {
        "label":      "M1 — BERT-base Teacher",
        "short":      "BERT-base (FP32)",
        "precision":  "FP32",
        "ckpt":       "textattack/bert-base-uncased-SST-2",
        "ckpt_type":  "hub",
        "desc":       "Fine-tuned BERT-base-uncased on SST-2. Used as teacher.",
    },
    "M2": {
        "label":      "M2 — DistilBERT Baseline",
        "short":      "DistilBERT (FP32)",
        "precision":  "FP32",
        "ckpt":       MODELS_DIR / "m2_baseline",
        "ckpt_type":  "hf_dir",
        "desc":       "DistilBERT fine-tuned directly on SST-2 (cross-entropy only).",
    },
    "M3": {
        "label":      "M3 — DistilBERT Distilled",
        "short":      "DistilBERT KD (FP32)",
        "precision":  "FP32",
        "ckpt":       MODELS_DIR / "m3_distilled",
        "ckpt_type":  "hf_dir",
        "desc":       "DistilBERT trained with knowledge distillation from M1 (α=0.5, T=2).",
    },
    "M4": {
        "label":      "M4 — Quantized Baseline ★ Best Trade-off",
        "short":      "DistilBERT INT8 (M2→M4)",
        "precision":  "INT8",
        "ckpt":       MODELS_DIR / "m4_quantized_baseline.pt",
        "ckpt_type":  "pt",
        "desc":       "Dynamic INT8 quantization of M2. Smallest model, fastest CPU inference.",
    },
    "M5": {
        "label":      "M5 — Quantized Distilled",
        "short":      "DistilBERT INT8 (M3→M5)",
        "precision":  "INT8",
        "ckpt":       MODELS_DIR / "m5_quantized_distilled.pt",
        "ckpt_type":  "pt",
        "desc":       "Dynamic INT8 quantization of M3.",
    },
}

TOKENIZER_ID = "bert-base-uncased"   # single tokenizer for all models
NUM_LABELS   = 2
MAX_LEN      = 128
LABEL_NAMES  = {0: "Negative 😞", 1: "Positive 😊"}
LABEL_COLORS = {0: "#FF4B4B", 1: "#21C354"}


# ════════════════════════════════════════════════════════════════════════
# Cached loaders
# ════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading tokenizer…")
def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


@st.cache_resource(show_spinner=False)
def load_model(model_id: str):
    """
    Load a model by its registry ID.
    Returns (model, error_message).  On failure returns (None, str).
    """
    from transformers import AutoModelForSequenceClassification

    reg = MODEL_REGISTRY[model_id]
    ckpt      = reg["ckpt"]
    ckpt_type = reg["ckpt_type"]

    try:
        if ckpt_type == "hub":
            model = AutoModelForSequenceClassification.from_pretrained(
                ckpt, num_labels=NUM_LABELS
            )

        elif ckpt_type == "hf_dir":
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                return None, f"Checkpoint directory not found: {ckpt_path}"
            model = AutoModelForSequenceClassification.from_pretrained(
                str(ckpt_path), num_labels=NUM_LABELS
            )

        elif ckpt_type == "pt":
            ckpt_path = Path(ckpt)
            if not ckpt_path.exists():
                return None, f"Checkpoint file not found: {ckpt_path}"
            # Quantized models are saved as full model objects via torch.save
            model = torch.load(str(ckpt_path), map_location="cpu",
                               weights_only=False)

        else:
            return None, f"Unknown checkpoint type: {ckpt_type}"

        model.eval()
        return model, None

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def is_model_available(model_id: str) -> bool:
    """Check whether the checkpoint file / directory for a model exists locally."""
    reg  = MODEL_REGISTRY[model_id]
    ckpt = reg["ckpt"]
    if reg["ckpt_type"] == "hub":
        return True   # always try Hub; we can't check without downloading
    path = Path(ckpt)
    return path.exists()


# ════════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════════

def run_inference(model, tokenizer, text: str):
    """
    Tokenise `text`, run a forward pass, return predicted label (int),
    confidence (float 0–1), and wall-clock time in milliseconds.

    Strips token_type_ids from the encoding when the model is a DistilBERT
    variant — DistilBERT has no segment embeddings and raises an error if
    token_type_ids are passed.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        padding=True,
    )
    # Quantized models only run on CPU
    device = torch.device("cpu")
    enc    = {k: v.to(device) for k, v in enc.items()}
    model  = model.to(device)

    # DistilBERT (M2–M5) has no token_type_ids — remove them unconditionally
    # for any model whose config says "distilbert", and also for quantized .pt
    # models that may not carry a full config attribute.
    model_type = ""
    if hasattr(model, "config"):
        model_type = getattr(model.config, "model_type", "").lower()
    # If model_type is empty (e.g. wrapped quantized model), infer from class name
    if not model_type:
        model_type = type(model).__name__.lower()
    if "distilbert" in model_type or "distil" in model_type:
        enc.pop("token_type_ids", None)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(**enc).logits
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs      = torch.softmax(logits, dim=-1).squeeze()
    pred_label = int(probs.argmax())
    confidence = float(probs[pred_label])

    return pred_label, confidence, elapsed_ms


# ════════════════════════════════════════════════════════════════════════
# Results loaders
# ════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_results_csv():
    csv_path = RESULTS_DIR / "model_comparison.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


@st.cache_resource
def load_summary_json():
    json_path = RESULTS_DIR / "experiment_summary.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════
# HTML table helper  (replaces st.table / st.dataframe — no pyarrow needed)
# ════════════════════════════════════════════════════════════════════════

def _html_table(
    headers: list,
    rows: list,
    highlight_rows: list = None,
    highlight_color: str = "#fff3cd",
) -> None:
    """
    Render a table as escaped HTML via st.markdown.

    All cell values are HTML-escaped so no injected markup can crash
    or alter the page.  Uses inline styles only — no external CSS.

    Args:
        headers:         List of column header strings.
        rows:            List of lists — each inner list is one row of cell strings.
        highlight_rows:  Zero-based indices of rows to colour.
        highlight_color: CSS background colour for highlighted rows.
    """
    import html as _html

    if highlight_rows is None:
        highlight_rows = []

    th_style = (
        "padding:8px 14px; text-align:left; border-bottom:2px solid #ddd;"
        "background:#f7f7f7; font-weight:600; white-space:nowrap;"
    )
    td_style = "padding:7px 14px; border-bottom:1px solid #eee; white-space:nowrap;"
    table_style = "border-collapse:collapse; width:100%; font-size:0.9rem;"

    parts = [f'<table style="{table_style}"><thead><tr>']
    for h in headers:
        parts.append(f'<th style="{th_style}">{_html.escape(str(h))}</th>')
    parts.append("</tr></thead><tbody>")

    for i, row in enumerate(rows):
        bg = f"background-color:{highlight_color};" if i in highlight_rows else ""
        parts.append(f'<tr style="{bg}">')
        for cell in row:
            parts.append(f'<td style="{td_style}">{_html.escape(str(cell))}</td>')
        parts.append("</tr>")

    parts.append("</tbody></table>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.title("🔬 KD + Quantization")
    st.sidebar.caption("VIT SEM 5 · Machine Learning Project")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Overview",
         "🤖 Live Prediction",
         "📊 Comparison Metrics",
         "📈 Training Curves",
         "📖 Methodology"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Experiment settings**")

    summary = load_summary_json()
    if summary:
        st.sidebar.metric("Train examples", f"{summary['n_train']:,}")
        st.sidebar.metric("Val examples",   f"{summary['n_val']:,}")
        st.sidebar.metric("Epochs",         summary["epochs"])
        st.sidebar.markdown(
            f"α = `{summary['alpha']}`  ·  T = `{summary['temperature']}`"
        )
    else:
        st.sidebar.info("experiment_summary.json not found.")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Teacher: `textattack/bert-base-uncased-SST-2`  \n"
        "Student: `distilbert-base-uncased`  \n"
        "Dataset: GLUE SST-2"
    )
    return page


# ════════════════════════════════════════════════════════════════════════
# Pages
# ════════════════════════════════════════════════════════════════════════

def page_overview(df):
    st.title("Knowledge Distillation + Quantization")
    st.subheader("Making Transformer models smaller, faster, and nearly as accurate")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    if df is not None:
        m1_acc = df.loc[df["model"] == "M1", "accuracy"].iloc[0]
        m4     = df[df["model"] == "M4"].iloc[0]
        m5     = df[df["model"] == "M5"].iloc[0]
        m2_lat = df.loc[df["model"] == "M2", "latency_ms"].iloc[0]

        col1.metric("Teacher accuracy (M1)",   f"{m1_acc:.1%}")
        col2.metric("Best student acc (M4)",   f"{m4['accuracy']:.1%}",
                    delta=f"{m4['accuracy'] - m1_acc:+.1%} vs teacher")
        col3.metric("Size reduction (M4 vs M1)",
                    f"{df.loc[df['model']=='M1','size_mb'].iloc[0] / m4['size_mb']:.1f}×")
        col4.metric("Speed-up M4 vs M2",
                    f"{m2_lat / m4['latency_ms']:.1f}×",
                    delta="faster on CPU")
    else:
        st.warning("model_comparison.csv not found — place kdq_results/ in the project root.")

    st.markdown("---")
    st.markdown("### Five Experimental Configurations")

    table_rows = []
    for mid, reg in MODEL_REGISTRY.items():
        avail = is_model_available(mid)
        star  = " ★" if mid == "M4" else ""
        table_rows.append({
            "Model":       mid + star,
            "Description": reg["short"],
            "Precision":   reg["precision"],
            "Available":   "✅" if avail else "❌ missing",
        })
    _html_table(
        headers=["Model", "Description", "Precision", "Available"],
        rows=[[r["Model"], r["Description"], r["Precision"], r["Available"]]
              for r in table_rows],
    )

    if df is not None:
        st.markdown("### Measured Results (from Colab experiment)")
        rows_data = []
        for _, row in df.iterrows():
            highlight = row["model"] == "M4"
            rows_data.append({
                "highlight": highlight,
                "cells": [
                    row["model"],
                    row["label"],
                    row["precision"],
                    f"{row['accuracy']:.4f}",
                    f"{row['f1_macro']:.4f}",
                    f"{row['size_mb']:.1f}",
                    f"{row['latency_ms']:.0f}",
                ],
            })
        _html_table(
            headers=["Model", "Description", "Prec.", "Accuracy",
                     "F1 Macro", "Size (MB)", "Latency (ms)"],
            rows=[r["cells"] for r in rows_data],
            highlight_rows=[i for i, r in enumerate(rows_data) if r["highlight"]],
            highlight_color="#fff3cd",
        )
        st.caption("★ M4 highlighted — best measured accuracy/size/speed trade-off among students.")


def page_prediction():
    st.title("🤖 Live Sentiment Prediction")
    st.markdown(
        "Type a sentence and choose a model. "
        "Prediction and inference time are measured live — no cached results."
    )
    st.markdown("---")

    # ── Session state initialisation ──────────────────────────────────
    if "pred_text" not in st.session_state:
        st.session_state.pred_text = ""
    if "pred_result" not in st.session_state:
        st.session_state.pred_result = None   # holds last result dict

    # ── Model selector ────────────────────────────────────────────────
    options = {}
    for mid, reg in MODEL_REGISTRY.items():
        avail  = is_model_available(mid)
        suffix = "" if avail else " (unavailable)"
        options[mid] = f"{mid} — {reg['short']}{suffix}"

    available_ids = [mid for mid in MODEL_REGISTRY if is_model_available(mid)]

    if not available_ids:
        st.error(
            "No model checkpoints found.  \n"
            f"Expected: `{MODELS_DIR}`  \n"
            "Copy the `kdq_results/` folder from your Colab output into the project root."
        )
        return

    col_sel, col_info = st.columns([2, 3])

    with col_sel:
        selected = st.selectbox(
            "Select model",
            options=list(options.keys()),
            format_func=lambda m: options[m],
        )
        model_ok = is_model_available(selected)
        if not model_ok:
            st.warning(f"**{selected}** checkpoint is not available locally.")

    with col_info:
        reg = MODEL_REGISTRY[selected]
        st.info(
            f"**{reg['label']}**  \n"
            f"Precision: `{reg['precision']}`  \n"
            f"{reg['desc']}"
        )

    st.markdown("---")

    # ── Example sentence buttons ──────────────────────────────────────
    # Clicking a button stores the sentence in session_state so it
    # survives the rerun that evaluates the Predict button.
    st.markdown("**Try an example or type your own:**")
    examples = [
        "The film was an absolute masterpiece.",
        "I found the movie incredibly boring and predictable.",
        "An emotionally gripping story with brilliant performances.",
        "The script was weak and the acting was unconvincing.",
        "A delightful surprise — witty, warm and wonderfully directed.",
    ]
    ex_col1, ex_col2 = st.columns(2)
    for i, ex in enumerate(examples):
        btn_col = ex_col1 if i % 2 == 0 else ex_col2
        label   = f'"{ex[:52]}…"' if len(ex) > 55 else f'"{ex}"'
        if btn_col.button(label, key=f"ex_{i}", use_container_width=True):
            st.session_state.pred_text   = ex
            st.session_state.pred_result = None   # clear previous result

    # ── Text input ────────────────────────────────────────────────────
    text_input = st.text_area(
        "Or enter your own sentence:",
        value=st.session_state.pred_text,
        height=90,
        key="pred_text_area",
        placeholder="Type a movie review sentence here…",
    )
    # Sync text area edits back to session_state
    st.session_state.pred_text = text_input

    # ── Predict button ────────────────────────────────────────────────
    predict_btn = st.button(
        "Predict sentiment",
        type="primary",
        disabled=not model_ok or not text_input.strip(),
    )

    if predict_btn and text_input.strip():
        # Clear previous result while loading
        st.session_state.pred_result = None
        with st.spinner(f"Loading {selected} and running inference…"):
            tokenizer = load_tokenizer()
            model, err = load_model(selected)

        if err:
            st.error(
                f"**Failed to load {selected}**  \n"
                f"`{err}`  \n"
                f"Checkpoint path: `{MODEL_REGISTRY[selected]['ckpt']}`"
            )
        else:
            try:
                pred_label, confidence, elapsed_ms = run_inference(
                    model, tokenizer, text_input.strip()
                )
                st.session_state.pred_result = {
                    "model":      selected,
                    "text":       text_input.strip(),
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "elapsed_ms": elapsed_ms,
                }
            except Exception as exc:
                import traceback as _tb
                st.error(
                    f"**Inference error for {selected}**  \n"
                    f"`{type(exc).__name__}: {exc}`"
                )
                st.code(_tb.format_exc(), language="python")

    # ── Display result ────────────────────────────────────────────────
    result = st.session_state.pred_result
    if result and result["model"] == selected:
        pred_label  = result["pred_label"]
        confidence  = result["confidence"]
        elapsed_ms  = result["elapsed_ms"]
        label_text  = LABEL_NAMES[pred_label]
        label_color = LABEL_COLORS[pred_label]

        st.markdown("---")
        res_col1, res_col2, res_col3 = st.columns(3)

        res_col1.markdown(
            f"<div style='text-align:center; font-size:2.2rem; "
            f"color:{label_color}; font-weight:bold; padding:8px'>"
            f"{label_text}</div>",
            unsafe_allow_html=True,
        )
        res_col2.metric("Confidence",      f"{confidence:.1%}")
        res_col3.metric("Inference time",  f"{elapsed_ms:.1f} ms")

        st.progress(float(confidence))

        prob_col1, prob_col2 = st.columns(2)
        neg_p = (1 - confidence) if pred_label == 1 else confidence
        pos_p = confidence       if pred_label == 1 else (1 - confidence)
        prob_col1.metric("Negative probability", f"{neg_p:.1%}")
        prob_col2.metric("Positive probability",  f"{pos_p:.1%}")

        st.caption(
            f"Model: **{selected}** · Precision: `{MODEL_REGISTRY[selected]['precision']}` "
            f"· Device: CPU · Input: {len(result['text'].split())} words"
        )


def page_comparison(df):
    st.title("📊 Model Comparison Metrics")

    if df is None:
        st.error(
            "model_comparison.csv not found.  \n"
            f"Expected path: `{RESULTS_DIR / 'model_comparison.csv'}`  \n"
            "Copy `kdq_results/` from your Colab output into the project root."
        )
        return

    st.markdown(
        "All metrics were measured during the Colab experiment on the same "
        "400-example validation subset. Latency is CPU-only (batch=32)."
    )

    # ── Main comparison chart ────────────────────────────────────────────
    png_path = RESULTS_DIR / "model_comparison.png"
    if png_path.exists():
        st.image(str(png_path), use_container_width=True,
                 caption="Left: Accuracy · Middle: Model size (MB) · Right: CPU latency (ms/batch)")
    else:
        st.warning("model_comparison.png not found — generating inline chart.")
        _render_inline_charts(df)

    st.markdown("---")
    st.markdown("### Detailed numbers")

    rows_data = []
    for _, row in df.iterrows():
        highlight = row["model"] == "M4"
        rows_data.append({
            "highlight": highlight,
            "cells": [
                row["model"],
                row["label"],
                row["precision"],
                f"{row['accuracy']:.4f}",
                f"{row['f1_macro']:.4f}",
                f"{row['size_mb']:.1f}",
                f"{row['latency_ms']:.0f}",
                f"{row['accuracy_vs_m1']:+.4f}",
                f"{row['compression_vs_m2']:.2f}×",
            ],
        })
    _html_table(
        headers=["Model", "Description", "Prec.", "Accuracy", "F1 Macro",
                 "Size MB", "Latency ms", "Δ Acc vs M1", "Compression vs M2"],
        rows=[r["cells"] for r in rows_data],
        highlight_rows=[i for i, r in enumerate(rows_data) if r["highlight"]],
        highlight_color="#fff3cd",
    )

    st.markdown("---")
    st.markdown("### Key findings")

    m1_acc = df.loc[df["model"] == "M1", "accuracy"].iloc[0]
    m4     = df[df["model"] == "M4"].iloc[0]
    m2_sz  = df.loc[df["model"] == "M2", "size_mb"].iloc[0]
    m1_sz  = df.loc[df["model"] == "M1", "size_mb"].iloc[0]
    m2_lat = df.loc[df["model"] == "M2", "latency_ms"].iloc[0]

    findings = [
        f"**Accuracy gap** — M4 achieves {m4['accuracy']:.1%} vs teacher {m1_acc:.1%} "
        f"({m4['accuracy'] - m1_acc:+.1%}), a gap of less than {abs(m4['accuracy'] - m1_acc):.1%}.",
        f"**Size** — M4 is {m1_sz / m4['size_mb']:.1f}× smaller than M1 "
        f"and {m2_sz / m4['size_mb']:.1f}× smaller than M2/M3.",
        f"**Speed** — M4 is {m2_lat / m4['latency_ms']:.2f}× faster on CPU than M2 "
        f"and {df.loc[df['model']=='M1','latency_ms'].iloc[0] / m4['latency_ms']:.1f}× "
        f"faster than M1.",
        "**M4 vs M5** — M4 (quantized baseline) achieved equal or better accuracy "
        "than M5 (quantized distilled) on this subset, showing quantization benefits "
        "are robust regardless of the distillation step.",
    ]
    for f in findings:
        st.markdown(f"- {f}")


def _render_inline_charts(df: pd.DataFrame):
    """Fallback: render comparison charts inline using Streamlit's native charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
    models = df["model"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Model Comparison", fontsize=13, fontweight="bold")

    for ax, col, title, ylabel in [
        (axes[0], "accuracy",   "Accuracy",        "Accuracy"),
        (axes[1], "size_mb",    "Model Size (MB)", "Size (MB)"),
        (axes[2], "latency_ms", "CPU Latency (ms/batch)", "ms"),
    ]:
        bars = ax.bar(models, df[col], color=colors, edgecolor="black", width=0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}" if col == "accuracy" else f"{val:.0f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def page_training_curves():
    st.title("📈 Training Curves")
    st.markdown("Loss and validation accuracy per epoch for M2 (baseline) and M3 (distilled).")

    # Try static image first
    png_path = RESULTS_DIR / "training_curves.png"
    if png_path.exists():
        st.image(str(png_path), use_container_width=True,
                 caption="Blue = M2 Supervised Baseline · Red = M3 Knowledge Distilled · "
                          "Dashed = M1 teacher accuracy")
        st.markdown("---")

    # Always render interactive version from JSON too
    summary = load_summary_json()
    if summary is None:
        st.warning("experiment_summary.json not found.")
        return

    m2h = summary.get("m2_training_history", [])
    m3h = summary.get("m3_training_history", [])

    if not m2h or not m3h:
        st.warning("Training history not found in experiment_summary.json.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs    = [h["epoch"] for h in m2h]
    m1_acc    = next(
        (m["accuracy"] for m in summary.get("models", []) if m["model"] == "M1"), None
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("M2 vs M3 Training — Actual Colab Results",
                 fontsize=12, fontweight="bold")

    # Loss
    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in m2h], "b-o",
            label="M2 baseline", linewidth=2)
    ax.plot(epochs, [h["train_loss"] for h in m3h], "r-o",
            label="M3 distilled", linewidth=2)
    ax.set_title("Training Loss per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for i, (l2, l3) in enumerate(
        zip([h["train_loss"] for h in m2h], [h["train_loss"] for h in m3h])
    ):
        ax.annotate(f"{l2:.3f}", (epochs[i], l2), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="blue")
        ax.annotate(f"{l3:.3f}", (epochs[i], l3), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color="red")

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, [h["accuracy"] for h in m2h], "b-o",
            label="M2 baseline", linewidth=2)
    ax.plot(epochs, [h["accuracy"] for h in m3h], "r-o",
            label="M3 distilled", linewidth=2)
    if m1_acc:
        ax.axhline(m1_acc, color="gray", linestyle="--", linewidth=1.5,
                   label=f"M1 teacher ({m1_acc:.3f})")
    ax.set_title("Validation Accuracy per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.80, 0.96)
    for i, (a2, a3) in enumerate(
        zip([h["accuracy"] for h in m2h], [h["accuracy"] for h in m3h])
    ):
        ax.annotate(f"{a2:.3f}", (epochs[i], a2), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="blue")
        ax.annotate(f"{a3:.3f}", (epochs[i], a3), textcoords="offset points",
                    xytext=(5, -12), fontsize=8, color="red")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Epoch-by-epoch table
    st.markdown("### Epoch-by-epoch detail")
    epoch_rows = []
    for h2, h3 in zip(m2h, m3h):
        epoch_rows.append([
            str(h2["epoch"]),
            f"{h2['train_loss']:.4f}",
            f"{h2['accuracy']:.4f}",
            f"{h3['train_loss']:.4f}",
            f"{h3['accuracy']:.4f}",
        ])
    _html_table(
        headers=["Epoch", "M2 Train Loss", "M2 Val Acc",
                 "M3 Train Loss", "M3 Val Acc"],
        rows=epoch_rows,
    )


def page_methodology():
    st.title("📖 Methodology")

    st.markdown("## Knowledge Distillation")
    st.markdown(
        """
Knowledge distillation transfers learned behaviour from a large, accurate
**teacher** model to a smaller **student** model by training the student on
a combination of two signals:

1. **Hard labels** — the correct SST-2 class (0 or 1), via standard cross-entropy loss.
2. **Soft labels** — the teacher's probability distribution over classes, made
   "softer" by a temperature parameter T > 1.  Soft labels carry information
   about the relative similarity of classes that hard labels do not.

The combined loss is:

$$\\mathcal{L} = \\alpha \\cdot T^2 \\cdot D_{KL}\\!\\left(\\sigma\\!\\left(\\frac{z_s}{T}\\right) \\| \\sigma\\!\\left(\\frac{z_t}{T}\\right)\\right) + (1 - \\alpha) \\cdot \\mathcal{L}_{CE}$$

where $z_s$ and $z_t$ are the student and teacher logits, $\\sigma$ is softmax,
$T$ is temperature, and $\\alpha$ balances the two terms.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Temperature (T)", "2.0", help="Higher T → softer probability distribution")
    col2.metric("Alpha (α)", "0.5",       help="Equal weight to KL and CE losses")
    col3.metric("Training subset", "1,500", help="Stratified SST-2 sample")

    st.markdown("---")
    st.markdown("## INT8 Dynamic Quantization")
    st.markdown(
        """
Post-training **dynamic quantization** converts the FP32 weights of
`nn.Linear` layers to INT8 at model-save time, and quantizes activations
dynamically at runtime.

| Property | FP32 (M2/M3) | INT8 (M4/M5) |
|----------|-------------|-------------|
| Weight storage | 4 bytes/param | 1 byte/param |
| Inference device | GPU or CPU | CPU only |
| PyTorch API | — | `torch.quantization.quantize_dynamic` |

Measured compression: **2.81×** smaller than the FP32 DistilBERT models
(255 MB → 91 MB).  CPU latency improved from ~1,366 ms to ~1,199 ms per batch.
        """
    )

    st.markdown("---")
    st.markdown("## Why M4 is the best student trade-off")
    st.markdown(
        """
Based on the measured results from the Colab experiment:

| Model | Accuracy | Size | Latency |
|-------|----------|------|---------|
| M1 (teacher) | 93.5% | 418 MB | 2,826 ms |
| M2 (FP32 baseline) | 87.3% | 255 MB | 1,366 ms |
| M3 (FP32 distilled) | 87.0% | 255 MB | 1,467 ms |
| **M4 (INT8 baseline)** | **87.0%** | **91 MB** | **1,199 ms** |
| M5 (INT8 distilled) | 86.0% | 91 MB | 1,203 ms |

M4 matches M3's accuracy while being **2.81× smaller** than M2/M3 and
**14% faster** on CPU. M5 is marginally slower and 1 percentage point less
accurate than M4 on this validation subset, so M4 represents the best
measured compression-accuracy trade-off among the student models.

> Note: these results are from a 1,500-example training subset.
> Training on the full SST-2 set (67,349 examples) would improve all student models.
        """
    )

    st.markdown("---")
    st.markdown("## Experiment setup")
    summary = load_summary_json()
    if summary:
        setup_rows = [
            ["Dataset",               "GLUE SST-2"],
            ["Training examples",     str(summary["n_train"])],
            ["Validation examples",   str(summary["n_val"])],
            ["Epochs",                str(summary["epochs"])],
            ["Learning rate",         "2e-5"],
            ["Batch size",            "16"],
            ["Mixed precision",       "FP16 (AMP) on T4 GPU"],
            ["Distillation α",        str(summary["alpha"])],
            ["Distillation T",        str(summary["temperature"])],
            ["Random seed",           str(summary["seed"])],
            ["Teacher model",         "textattack/bert-base-uncased-SST-2"],
            ["Student model",         "distilbert-base-uncased"],
            ["Quantization",          "torch.quantization.quantize_dynamic (INT8, Linear)"],
        ]
        _html_table(headers=["Parameter", "Value"], rows=setup_rows)


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    df      = load_results_csv()
    page    = render_sidebar()

    if page == "🏠 Overview":
        page_overview(df)
    elif page == "🤖 Live Prediction":
        page_prediction()
    elif page == "📊 Comparison Metrics":
        page_comparison(df)
    elif page == "📈 Training Curves":
        page_training_curves()
    elif page == "📖 Methodology":
        page_methodology()


if __name__ == "__main__":
    main()
