"""
Brain Tumor MRI Classifier — Streamlit Interface
Run: cd app && streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from model import (
    CLASS_INFO,
    CLASS_NAMES,
    MODEL_NAME,
    predict,
    PredictionResult,
)


# ─────────────────────────── Page config ─────────────────────────

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────── CSS ─────────────────────────────────

st.markdown("""
<style>
    html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }
    .block-container { padding-top: 1.5rem; }

    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        border-left: 5px solid var(--card-color, #4C72B0);
    }
    .metric-card h4 {
        margin: 0 0 0.2rem 0;
        font-size: 0.78rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card p {
        margin: 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff;
    }

    .pred-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 999px;
        font-size: 1rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 0.5rem;
    }

    .topk-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 2px;
    }

    .info-box {
        background: #12122a;
        border: 1px solid #2e2e50;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 0.5rem;
        font-size: 0.88rem;
        color: #ccc;
        line-height: 1.6;
    }

    .section-header {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #888;
        margin: 1.2rem 0 0.5rem 0;
    }

    .disclaimer {
        background: #2a1a1a;
        border-left: 4px solid #c0392b;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        font-size: 0.8rem;
        color: #e0a0a0;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────── Sidebar ─────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 Brain Tumor MRI")
    st.markdown("**AI Diagnosis Assistant**")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "Upload a brain MRI scan. The model predicts the tumor type "
        "and highlights the region it focused on using **Grad-CAM**."
    )

    st.markdown("### Model")
    st.markdown(f"""
- **Architecture:** ViT-Base/16
- **Backbone:** `{MODEL_NAME}`
- **Fine-tuned on:** Brain Tumor MRI
- **Classes:** {len(CLASS_NAMES)}
- **Input size:** 224 × 224
    """)

    st.markdown("### Classes")
    for cls, info in CLASS_INFO.items():
        color = info["color"]
        st.markdown(
            f'<span style="color:{color}; font-size:0.9rem;">● </span>'
            f'**{info["full_name"]}**',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">⚠️ For research use only. '
        'Not a substitute for professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────── Header ──────────────────────────────

st.markdown("# 🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI image to get an AI-powered diagnosis with visual explanation.")
st.markdown("---")


# ─────────────────────────── Upload ──────────────────────────────

uploaded_file = st.file_uploader(
    "Upload MRI scan (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a T1-weighted axial brain MRI image.",
)

if uploaded_file is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem; background:#1a1a2e;
                    border-radius:16px; border: 2px dashed #444; margin-top:2rem;">
            <div style="font-size:3.5rem;">🧠</div>
            <div style="font-size:1.1rem; color:#aaa; margin-top:0.8rem;">
                Drop your MRI scan here
            </div>
            <div style="font-size:0.8rem; color:#666; margin-top:0.4rem;">
                Supports JPG and PNG
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────── Inference ───────────────────────────

image = Image.open(uploaded_file)

with st.spinner("🔍 Analyzing MRI scan..."):
    try:
        result: PredictionResult = predict(image)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Inference failed: {e}")
        st.stop()


# ─────────────────────────── Layout ──────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")


# ── Left column: images ──────────────────────────────────────────

with left_col:

    st.markdown('<p class="section-header">Original MRI</p>', unsafe_allow_html=True)
    st.image(
    image.convert("RGB"),
    use_container_width=True,
    caption="Uploaded scan",
)

    st.markdown('<p class="section-header">Grad-CAM Explanation</p>', unsafe_allow_html=True)

    gradcam_rgb = cv2.cvtColor(result.gradcam_image, cv2.COLOR_BGR2RGB)
    st.image(
    gradcam_rgb,
    use_container_width=True,
    caption="Regions the model focused on (red = high attention)",
)


# ── Right column: results ────────────────────────────────────────

with right_col:

    info        = CLASS_INFO[result.predicted_class]
    pred_color  = info["color"]
    severity    = info["severity"]
    description = info["description"]

    # Prediction badge
    st.markdown('<p class="section-header">Diagnosis Result</p>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pred-badge" style="background:{pred_color};">'
        f'{info["full_name"]}</div>',
        unsafe_allow_html=True,
    )

    # Confidence card
    st.markdown(
        f'<div class="metric-card" style="--card-color:{pred_color};">'
        f'<h4>Confidence</h4><p>{result.confidence * 100:.2f}%</p></div>',
        unsafe_allow_html=True,
    )

    # Severity card
    severity_colors = {"None": "#27ae60", "Moderate": "#e67e22", "High": "#e74c3c"}
    sev_color = severity_colors.get(severity, "#888")
    st.markdown(
        f'<div class="metric-card" style="--card-color:{sev_color};">'
        f'<h4>Severity</h4><p style="color:{sev_color};">{severity}</p></div>',
        unsafe_allow_html=True,
    )

    # Description
    st.markdown(
        f'<div class="info-box">📋 {description}</div>',
        unsafe_allow_html=True,
    )

    # Top-K bars
    st.markdown('<p class="section-header">Top Predictions</p>', unsafe_allow_html=True)

    for item in result.top_k:
        cls_name   = item["class"]
        full_name  = item["full_name"]
        confidence = item["confidence"]
        color      = CLASS_INFO[cls_name]["color"]

        st.markdown(
            f'<div class="topk-label">'
            f'<span>{full_name}</span>'
            f'<span style="color:{color}; font-weight:600;">'
            f'{confidence * 100:.2f}%</span></div>',
            unsafe_allow_html=True,
        )
        st.progress(float(confidence))
        st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # Full probability table
    with st.expander("📊 Full probability breakdown"):
        table_data = {
            "Class": [CLASS_INFO[c]["full_name"] for c in CLASS_NAMES],
            "Probability": [
                f"{next(i['confidence'] for i in result.top_k if i['class'] == c) * 100:.4f}%"
                for c in CLASS_NAMES
            ],
        }
        st.table(table_data)

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        '⚠️ This result is AI-generated and intended for research purposes only. '
        'Always consult a qualified medical professional for diagnosis and treatment.'
        '</div>',
        unsafe_allow_html=True,
    )