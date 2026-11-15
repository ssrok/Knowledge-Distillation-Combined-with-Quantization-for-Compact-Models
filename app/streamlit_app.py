"""
streamlit_app.py
----------------
Interactive demo for all five experimental models.

Phase 1: placeholder — UI and inference logic implemented in Phase 7.

Run with:
    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Knowledge Distillation + Quantization Demo",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Knowledge Distillation + Quantization")
st.subheader("Sentiment Analysis — GLUE SST-2")

st.info(
    "This demo will allow interactive comparison of all five experimental "
    "models (M1–M5) once training is complete.\n\n"
    "**Current status:** Phase 1 — project initialised, models not yet trained."
)

st.markdown(
    """
| Model | Architecture | Precision | Strategy |
|-------|-------------|-----------|----------|
| M1 | BERT-base | FP32 | Fine-tuned teacher |
| M2 | DistilBERT | FP32 | Supervised baseline |
| M3 | DistilBERT | FP32 | Knowledge distillation |
| M4 | DistilBERT | INT8 | Quantized baseline |
| M5 | DistilBERT | INT8 | Quantized distilled |
"""
)

text_input = st.text_area("Enter a sentence to classify:", height=100)
model_choice = st.selectbox("Select model:", ["M1", "M2", "M3", "M4", "M5"])

if st.button("Classify"):
    st.warning("Models are not trained yet. Inference available after Phase 5.")
