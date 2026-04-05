# app.py

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

import streamlit as st
from src.pipelines.inference_pipeline import predict

st.set_page_config(
    page_title="Customer Complaint Classifier",
    page_icon="📋",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        .main { background-color: #f8f9fb; }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        .title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
        .subtitle { font-size: 1rem; color: #6c757d; margin-bottom: 1.5rem; }
        .result-box {
            background-color: #ffffff;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1.5rem;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .category-label {
            font-size: 0.85rem;
            color: #6c757d;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .category-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #1a1a2e;
            margin-top: 0.3rem;
        }
        .confidence-label {
            font-size: 0.85rem;
            color: #6c757d;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .confidence-value {
            font-size: 1.4rem;
            font-weight: 700;
            color: #2563eb;
            margin-top: 0.3rem;
        }
        .footer {
            font-size: 0.78rem;
            color: #adb5bd;
            text-align: center;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">📋 Customer Complaint Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Automatically classify consumer complaints into financial product categories using Machine Learning.</div>', unsafe_allow_html=True)
st.divider()

# Category colors
category_colors = {
    "Checking or savings account":                        "#2563eb",
    "Money transfer, virtual currency, or money service": "#7c3aed",
    "Mortgage":                                           "#059669",
    "Student loan":                                       "#d97706",
    "Vehicle loan or lease":                              "#dc2626",
}

# Input
text = st.text_area(
    label="Enter a customer complaint:",
    placeholder="e.g. I have been charged twice for the same transaction on my checking account and the bank refuses to refund me.",
    height=180
)

col_btn, _ = st.columns([1, 3])
with col_btn:
    classify = st.button("Classify", type="primary", use_container_width=True)

if classify:
    if not text.strip():
        st.warning("Please enter a complaint before classifying.")
    else:
        with st.spinner("Analyzing complaint..."):
            results = predict([text])
            result  = results[0]

        label = result['predicted_label']
        score = result['confidence_score']
        color = category_colors.get(label, "#1a1a2e")

        st.markdown(f"""
            <div class="result-box">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                    <div>
                        <div class="category-label">Predicted Category</div>
                        <div class="category-value">
                            <span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{color}; margin-right:8px;"></span>
                            {label}
                        </div>
                    </div>
                    <div>
                        <div class="confidence-label">Confidence Score</div>
                        <div class="confidence-value">{score}</div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.success("Classification complete.")

# Footer
st.divider()
st.markdown("""
    <div class="footer">
        Model: Logistic Regression &nbsp;|&nbsp; Dataset: CFPB Consumer Complaints
        &nbsp;|&nbsp; Weighted F1: 0.8971 &nbsp;|&nbsp; 5 Categories &nbsp;|&nbsp; 245,228 complaints
    </div>
""", unsafe_allow_html=True)