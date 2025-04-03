import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import random

# page settings
st.set_page_config(
    page_title="Monarch - Mental Health Text Analyzer",
    layout="centered"
)

st.markdown("<h1 style='text-align: center;'>Monarch</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Private AI for Emotional Pattern Detection</h4>", unsafe_allow_html=True)
st.markdown("---")

# text Input
st.subheader("Paste your text below:")
user_input = st.text_area("Enter journal entries, chat logs, or anything you want to analyze.", height=200)

def analyze_text(text):
    # placeholder... REAL DATA COMING SOON.
    fake_score = round(random.uniform(10, 95), 2)
    if fake_score >= 85:
        label, color = "High Risk", "red"
    elif fake_score >= 65:
        label, color = "Warning", "orange"
    elif fake_score >= 40:
        label, color = "Neutral", "yellow"
    else:
        label, color = "Low Concern", "green"

    return {
        "score": fake_score,
        "label": label,
        "color": color,
        "explanation": "This is placeholder data. Real analysis coming soon."
    }
if user_input.strip():
    result = analyze_text(user_input)

    st.markdown(f"### Emotional Risk Score: **{result['score']}%**")
    st.markdown(f"#### Status: `{result['label']}`")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['score'],
        title={'text': "Risk %"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': result['color']},
            'steps': [
                {'range': [0, 40], 'color': "green"},
                {'range': [40, 65], 'color': "yellow"},
                {'range': [65, 85], 'color': "orange"},
                {'range': [85, 100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': result['color'], 'width': 4},
                'thickness': 0.75,
                'value': result['score']
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Start typing above to receive live feedback.")

# footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")
