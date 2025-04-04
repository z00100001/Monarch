import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import random

# page settings
st.set_page_config(
    page_title="Monarch - Mental Health Text Analyzer",
    layout="centered"
)

# header
st.markdown("<h1 style='text-align: center;'>Monarch</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Private AI for Emotional Pattern Detection</h4>", unsafe_allow_html=True)
st.markdown("---")

# input
st.subheader("Paste your text below:")
user_input = st.text_area("Enter journal entries, chat logs, or anything you want to analyze.", height=200)

# color mapping
def get_color(score):
    if score >= 85:
        return "red"
    elif score >= 65:
        return "orange"
    elif score >= 40:
        return "yellow"
    else:
        return "green"

# placeholder analyzer
def analyze_text(text):
    return {
        "depression": round(random.uniform(30, 95), 2),
        "anxiety": round(random.uniform(10, 90), 2),
        "anger": round(random.uniform(0, 80), 2),
        "sadness": round(random.uniform(40, 100), 2)
    }

# results
if user_input.strip():
    scores = analyze_text(user_input)
    st.markdown("### Emotional Analysis Results:")

    # 4 horizontal gauges
    cols = st.columns(4)
    for i, (emotion, score) in enumerate(scores.items()):
        with cols[i]:
            color = get_color(score)
            label = emotion.capitalize()

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"{label}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 40], 'color': "green"},
                        {'range': [40, 65], 'color': "yellow"},
                        {'range': [65, 85], 'color': "orange"},
                        {'range': [85, 100], 'color': "red"},
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

    # donut chart below
    st.markdown("### Overall Emotional Composition:")
    fig_pie = px.pie(
        names=[e.capitalize() for e in scores.keys()],
        values=list(scores.values()),
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    fig_pie.update_traces(textinfo='label+percent', pull=[0.05]*len(scores))
    st.plotly_chart(fig_pie, use_container_width=True)

else:
    st.info("Start typing above to receive live feedback.")

# footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")
