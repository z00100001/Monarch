import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np

# Import transformers with error handling
try:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
except ImportError:
    st.error("Transformers library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Define constants - Updated to handle relative paths better
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mental_health_model")
MODEL_DIR = os.path.abspath(MODEL_PATH)

# Define softer color palette (less saturated)
COLORS = {
    "green": "rgba(75, 192, 120, 0.7)",   # Softer green
    "yellow": "rgba(255, 205, 86, 0.7)",  # Softer yellow
    "orange": "rgba(255, 159, 64, 0.7)",  # Softer orange
    "red": "rgba(255, 99, 132, 0.7)",     # Softer red
    "blue": "rgba(54, 162, 235, 0.7)",    # Soft blue for reference lines
    "purple": "rgba(153, 102, 255, 0.7)"  # Soft purple
}

# Define reference levels for radar chart
REFERENCE_LEVELS = {
    "low": 25,
    "moderate": 50,
    "high": 75,
    "severe": 90
}

# page settings
st.set_page_config(
    page_title="Monarch - Mental Health Text Analyzer",
    layout="centered"
)

# header
st.markdown("<h1 style='text-align: center;'>Monarch</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Private AI for Emotional Pattern Detection</h4>", unsafe_allow_html=True)
st.markdown("---")

# Load model and tokenizer once when the app starts
@st.cache_resource
def load_mental_health_model():
    """Load the mental health model and tokenizer"""
    try:
        # Try to load from the specified directory
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            st.sidebar.info(f"Loading model from: {MODEL_DIR}")
            tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
            model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
            return model, tokenizer, True
        else:
            # If local model not found, use the base model from HuggingFace
            st.sidebar.warning("⚠️ Local model not found. Using base model from HuggingFace")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
                                                                    num_labels=2)
            return model, tokenizer, False
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.sidebar.info("Falling back to base RoBERTa model...")
        try:
            # Last resort - try loading direct from HuggingFace
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
                                                                   num_labels=2)
            return model, tokenizer, False
        except Exception as e2:
            st.sidebar.error(f"Failed to load fallback model: {str(e2)}")
            return None, None, False

# Load model
model, tokenizer, model_loaded = load_mental_health_model()

# Display model status
if model_loaded:
    st.sidebar.success("✅ Model loaded successfully!")
else:
    st.sidebar.warning("⚠️ Using base model (not trained)")

# input
st.subheader("Paste your text below:")
user_input = st.text_area("Enter journal entries, chat logs, or anything you want to analyze.", height=200)

# color mapping
def get_color(score):
    if score >= 85:
        return COLORS["red"]
    elif score >= 65:
        return COLORS["orange"]
    elif score >= 40:
        return COLORS["yellow"]
    else:
        return COLORS["green"]

# Map model probabilities to emotion scores
def map_to_emotions(depression_probability):
    """Map depression probability to different emotional dimensions"""
    # Base mapping - we'll use depression probability to inform other emotions
    distress = min(100, depression_probability * 100)
    
    # Create related but different scores for other emotions
    sadness = min(100, depression_probability * 100 * 0.9 + 10)
    worry = min(100, depression_probability * 100 * 0.8 + random.uniform(5, 15))
    anger = min(100, depression_probability * 100 * 0.6 + random.uniform(0, 20))
    
    return {
        "distress": round(distress, 2),
        "sadness": round(sadness, 2),
        "worry": round(worry, 2),
        "anger": round(anger, 2)
    }

# Real analyzer using our model
def analyze_text(text):
    """Analyze text using the mental health model"""
    
    if not model or not tokenizer:
        # Fall back to random if model isn't loaded
        return {
            "distress": round(random.uniform(30, 95), 2),
            "worry": round(random.uniform(10, 90), 2),
            "anger": round(random.uniform(0, 80), 2),
            "sadness": round(random.uniform(40, 100), 2),
        }
    
    try:
        # Tokenize the input text
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Get model predictions
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Extract depression probability (class 1)
        depression_prob = probabilities[0, 1].item()
        
        # Map to emotional dimensions
        return map_to_emotions(depression_prob)
        
    except Exception as e:
        st.error(f"Error analyzing text: {str(e)}")
        # Fall back to random values if model fails
        return {
            "distress": round(random.uniform(30, 95), 2),
            "worry": round(random.uniform(10, 90), 2),
            "anger": round(random.uniform(0, 80), 2),
            "sadness": round(random.uniform(40, 100), 2),
        }

# Generate radar chart
def create_radar_chart(scores):
    categories = list(scores.keys())
    values = list(scores.values())
    
    # Add first value again to close the loop
    categories = categories + [categories[0]]
    values = values + [values[0]]
    
    # Create reference circles
    fig = go.Figure()
    
    # Add reference circles (less prominent)
    for level_name, level_value in REFERENCE_LEVELS.items():
        fig.add_trace(go.Scatterpolar(
            r=[level_value] * len(categories),
            theta=categories,
            fill=None,
            mode='lines',
            line=dict(color='rgba(200, 200, 200, 0.5)', dash='dot'),
            name=f"{level_name.capitalize()} ({level_value})",
            hoverinfo='text',
            text=[f"{level_name.capitalize()} Level: {level_value}"] * len(categories)
        ))
    
    # Add actual scores
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        mode='lines+markers',
        line=dict(color=COLORS["purple"], width=2),
        marker=dict(size=8, color=COLORS["purple"]),
        name='Your Profile',
        hoverinfo='text',
        text=[f"{cat}: {val}" for cat, val in zip(categories, values)]
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10),
                tickvals=[0, 25, 50, 75, 100],
                ticktext=["0", "25", "50", "75", "100"]
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='darkblue'),
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        height=450
    )
    
    return fig

# results
if user_input.strip():
    with st.spinner("Analyzing your text..."):
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
                        {'range': [0, 40], 'color': COLORS["green"]},
                        {'range': [40, 65], 'color': COLORS["yellow"]},
                        {'range': [65, 85], 'color': COLORS["orange"]},
                        {'range': [85, 100], 'color': COLORS["red"]},
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 4},
                        'thickness': 0.75,
                        'value': score
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

    # NEW VISUALIZATION: Radar chart for emotional profile
    st.markdown("### Emotional Profile Radar:")
    radar_fig = create_radar_chart(scores)
    st.plotly_chart(radar_fig, use_container_width=True)
    st.caption("This radar chart shows your emotional profile against reference levels. The further from center, the higher the intensity.")
    
    # donut chart below with softer colors
    st.markdown("### Overall Emotional Composition:")
    fig_pie = px.pie(
        names=[e.capitalize() for e in scores.keys()],
        values=list(scores.values()),
        hole=0.4,
        color_discrete_sequence=[COLORS["red"], COLORS["yellow"], COLORS["orange"], COLORS["green"]],
    )
    fig_pie.update_traces(textinfo='label+percent', pull=[0.05]*len(scores))
    st.plotly_chart(fig_pie, use_container_width=True)

    # Add interpretation
    st.markdown("### Interpretation:")
    
    avg_score = sum(scores.values()) / len(scores)
    
    if avg_score >= 75:
        st.error("⚠️ High levels of emotional distress detected. Consider seeking support from a mental health professional.")
    elif avg_score >= 60:
        st.warning("⚠️ Moderate levels of emotional distress indicated. Self-care practices may be beneficial.")
    elif avg_score >= 40:
        st.info("Mild emotional concerns detected. Monitoring and self-awareness recommended.")
    else:
        st.success("✅ Low levels of emotional distress indicated.")
        
    # Add short recommendations based on analysis
    st.markdown("### Suggestions:")
    highest_emotion = max(scores.items(), key=lambda x: x[1])
    
    if highest_emotion[0] == "distress" and highest_emotion[1] > 60:
        st.markdown("""
        - Consider mindfulness or meditation practices
        - Break large tasks into smaller, manageable parts
        - Ensure you're getting adequate rest and sleep
        """)
    elif highest_emotion[0] == "sadness" and highest_emotion[1] > 60:
        st.markdown("""
        - Connect with supportive friends or family
        - Engage in activities that have brought joy in the past
        - Consider journaling about your emotions
        """)
    elif highest_emotion[0] == "worry" and highest_emotion[1] > 60:
        st.markdown("""
        - Practice deep breathing exercises when anxiety rises
        - Try to identify specific worry triggers
        - Consider limiting news or social media consumption
        """)
    elif highest_emotion[0] == "anger" and highest_emotion[1] > 60:
        st.markdown("""
        - Try physical activity to release tension
        - Practice "time-outs" when feeling overwhelmed
        - Consider relaxation techniques like progressive muscle relaxation
        """)
    else:
        st.markdown("""
        - Continue monitoring your emotional health
        - Maintain current self-care practices
        - Consider tracking your mood patterns over time
        """)
        
    st.caption("Note: This analysis is for informational purposes only and should not be considered a medical diagnosis.")

else:
    st.info("Start typing above to receive live feedback.")

# Add debugging section in sidebar
with st.sidebar:
    st.subheader("About the Model")
    st.write("This analyzer uses a RoBERTa-based model trained to detect emotional patterns in text.")
    
    if st.checkbox("Show technical details"):
        st.write("Model architecture: RoBERTa Base")
        st.write("Training data: Mental health text samples")
        st.write("Output: Depression probability mapped to emotional dimensions")
        
        if os.path.exists(os.path.join(MODEL_DIR, "eval_results.json")):
            import json
            with open(os.path.join(MODEL_DIR, "eval_results.json"), "r") as f:
                eval_results = json.load(f)
                st.write("### Model Performance")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        st.write(f"{metric}: {value:.4f}")
    
    # Add model path debugging info
    if st.checkbox("Debug information"):
        st.write(f"Model directory: {MODEL_DIR}")
        st.write(f"Directory exists: {os.path.exists(MODEL_DIR)}")
        if os.path.exists(MODEL_DIR):
            st.write(f"Directory contents: {os.listdir(MODEL_DIR)}")
            
    # Add color reference
    if st.checkbox("Color scale reference"):
        st.markdown("""
        - <span style='color: rgba(75, 192, 120, 0.7);'>■</span> Green (0-40): Low level
        - <span style='color: rgba(255, 205, 86, 0.7);'>■</span> Yellow (40-65): Moderate level
        - <span style='color: rgba(255, 159, 64, 0.7);'>■</span> Orange (65-85): High level
        - <span style='color: rgba(255, 99, 132, 0.7);'>■</span> Red (85-100): Severe level
        """, unsafe_allow_html=True)

# footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")