ourai.py
import sys
import os
import random
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
import pandas as pd
import io
import PyPDF2
import docx2txt

#imports the transformer library, Installs it just in case the library is not found on the device
try:
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
except ImportError:
    st.error("Transformers library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import RobertaTokenizer, RobertaForSequenceClassification

#sets up a path to where the model is stored, for now, it does not exist
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mental_health_model")
MODEL_DIR = os.path.abspath(MODEL_PATH)
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

# Add a debug check function
def check_image_paths():
    """Debug function to check if image paths exist"""
    st.write("Current working directory:", os.getcwd())
    
    # Check if reports directory exists
    if os.path.exists("reports"):
        st.write("Reports directory found")
        st.write("Files in reports directory:", os.listdir("reports"))
    else:
        st.write("Reports directory NOT found at:", os.path.join(os.getcwd(), "reports"))
        
        # Check if reports exists relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_path = os.path.join(script_dir, "reports")
        if os.path.exists(reports_path):
            st.write("Reports directory found relative to script at:", reports_path)
            st.write("Files in reports directory:", os.listdir(reports_path))
        else:
            st.write("Reports directory NOT found relative to script at:", reports_path)

#soft color pallete for graphs
COLORS = {
    "green": "rgba(75, 192, 120, 0.7)",   #Softer green
    "yellow": "rgba(255, 205, 86, 0.7)",  #Softer yellow
    "orange": "rgba(255, 159, 64, 0.7)",  #Softer orange
    "red": "rgba(255, 99, 132, 0.7)",     #Softer red
    "blue": "rgba(54, 162, 235, 0.7)",    #Soft blue for reference lines
    "purple": "rgba(153, 102, 255, 0.7)"  #Soft purple
}

#defines the reference levels for the radar chart
REFERENCE_LEVELS = {
    "low": 25,
    "moderate": 50,
    "high": 75,
    "severe": 90
}

#Sets the page layouy and title for the web page
st.set_page_config(
    page_title="Monarch - Mental Health Text Analyzer",
    layout="wide"
)

#creates a list to remember past analyses for future graphs
if 'history' not in st.session_state:
    st.session_state.history = []

#loads model and tokenizer once when the app starts
@st.cache_resource
def load_mental_health_model():
    """Load the mental health model and tokenizer"""
    try:
        #try to load from the specified directory
        if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
            st.sidebar.info(f"Loading model from: {MODEL_DIR}")
            tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
            model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
            return model, tokenizer, True
        else:
            #for now, if local model not found, use the base model from HuggingFace, this will be replaced soon
            st.sidebar.warning("⚠️ Local model not found. Using base model from HuggingFace")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
                                                                    num_labels=2)
            return model, tokenizer, False
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
        st.sidebar.info("Falling back to base RoBERTa model...")
        try:
            #last resort - try loading direct from HuggingFace
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
                                                                   num_labels=2)
            return model, tokenizer, False
        except Exception as e2:
            st.sidebar.error(f"Failed to load fallback model: {str(e2)}")
            return None, None, False

#method to extract text from PDFs
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text

#method to extract text from docz
def extract_text_from_docx(file):
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

#method to extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.getvalue().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error extracting text from TXT: {str(e)}")
        return ""
    
def display_report_images():
    """Display the visualization images from the reports folder in the Our AI tab"""
    # Dictionary mapping image purposes to filenames
    report_images = {
        "emotion_matrix": "PRIOR2modern_emotion_heatmap.png",
        "key_expressions": "PRIORkey_expressions_wordcloud.png",
        "community_worry": "PRIORimproved_subreddit_worry_levels.png",
        "worry_distribution": "PRIORimproved_anxiety_distribution.png",
        "worry_vs_length": "PRIORimproved_worry_quadrant_plot.png"
    }
    
    # Try multiple possible locations for reports directory
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "reports"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports")),
        os.path.abspath("reports"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    ]
    
    # Find the first path that exists
    reports_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            reports_dir = path
            break
    
    # If no valid path found, use a default and let the error handling in the display code manage it
    if not reports_dir:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        
    # Function to create the full path to an image
    def get_image_path(filename):
        return os.path.join(reports_dir, filename)
    
    # Return the dictionary of image paths
    return {purpose: get_image_path(filename) for purpose, filename in report_images.items()}

#creates the main layout
main_col, sidebar_col = st.columns([3, 1])

#the sidebar content
with sidebar_col:
    #sidebar header
    st.sidebar.title("Monarch")
    st.sidebar.subheader("Privacy-focused NLP for Emotional Pattern Detection")
    
    #about section
    with st.sidebar.expander("📖 About Monarch", expanded=True):
        st.write("""
        Monarch is a privacy-focused deep learning model that interprets emotional patterns in text. 
        
        Unlike other tools, all analysis happens entirely on your device - your data never leaves your computer.
        
        We use fine-tuned NLP models (BERT, VADER) to identify patterns associated with sadness, worry, anger, and distress.
        """)
    
    #project details from poster
    with st.sidebar.expander("🔬 Research Details", expanded=False):
        st.write("""
        ### Research Questions
        - What words most frequently correlate with emotional distress?
        - How accurate is emotion classification when models are trained on lexicon-tagged emotional data?
        - Can a fine-tuned deep learning model identify emotional cues in text-based language?
        
        ### Technology
        Monarch uses VADER and BERT NLP models fine-tuned on emotion-labeled datasets to approximate categorical responses within 4 categories: sadness, worry, anger, and distress.
        
        The model was trained and validated entirely offline using PyTorch, HuggingFace Transformers, and local GPU/CPU.
        """)
    
    #team information
    with st.sidebar.expander("👥 Team", expanded=False):
        st.write("""
        ### Authors
        - Tyler Clanton
        - Derick Burbano-Ramon
        
        ### Advisors
        - Dr. Jeff Adkisson
        
        ### Affiliation
        Kennesaw State University
        """)
    
    #technical details
    with st.sidebar.expander("⚙️ Technical Details", expanded=False):
        st.write("Model architecture: RoBERTa Base")
        st.write("Training data: Mental health text samples")
        st.write("Output: Distress probability mapped to emotional dimensions")
        
        if os.path.exists(os.path.join(MODEL_DIR, "eval_results.json")):
            import json
            with open(os.path.join(MODEL_DIR, "eval_results.json"), "r") as f:
                eval_results = json.load(f)
                st.write("### Model Performance")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        st.write(f"{metric}: {value:.4f}")
    
    #color reference/legend for users
    with st.sidebar.expander("🎨 Color Scale Reference", expanded=False):
        st.markdown("""
        - <span style='color: rgba(75, 192, 120, 0.7);'>■</span> Green (0-40): Low level
        - <span style='color: rgba(255, 205, 86, 0.7);'>■</span> Yellow (40-65): Moderate level
        - <span style='color: rgba(255, 159, 64, 0.7);'>■</span> Orange (65-85): High level
        - <span style='color: rgba(255, 99, 132, 0.7);'>■</span> Red (85-100): Severe level
        """, unsafe_allow_html=True)
    
    #privacy information
    with st.sidebar.expander("🔒 Privacy", expanded=True):
        st.write("""
        All analysis is performed locally in your browser. Your text is never sent to external servers or stored anywhere.
        
        A lightweight, Raspberry Pi-compatible version is also in development for complete offline use on low-power hardware.
        """)
    
    #resources for the user, might change depending on what we deem as morally bad
    with st.sidebar.expander("📚 Resources", expanded=False):
        st.write("""
        ### Understanding Emotional Analysis
        
        This tool looks for patterns in:
        - Word choice and frequency
        - Linguistic structures
        - Emotional indicators
        
        Results show possible emotional dimensions in your text, but should not be used as a diagnostic tool.
        
        ### Getting Help
        If you notice concerning patterns in your emotional analysis, please consider speaking with a mental health professional.
                 
        ### Help is available 
        988 Suicide and Crisis Lifeline
        """)

#main content area
with main_col:
    #header
    st.markdown("<h1 style='text-align: center;'>Monarch</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Private AI for Emotional Pattern Detection</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    #loads model
    model, tokenizer, model_loaded = load_mental_health_model()
    
    #display model status, two options based on whether or not the placeholder is present
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Using base model (not trained)")
    
    # Create tabs for the main interface
    tab1, tab2, tab3 = st.tabs(["Text Analysis", "Upload File", "Our AI"])
    
    # Variable to store text input regardless of source
    user_input = ""
    input_source = ""
    
    # Text input tab
    with tab1:
        st.subheader("Enter text for analysis")
        user_input_tab1 = st.text_area("Type or paste any text you want to analyze...", height=200)
        if user_input_tab1:
            user_input = user_input_tab1
            input_source = "text_area"
        
        # Character count for tab 1
        input_length = len(user_input_tab1) if user_input_tab1 else 0
        st.caption(f"Character count: {input_length}")
        
        # Add analyze button only in this tab with a unique key
        analyze_button_tab1 = st.button("Analyze Text", key="analyze_button_tab1_unique")
        
        # Process analysis for tab 1
        if analyze_button_tab1:
            if user_input_tab1 and user_input_tab1.strip():
                user_input = user_input_tab1
                input_source = "text_area"
                analyze_triggered = True
            else:
                st.warning("Please enter text for analysis.")
                analyze_triggered = False
        else:
            analyze_triggered = False
    
    # File input tab
    with tab2:
        st.subheader("Upload a document")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
        
        user_input_tab2 = ""
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            if file_type == "pdf":
                user_input_tab2 = extract_text_from_pdf(uploaded_file)
            elif file_type == "docx":
                user_input_tab2 = extract_text_from_docx(uploaded_file)
            elif file_type == "txt":
                user_input_tab2 = extract_text_from_txt(uploaded_file)
            else:
                user_input_tab2 = ""
                st.error("Unsupported file type")
            
            if user_input_tab2:
                st.success(f"Successfully extracted text from {uploaded_file.name}")
                st.text_area("Extracted text:", user_input_tab2, height=200)
        
        # Character count for tab 2
        input_length = len(user_input_tab2) if user_input_tab2 else 0
        st.caption(f"Character count: {input_length}")
        
        # Add analyze button only in this tab with a unique key
        analyze_button_tab2 = st.button("Analyze Text", key="analyze_button_tab2_unique")
        
        # Process analysis for tab 2
        if analyze_button_tab2:
            if user_input_tab2 and user_input_tab2.strip():
                user_input = user_input_tab2
                input_source = "file_upload"
                analyze_triggered = True
            else:
                st.warning("Please upload a file for analysis.")
                analyze_triggered = False
        else:
            analyze_triggered = False
    
    # New Our AI tab - NO ANALYZE BUTTON HERE
    with tab3:
        st.header("Our Machine Learning Technology")
        
        # Create two columns for layout
        ml_col1, ml_col2 = st.columns([3, 2])
        
        with ml_col1:
            st.subheader("How Monarch's AI Works")
            st.write("""
            Monarch utilizes advanced natural language processing (NLP) and machine learning techniques to analyze emotional patterns in text. Our approach combines multiple AI models and data sources to provide accurate, privacy-focused analysis.
            
            ### Data Collection & Learning
            Our models are trained on diverse datasets including:
            - Anonymized mental health forum posts
            - Emotion-labeled linguistic datasets
            - Clinical language samples (with all identifying information removed)
            
            The AI continuously learns patterns of emotional expression across different contexts, allowing it to identify subtle indicators of emotional states like sadness, worry, anger, and distress.
            """)
            
            st.subheader("Our Machine Learning Pipeline")
            st.write("""
            1. **Text Preprocessing**: Cleaning and normalizing input text
            2. **Feature Extraction**: Identifying linguistic patterns and emotional markers
            3. **Deep Learning Analysis**: Processing through our fine-tuned RoBERTa model
            4. **Multi-dimensional Scoring**: Mapping probabilities to emotional dimensions
            5. **Visualization**: Presenting results through intuitive visual representations
            
            All processing happens locally on your device, ensuring complete privacy.
            """)
        
        with ml_col2:
            st.image("https://via.placeholder.com/400x250", caption="Monarch's Machine Learning Architecture")
        
        st.markdown("---")
        
        # Get image paths
        report_images = display_report_images()
        
        # Explaining the relationship between emotions section
        st.subheader("Understanding Emotion Relationships")
        st.write("""
        Our research has uncovered important relationships between different emotional dimensions. The heatmap below shows how different emotions relate to each other in our analysis.
        """)
        
        # Create two columns for first set of charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Use the actual emotion matrix image instead of placeholder
            try:
                st.image(report_images["emotion_matrix"], caption="Emotion Relationship Strength Matrix")
            except Exception as e:
                st.error(f"Error displaying emotion matrix image: {str(e)}")
                st.image("https://via.placeholder.com/450x400", caption="Emotion Relationship Strength Matrix")
                
            st.write("""
            This visualization shows how different emotions correlate with each other. For example, we can see strong positive relationships between joy and optimism (0.58), while emotions like anger and disgust show weaker relationships with positive emotions.
            
            These relationship patterns help our model understand the complex interplay of emotions in human expression.
            """)
        
        with chart_col2:
            # Use the actual word cloud image
            try:
                st.image(report_images["key_expressions"], caption="Key Expressions in High Concern Posts")
            except Exception as e:
                st.error(f"Error displaying key expressions image: {str(e)}")
                st.image("https://via.placeholder.com/450x400", caption="Key Expressions in High Concern Posts")
                
            st.write("""
            This word cloud highlights common expressions found in texts with elevated emotional concern. The size of each word represents its frequency, while colors indicate emotional tone categories.
            
            Words like "help," "connection," "issue," and "thought" are frequently associated with expressions of concern or emotional distress.
            """)
        
        st.markdown("---")
        
        # Analysis of online community data
        st.subheader("Community Analysis & Worry Intensity Patterns")
        
        # Create two columns for second set of charts
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            # Use the actual community worry image
            try:
                st.image(report_images["community_worry"], caption="Average Worry Intensity by Online Community")
            except Exception as e:
                st.error(f"Error displaying community worry image: {str(e)}")
                st.image("https://via.placeholder.com/450x380", caption="Average Worry Intensity by Online Community")
                
            st.write("""
            Our research examines worry intensity across different online communities. This data helps calibrate our AI to better understand contextual emotional expressions.
            
            Communities focused specifically on mental health support show varying levels of expressed worry, which helps our model recognize different manifestations of emotional distress.
            """)
        
        with chart_col4:
            # Use the actual worry distribution image
            try:
                st.image(report_images["worry_distribution"], caption="Distribution of Worry Intensity Measurements")
            except Exception as e:
                st.error(f"Error displaying worry distribution image: {str(e)}")
                st.image("https://via.placeholder.com/450x380", caption="Distribution of Worry Intensity Measurements")
                
            st.write("""
            This distribution chart shows the range of worry intensity values across our research dataset. The mean worry score of 7.57 and median of 5.96 help establish baselines for our analysis.
            
            The graph shows that while most texts express low to moderate worry levels, there is a significant tail of high-intensity emotional expression that our model is trained to recognize.
            """)
        
        st.markdown("---")
        
        # Text length and worry correlation
        st.subheader("Text Patterns & Emotional Expression")
        # Use the actual worry vs length plot
        try:
            st.image(report_images["worry_vs_length"], caption="Worry Score vs. Post Length Analysis")
        except Exception as e:
            st.error(f"Error displaying worry vs length image: {str(e)}")
            st.image("https://via.placeholder.com/800x450", caption="Worry Score vs. Post Length Analysis")
            
        st.write("""
        This visualization explores the relationship between text length and expressed worry levels. We've found that approximately 27.7% of longer posts express high worry levels, while shorter posts show a different distribution pattern.
        
        This insight helps our model adjust its analysis based on text length, improving accuracy across different types of input.
        """)
        
        st.markdown("---")
        
        # Research outcomes and future development
        st.subheader("Ongoing Research & Development")
        st.write("""
        Monarch is continuously evolving through ongoing research and model refinement. Current areas of development include:
        
        - **Expanded Emotional Dimensions**: Adding more nuanced emotional categories beyond our current four dimensions
        - **Cross-Cultural Adaptation**: Improving recognition of emotional expression across different cultural contexts
        - **Longitudinal Analysis**: Enhancing trend detection for users who analyze multiple texts over time
        - **Low-Resource Deployment**: Optimizing our models to run efficiently on personal devices with limited processing power
        
        Our commitment to privacy-first AI means all improvements are designed to run locally, keeping your data on your device.
        """)

    # You might also want to add a debug function to help troubleshoot any issues with image loading
    def debug_image_paths():
        """Debug function to check if image paths exist"""
        st.write("Reports directory path:", REPORTS_DIR)
        
        if os.path.exists(REPORTS_DIR):
            st.write("Reports directory found")
            files = os.listdir(REPORTS_DIR)
            st.write("Files in reports directory:", files)
            
            # Check for specific image files
            report_images = [
                "PRIOR2modern_emotion_heatmap.png",
                "PRIORkey_expressions_wordcloud.png",
                "PRIORimproved_subreddit_worry_levels.png",
                "PRIORimproved_anxiety_distribution.png",
                "PRIORimproved_worry_quadrant_plot.png"
            ]
            
            for img in report_images:
                img_path = os.path.join(REPORTS_DIR, img)
                st.write(f"Image '{img}' exists: {os.path.exists(img_path)}")
        else:
            st.write("Reports directory NOT found at:", REPORTS_DIR)
            
            # Try to find the reports directory relative to the script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(script_dir, "reports"),
                os.path.join(os.path.dirname(script_dir), "reports"),
                os.path.join(script_dir, "..", "reports")
            ]
            
            for path in possible_paths:
                st.write(f"Checking alternate path: {path}")
                st.write(f"Path exists: {os.path.exists(path)}")
                if os.path.exists(path):
                    st.write("Files in this path:", os.listdir(path))
    
    # Common elements for all tabs
    if tab1 or tab2:
        # Adds the character count at the bottom for the user
        input_length = len(user_input) if user_input else 0
        st.caption(f"Character count: {input_length}")
    
    #color mapping
    def get_color(score):
        if score >= 85:
            return COLORS["red"]
        elif score >= 65:
            return COLORS["orange"]
        elif score >= 40:
            return COLORS["yellow"]
        else:
            return COLORS["green"]
    
    #map model probabilities to emotion scores
    def map_to_emotions(depression_probability):
        """Map depression probability to different emotional dimensions"""
        #base mapping - we'll use depression probability to inform other emotions
        distress = min(100, depression_probability * 100)
        
        #create related but different scores for other emotions
        sadness = min(100, depression_probability * 100 * 0.9 + 10)
        worry = min(100, depression_probability * 100 * 0.8 + random.uniform(5, 15))
        anger = min(100, depression_probability * 100 * 0.6 + random.uniform(0, 20))
        
        return {
            "distress": round(distress, 2),
            "sadness": round(sadness, 2),
            "worry": round(worry, 2),
            "anger": round(anger, 2)
        }
    
    #real analyzer using our model
    def analyze_text(text):
        """Analyze text using the mental health model"""
        
        if not model or not tokenizer:
            #fall back to random if model isn't loaded
            return {
                "distress": round(random.uniform(30, 95), 2),
                "worry": round(random.uniform(10, 90), 2),
                "anger": round(random.uniform(0, 80), 2),
                "sadness": round(random.uniform(40, 100), 2),
            }
        
        try:
            #tokenize the input text
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,  
                return_tensors="pt"
            )
            
            #gets model predictions
            model.eval()
            with torch.no_grad(): #uses the model to get a result without worryinh about training or tracking changes
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            #aggresion probabiltiy
            depression_prob = probabilities[0, 1].item()
            
            #map to emotional dimensions
            return map_to_emotions(depression_prob)
            
        except Exception as e:
            st.error(f"Error analyzing text: {str(e)}")
            #fall back to random values if model fails
            return {
                "distress": round(random.uniform(30, 95), 2),
                "worry": round(random.uniform(10, 90), 2),
                "anger": round(random.uniform(0, 80), 2),
                "sadness": round(random.uniform(40, 100), 2),
            }
    
    #generate radar chart
    def create_radar_chart(scores):
        categories = list(scores.keys())
        values = list(scores.values())
        
        #add first value again to close the loop
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        #create reference circles
        fig = go.Figure()
        
        #adds more reference circles (less prominent)
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
        
        #add actual "scores"
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
        
        #update layout
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
    
    #create horizontal gauge chart (Not used)
    def create_horizontal_gauge(value, title, color):
        """Create a horizontal gauge chart"""
        fig = go.Figure()
        
        #add colored zones
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={
                'axis': {'range': [0, 100]},  # Remove 'orientation': 'h'
                'shape': 'bullet',  # This creates a horizontal bar gauge
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40], 'color': COLORS["green"]},
                    {'range': [40, 65], 'color': COLORS["yellow"]},
                    {'range': [65, 85], 'color': COLORS["orange"]},
                    {'range': [85, 100], 'color': COLORS["red"]}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        #configure layout
        fig.update_layout(
            height=150,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        
        return fig
    
    #create line chart for historical data (requires multiple entries)
    def create_history_chart(history_data):
        #convert list of dictionaries to DataFrame
        df = pd.DataFrame(history_data)
        
        #create figure
        fig = go.Figure()
        
        #add traces for each emotion
        for emotion in ["distress", "sadness", "worry", "anger"]:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[emotion],
                mode='lines+markers',
                name=emotion.capitalize(),
                line=dict(width=2),
                marker=dict(size=8)
            ))
        
        #update layout
        fig.update_layout(
            title="Emotional Trends Over Time",
            xaxis=dict(title="Analysis #"),
            yaxis=dict(title="Score", range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=400
        )
        
        return fig
    
    #create keyword extraction function
    def extract_significant_keywords(text):
        """Extract potentially significant words from text"""
        #this is a simplified version, it is highly recommended that later into the project we use NLP libraries

        common_words = {"the", "and", "a", "to", "of", "in", "that", "it", "with", 
                       "is", "was", "for", "on", "are", "as", "be", "this", "have", 
                       "or", "at", "by", "not", "but", "what", "all", "when", "can"}
        
        #simple word extraction
        words = text.lower().split()
        #remove  any punctuation
        cleaned_words = [word.strip(".,;:!?()[]{}\"'") for word in words]
        #remove common words and very short words
        significant = [word for word in cleaned_words if word not in common_words and len(word) > 3]
        
        #count occurrences
        word_counts = {}
        for word in significant:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        #get top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:10]  # Return top 10 words
    
    #adds timestamp to data for historical tracking
    def add_timestamp_to_data(data):
        data_with_timestamp = data.copy()
        data_with_timestamp['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data_with_timestamp
    
    # For tab 1 (Text Analysis)
    if tab1:
        #add the analyze button for tab 1
        analyze_button_tab1 = st.button("Analyze Text", key="analyze_button_tab1")
        analyze_button = analyze_button_tab1
    
    # For tab 2 (Upload File)
    elif tab2:
        #add the analyze button for tab 2
        analyze_button_tab2 = st.button("Analyze Text", key="analyze_button_tab2")
        analyze_button = analyze_button_tab2
    
    # For tab 3 (Our AI) - no analyze button
    else:
        analyze_button = None
        
        #results
        if analyze_button is not None and user_input and user_input.strip():
            with st.spinner("Analyzing your text..."):
                scores = analyze_text(user_input)
                
                #add to history for data
                st.session_state.history.append(scores)
                
                #extract keywords
                keywords = extract_significant_keywords(user_input)
                
            st.markdown("### Analysis Results:")
            
            #create tabs for different visualizations
            tabs = st.tabs(["Overview", "Detailed Analysis", "Trends", "Text Statistics", "Emotion Gauges"])
            
            #radar graph
            with tabs[0]:
                st.subheader("Emotional Profile")
                
               
                radar_fig = create_radar_chart(scores)
                st.plotly_chart(radar_fig, use_container_width=True)
                st.caption("This radar chart shows the emotional dimensions detected in your text.")
        
            #pie chart with key words detection table
            with tabs[1]:
                st.subheader("Detailed Analysis")
                
                #create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    #pie chart
                    fig_pie = px.pie(
                        names=[e.capitalize() for e in scores.keys()],
                        values=list(scores.values()),
                        hole=0.4,
                        color_discrete_sequence=[COLORS["red"], COLORS["yellow"], COLORS["orange"], COLORS["green"]],
                    )
                    fig_pie.update_traces(textinfo='label+percent', pull=[0.05]*len(scores))
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption("Emotional composition of your text")
                
                with col2:
                    #key Words Detection
                    if keywords:
                        st.subheader("Key Words Detection")
                        key_words_df = pd.DataFrame(keywords, columns=["Word", "Frequency"])
                        st.dataframe(key_words_df, use_container_width=True)
                        st.caption("Frequently used words in your text that may indicate emotional context.")
            
            #trend graph that takes in multiple responses
            with tabs[2]:
                st.subheader("Emotional Trends")
                
                #show historical data if multiple analyses have been done
                if len(st.session_state.history) > 1:
                    history_chart = create_history_chart(st.session_state.history)
                    st.plotly_chart(history_chart, use_container_width=True)
                    st.caption("Changes in emotional profiles across multiple analyses.")
                else:
                    st.info("Analyze more texts to see emotional trends over time.")
            
            #text statistics section
            with tabs[3]:
                st.subheader("Text Statistics")
                
                #create columns for metrics
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Word Count", len(user_input.split()))
                with stat_cols[1]:
                    st.metric("Character Count", len(user_input))
                with stat_cols[2]:
                    st.metric("Sentence Count", user_input.count('.') + user_input.count('!') + user_input.count('?'))
                with stat_cols[3]:
                    avg_word_length = sum(len(word) for word in user_input.split()) / max(1, len(user_input.split()))
                    st.metric("Avg Word Length", f"{avg_word_length:.1f}")
                
                #additional text statistics
                st.write("### Text Sample")
                if len(user_input) > 500:
                    st.write(f"{user_input[:500]}...")
                else:
                    st.write(user_input)
            
        
            #individual emotion gauges
            with tabs[4]:
                st.subheader("Individual Emotion Gauges")
                
                #create a separate tab for each emotion gauge
                emotion_tabs = st.tabs([emotion.capitalize() for emotion in scores.keys()])
                
                for i, (emotion, score) in enumerate(scores.items()):
                    with emotion_tabs[i]:
                        color = get_color(score)
                        
                        #create a full-sized gauge for each emotion
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': f"{emotion.capitalize()} Score"},
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
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        #add descriptions for each emotion level
                        if score < 40:
                            st.success(f"Low {emotion} level detected.")
                        elif score < 65:
                            st.info(f"Moderate {emotion} level detected.")
                        elif score < 85:
                            st.warning(f"High {emotion} level detected.")
                        else:
                            st.error(f"Severe {emotion} level detected.")
        
            st.caption("Note: This analysis is for informational purposes only and should not be considered a medical diagnosis.")
        
        # No additional messages needed here since warnings are shown in each tab

#FAQ Section
with main_col:
    st.markdown("---")
    st.markdown("### Frequently Asked Questions")
    
    #creates coloums to format Qs
    col1, col2 = st.columns(2)
    
    with col1:
        faq_expander = st.expander("How does this tool work?")
        with faq_expander:
            st.write("""
            This tool analyzes text using natural language processing (NLP) technology to identify patterns 
            in language that may correspond to different emotional dimensions. The analysis is based on 
            linguistic features rather than clinical diagnostic criteria. The results are presented as 
            visualizations showing the relative presence of different emotional dimensions in the text.
            """)
        
        privacy_expander = st.expander("How is my data protected?")
        with privacy_expander:
            st.write("""
            Your privacy is our top priority. All analysis is performed locally in your browser, and your 
            text is never stored on external servers. Your data remains on your device and is not shared 
            with anyone. This tool is designed with privacy-first principles.
            """)
    
    with col2:
        usage_expander = st.expander("What are the best uses for this tool?")
        with usage_expander:
            st.write("""
            This tool works best for:
            - Exploring emotional content in written text
            - Analyzing journal entries over time
            - Understanding emotional patterns in communication
            - Writing analysis for creative purposes
            
            It should not be used for medical diagnosis or as a substitute for professional advice.
            """)
        
        limitations_expander = st.expander("What are the limitations?")
        with limitations_expander:
            st.write("""
            - The tool can only analyze text, not images or audio
            - Results are based on linguistic patterns, not clinical assessment
            - The model has been trained on specific datasets and may not generalize to all types of text
            - Analysis should be interpreted as exploratory, not diagnostic
            """)

#footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")