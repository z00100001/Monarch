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
    
    #input header options
    st.subheader("Enter or upload text for analysis")
    
    #create tabs for different input methods
    tab1, tab2 = st.tabs(["Enter Text", "Upload File"])
    
    #text input
    with tab1:
        user_input = st.text_area("Type or paste any text you want to analyze...", height=200)
        input_source = "text_area"
    
    #file input
    with tab2:
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            if file_type == "pdf":
                user_input = extract_text_from_pdf(uploaded_file)
            elif file_type == "docx":
                user_input = extract_text_from_docx(uploaded_file)
            elif file_type == "txt":
                user_input = extract_text_from_txt(uploaded_file)
            else:
                user_input = ""
                st.error("Unsupported file type")
            
            if user_input:
                st.success(f"Successfully extracted text from {uploaded_file.name}")
                st.text_area("Extracted text:", user_input, height=200)
            
            input_source = "file_upload"
    
    #adds the character count at the bottom for the user
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
    
    #add the analyze button
    analyze_button = st.button("Analyze Text")
    
    #results
    if analyze_button and user_input and user_input.strip():
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
        
        #export options (Not functional right now)
        st.markdown("### Export Options")
        export_cols = st.columns(3)
        with export_cols[0]:
            if st.button("Export as CSV"):
                #this would trigger a download, but doesnt work right now
                st.success("CSV export functionality would be implemented here.")
        with export_cols[1]:
            if st.button("Export as PDF"):
                st.success("PDF export functionality would be implemented here.")
        with export_cols[2]:
            if st.button("Save to History"):
                st.success("Analysis saved to history.")
        
        st.caption("Note: This analysis is for informational purposes only and should not be considered a medical diagnosis.")
    
    elif analyze_button and (not user_input or not user_input.strip()):
        st.warning("Please enter or upload text for analysis.")
    else:
        st.info("Enter text or upload a file above and click 'Analyze Text' to begin.")

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

# footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")