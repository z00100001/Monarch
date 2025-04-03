import streamlit as st
from datetime import datetime

# Page settings
st.set_page_config(
    page_title="Monarch - Mental Health Text Analyzer",
    layout="centered"
)

# Logo + Title
st.markdown("<h1 style='text-align: center;'>Monarch</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Private AI for Emotional Pattern Detection</h4>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("Paste your text below:")
user_input = st.text_area("Enter journal entries, chat logs, or anything you want to analyze.", height=200)
analyze_btn = st.button("Analyze")

# this will be the placeholder
if analyze_btn:
    st.markdown("### Results:")
    st.info("This is where the emotional analysis will appear.")

# Footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")
