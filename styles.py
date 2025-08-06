# styles.py

import streamlit as st

def apply_text_size(text_size_percent: int):
    """
    Inject CSS to globally scale text size by a percentage.
    """
    css = f"""
    <style>
    /* Scale base font size */
    html, body, [class*="css"], .stApp {{
        font-size: {text_size_percent}% !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
