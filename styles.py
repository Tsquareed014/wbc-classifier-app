# styles.py

import streamlit as st

def apply_text_size(text_size_percent: int):
    """
    Inject CSS to scale the app’s base font size by a percentage.
    """
    st.markdown(
        f"""
        <style>
        /* Scale all text in the app */
        html, body, [class*="css"], .stApp {{
            font-size: {text_size_percent}% !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

