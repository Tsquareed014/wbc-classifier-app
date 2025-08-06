# keyboard_nav.py

import streamlit as st
from streamlit_keys import key_listener

def nav_index(current_idx: int, max_idx: int) -> int:
    """
    Listen for left/right arrow key presses and return a new index.
    Requires `pip install streamlit-keys`.
    """
    key = key_listener()
    if key == "ArrowLeft":
        return max(0, current_idx - 1)
    if key == "ArrowRight":
        return min(max_idx, current_idx + 1)
    return current_idx
