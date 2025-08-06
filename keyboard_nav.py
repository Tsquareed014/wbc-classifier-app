# keyboard_nav.py
import streamlit as st
import streamlit.components.v1 as components

def arrow_key_nav(
    max_idx: int,
    session_key: str = "current_idx",
    input_key: str = "__keynav__",
) -> int:
    """
    Capture left/right arrow key presses to navigate between indices [0…max_idx].
    Returns the updated index stored under st.session_state[session_key].
    """
    # 1) Initialize session_state index if missing
    if session_key not in st.session_state:
        st.session_state[session_key] = 0

    # 2) Hidden text_input to shuttle key events back to Python
    _ = st.text_input(
        "",
        key=input_key,
        value="",
        label_visibility="collapsed",
    )

    # 3) Inject JS listener for ArrowLeft/ArrowRight
    components.html(
        """
        <script>
        document.addEventListener("keydown", (e) => {
          if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
            const dir = e.key === "ArrowLeft" ? "prev" : "next";
            const inp = window.parent.document.getElementById("__keynav__");
            if (inp) {
              inp.value = dir;
              inp.dispatchEvent(new Event('input', { bubbles: true }));
            }
          }
        });
        </script>
        """,
        height=0,
    )

    # 4) React to the hidden input’s value
    nav = st.session_state.get(input_key, "")
    if nav == "prev":
        st.session_state[session_key] = max(0, st.session_state[session_key] - 1)
    elif nav == "next":
        st.session_state[session_key] = min(max_idx, st.session_state[session_key] + 1)

    # 5) Clear for the next keypress
    st.session_state[input_key] = ""

    return st.session_state[session_key]

