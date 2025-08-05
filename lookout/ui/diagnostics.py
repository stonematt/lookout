import streamlit as st


def render():
    """
    Render the Diagnostics tab for archive data review.
    """
    st.title("📊 Archive Diagnostics")

    if "history_df" not in st.session_state:
        st.warning("No archive data loaded.")
        return

    history_df = st.session_state["history_df"]

    st.info(
        "Diagnostics tools for identifying archive gaps, sensor downtime, and data quality."
    )
    st.write(
        f"Loaded archive range: {history_df['date'].min()} → {history_df['date'].max()}"
    )
