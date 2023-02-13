import streamlit as st

import parse_csv
import util


def run(current_csv_file_df):
    new_csv_file = None
    util.include_markdown("welcome")
    new_csv_file = st.file_uploader("Upload CSV here.")
    if new_csv_file:
        df = parse_csv.process_input_file(new_csv_file)
        if not df.equals(current_csv_file_df):
            st.session_state["uploaded"] = df
            if "analyze" in st.session_state:
                del st.session_state["analyze"]
            st.experimental_rerun()  # TODO: why doesn't it open 1st tab here?
    with st.expander("About Feedback Map", expanded=False):
        util.include_markdown("about")
