import streamlit as st

import parse_csv
import util


def run(current_csv_file_df):
    util.include_markdown("welcome")
    new_csv_file = st.file_uploader("")
    if new_csv_file:
        try:
            df = parse_csv.process_csv(new_csv_file)
        except Exception as e:
            st.warning(
                "We failed to parse that as a CSV file.  We'll attempt to load it as a JSONLines file and then as a plain-text file with a single column."
            )
            try:
                df = parse_csv.process_txt(new_csv_file)
            except Exception as e:
                st.error("We failed to parse your file.  The error is printed below.")
                st.exception(e)
                df = None
        if df is not None and not df.equals(current_csv_file_df):
            st.session_state["uploaded"] = df
            if "analyze" in st.session_state:
                del st.session_state["analyze"]
            st.experimental_rerun()  # TODO: why doesn't it open 1st tab here?
    with st.expander("About + Privacy", expanded=False):
        util.include_markdown("about")
    with st.expander("Frequently Asked Questions", expanded=False):
        util.include_markdown("faqs")
