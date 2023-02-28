import streamlit as st

import parse_csv
import util
import logger


def run(current_csv_file_df):
    util.include_markdown("welcome")
    new_csv_file = st.file_uploader(
        "", on_change=logger.log, kwargs=dict(action="UPLOADING_FILE")
    )
    if new_csv_file:
        try:
            df = parse_csv.process_file(new_csv_file)
        except Exception as e:
            st.error("We failed to parse your file.  The error is printed below.")
            st.exception(e)
            df = None
        if df is not None and not df.equals(current_csv_file_df):
            st.session_state["uploaded"] = df
            for x in ["analyze", "categories", "grouping_key",
                      "text_response_columns"]:
                if x in st.session_state:
                    del st.session_state[x]
            st.experimental_rerun()  # TODO: why doesn't it open 1st tab here?
    with st.expander("About + Privacy", expanded=False):
        util.include_markdown("about")
    with st.expander("Frequently Asked Questions", expanded=False):
        util.include_markdown("faqs")
