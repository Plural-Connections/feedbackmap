#!/user/bin/env python3

import streamlit as st

import import_tab
import summary_tab
import analysis_tab

import app_config
import gpt3_model
import local_models
import parse_csv


@st.cache_resource
def get_config(mock_mode):
    config = local_models.get_config(mock_mode)
    config.update(gpt3_model.get_config())
    return config


def streamlit_app():
    st.set_page_config(page_title=app_config.TITLE, page_icon=app_config.ICON,
                       layout="wide")
    st.title(app_config.ICON + " " + app_config.TITLE)
    app_config.CONFIG.update(get_config(app_config.MOCK_MODE))
    columns_to_analyze = None
    df = None

    if "analyze" in st.session_state:
        columns_to_analyze = st.session_state["analyze"]
    if "uploaded" in st.session_state:
        df = st.session_state["uploaded"]

    # Arrange tabs
    tab_placeholder = st.empty()
    with tab_placeholder:
        if "analyze" in st.session_state:
            analyze_tab_st, summary_tab_st, import_tab_st = st.tabs(
                ["Response analysis", "Summary", "Welcome"]
            )
        elif "uploaded" in st.session_state:
            summary_tab_st, import_tab_st = st.tabs(["Summary", "Welcome"])
        else:
            import_tab_st = st.tabs(["Welcome"])[0]

    with import_tab_st:
        import_tab.run(df)

    if df is not None:
        with st.spinner():
            categories, text_response_columns = parse_csv.infer_column_types(df)
        with summary_tab_st:
            summary_tab.run(df, text_response_columns, categories)

    if columns_to_analyze:
        with analyze_tab_st:
            analysis_tab.run(columns_to_analyze, df, categories)


if __name__ == "__main__":
    streamlit_app()
