#!/usr/bin/env python3

import streamlit as st
import extra_streamlit_components as stx

import import_tab
import summary_tab
import analysis_tab

import app_config
import gpt3_model
import local_models
import parse_csv

import logger


@st.cache_resource
def get_config(mock_mode):
    config = local_models.get_config(mock_mode)
    config.update(gpt3_model.get_config())
    return config


def streamlit_app():
    st.set_page_config(
        page_title=app_config.TITLE, page_icon=app_config.ICON, layout="wide"
    )
    st.title(app_config.ICON + " " + app_config.TITLE)
    app_config.CONFIG.update(get_config(app_config.MOCK_MODE))
    logger.init()
    logger.log(action="APP_LOADED")
    columns_to_analyze = None
    df = None

    # st.runtime.legacy_caching.caching.clear_cache()

    # See https://discuss.streamlit.io/t/mini-tutorial-initializing-widget-values-and-getting-them-to-stick-without-double-presses/31391/4
    if "grouping_key" in st.session_state:
        st.session_state["grouping_key"] = st.session_state["grouping_key"]

    if "analyze" in st.session_state:
        columns_to_analyze = st.session_state["analyze"]
    if "uploaded" in st.session_state:
        df = st.session_state["uploaded"]

    # Arrange tabs
    tab_bar = st.empty()
    content_placeholder = st.container()
    summary_tab_st = None
    analyze_tab_st = None

    if "analyze" in st.session_state:
        default_tab = 3
    elif "uploaded" in st.session_state:
        default_tab = 2
    else:
        default_tab = 1

    with tab_bar:
        if "uploaded_file" in st.session_state:
            summary_title = "Summary (%s)" % (st.session_state["uploaded_file"].name)
        else:
            summary_title = "File summary"
        chosen_id = stx.tab_bar(data=[
            stx.TabBarItemData(id=1, title="Welcome", description=""),
            stx.TabBarItemData(id=2, title=summary_title, description=""),
            stx.TabBarItemData(id=3, title="Response analysis", description=""),
        ], default=default_tab)

        with content_placeholder:
            if chosen_id == "1":
                import_tab.run(df)
            elif chosen_id == "2":
                if "uploaded" in st.session_state:
                    summary_tab_st = content_placeholder.container()
                else:
                    st.write("Select a file in the Welcome tab to see a summary of it here.")
            elif chosen_id == "3":
                if "analyze" in st.session_state:
                    analyze_tab_st = content_placeholder.container()
                else:
                    st.write("Select a column to analyze in the Summary tab to see an analysis here.")

    if df is not None:
        if (
            "categories" in st.session_state
            and "text_response_columns" in st.session_state
        ):
            categories = st.session_state["categories"]
            text_response_columns = st.session_state["text_response_columns"]
        else:
            with st.spinner():
                categories, text_response_columns = parse_csv.infer_column_types(df)
                st.session_state["categories"] = categories.copy()
                st.session_state["text_response_columns"] = text_response_columns.copy()
        if summary_tab_st:
            with summary_tab_st:
                summary_tab.run(df, text_response_columns, categories)

    if columns_to_analyze:
        with analyze_tab_st:
            analysis_tab.run(columns_to_analyze, df, categories)


if __name__ == "__main__":
    streamlit_app()
