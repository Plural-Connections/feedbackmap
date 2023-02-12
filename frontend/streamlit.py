#!/user/bin/env python3

import streamlit as st
import json
from collections import defaultdict

import hdbscan
from umap import umap_ as um

import charts
import gpt3_model
import local_models
import parse_csv
import util

_CONFIG = {}
_CLUSTER_OPTION_TEXT = "[Auto-pick colors based on the topic of the response text]"
_MOCK_MODE = False  # Set to true to run without transformers, spacy, or gpt-3
_TITLE = "Feedback Map"
_CATEGORICAL_QUESTIONS_BGCOLOR = "lightyellow"
_MAX_VALUES_TO_SUMMARIZE = 10

@st.cache_resource
def get_config(mock_mode):
    config = local_models.get_config(mock_mode)
    config.update(gpt3_model.get_config(mock_mode))
    return config


def get_color_key_of_interest(categories):
    res = st.selectbox(
        "Group the points based on the respondent's answer to:",
        [_CLUSTER_OPTION_TEXT] + list(categories.keys()),
        format_func=lambda x: str(x),
        index=1,
    )
    return res


# hash_funcs={dict: (lambda _: None)})
@st.cache_data
def embed_responses(df, q):
    # Split raw responses into sentences and embed
    parent_records = []
    all_sentences = []
    all_embeddings = []
    for _, row in df.iterrows():
        doc = _CONFIG["nlp"](row[q])
        for sent in doc.sents:
            cleaned_sent = sent.text.strip()
            if cleaned_sent:
                parent_records.append(row)
                all_sentences.append(cleaned_sent)
    all_embeddings = _CONFIG["model"].encode(all_sentences)
    # UMAP everything
    all_umap_emb = um.UMAP(n_components=2, metric="euclidean").fit_transform(
        all_embeddings
    )

    return all_sentences, all_umap_emb, parent_records, all_embeddings


def show_import_tab(current_csv_file_df):
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


def show_summary_tab(df, text_response_columns, categories):
    st.write(
        "Processed **%d** responses with **%d** text response questions and **%d** categorical questions"
        % (len(df), len(text_response_columns), len(categories))
    )
    st.write(
        "Click a text response button below to analyze the results for a specific question."
    )
    st.subheader("Text response questions:")
    buttons = {}
    for k, v in text_response_columns.items():
        btn_col, info_col = st.columns(2)
        with btn_col:
            buttons[k] = st.button(k, use_container_width=True, type="primary")
        with info_col:
            st.write(
                "%0.1f%% response rate" % (100.0 * (1.0 - (v.get("", 0.0) / len(df))))
            )
    st.subheader("Categorical questions:")
    for k, v in categories.items():
        q_col, info_col = st.columns(2)
        with q_col:
            st.markdown(
                '<p style="background-color:%s; border-radius: 5px; padding:10px">%s</p>'
                % (_CATEGORICAL_QUESTIONS_BGCOLOR, k),
                unsafe_allow_html=True,
            )
        with info_col:
            st.write(
                "%d different values\\\n%0.2f selections per response"
                % (len(v), sum(v.values()) / len(df))
            )

    for k, b in buttons.items():
        if b:
            st.session_state["analyze"] = [k]
            st.experimental_rerun()


def show_analysis_tab(columns_to_analyze, df, categories):
    st.subheader(", ".join(columns_to_analyze))
    # Compute GPT3-based summary
    with st.spinner():
        # Layout expanders
        overall_summary_expander = st.expander("Auto-generated summary of the responses", expanded=True)
        color_key = get_color_key_of_interest(categories)
        scatterplot_exapander = st.expander("Topic scatterplot of the responses", expanded=True)
        value_table_exapander = st.expander("Summary of values", expanded=True)

        # Overall summary
        with overall_summary_expander:
            res = _CONFIG["llm"].get_summary(df, columns_to_analyze[0])
            st.write("**%s** %s" % (res["instructions"], res["answer"]))


        # Compute embeddings and plot scatterplot
        with scatterplot_exapander:        
            with st.spinner():
                data = []
                for q in columns_to_analyze:
                    sents, embs, parent_records, full_embs = embed_responses(df, q)
                    data.extend(
                        [
                            {
                                "sentence": sents[i],
                                "vec": embs[i],
                                "rec": parent_records[i],
                            }
                            for i in range(len(sents))
                        ]
                    )
                if color_key == _CLUSTER_OPTION_TEXT:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
                    clusterer.fit(full_embs)
                    data = data.copy()  # Copy, to avoid ST cache warning
                    for i, x in enumerate(data):
                        x["rec"][_CLUSTER_OPTION_TEXT] = "Cluster %s" % (
                            str(clusterer.labels_[i])
                        )
                scatterplot = charts.make_scatterplot_base(data, color_key)
                st.altair_chart(scatterplot)

        # Per-value summary table
        with value_table_exapander:
            table = []
            values = list(categories.get(color_key, {}).keys())
            summaries = _CONFIG["llm"].get_summaries(df, columns_to_analyze[0], color_key, values[:_MAX_VALUES_TO_SUMMARIZE])
            for i, res in enumerate(summaries):
                num_responses = categories.get(color_key, {}).get(values[i], 0)
                table.append({"Answer to \"%s\"" % (color_key): values[i],
                              "Number of responses": num_responses,
                              "Auto-generated summary for their answer to \"%s\"" % (columns_to_analyze[0]): res["answer"],
                              "Nonempty rate": "%0.1f%%" % (100.0 * res["nonempty_responses"] / num_responses)})
            st.table(table)

def streamlit_app():
    st.set_page_config(page_title=_TITLE, layout="wide")
    st.title(_TITLE)
    _CONFIG.update(get_config(_MOCK_MODE))
    columns_to_analyze = None
    csv_file_df = None

    if "analyze" in st.session_state:
        columns_to_analyze = st.session_state["analyze"]
    if "uploaded" in st.session_state:
        csv_file_df = st.session_state["uploaded"]

    # Arrange tabs
    tab_placeholder = st.empty()
    with tab_placeholder:
        if "analyze" in st.session_state:
            analyze_tab, summary_tab, import_tab = st.tabs(
                ["Response analysis", "Summary", "Import file"]
            )
        elif "uploaded" in st.session_state:
            summary_tab, import_tab = st.tabs(["Summary", "Import file"])
        else:
            import_tab = st.tabs(["Import file"])[0]

    with import_tab:
        show_import_tab(csv_file_df)

    if csv_file_df is not None:
        with st.spinner():
            df = csv_file_df
            categories, text_response_columns = parse_csv.infer_column_types(df)
        with summary_tab:
            show_summary_tab(df, text_response_columns, categories)

    if columns_to_analyze:
        with analyze_tab:
            show_analysis_tab(columns_to_analyze, df, categories)


if __name__ == "__main__":
    streamlit_app()
