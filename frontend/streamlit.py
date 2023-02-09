#!/user/bin/env python3

import streamlit as st
import pandas as pd
import json
from collections import defaultdict

import hdbscan
from umap import umap_ as um

import charts
import gpt3_model
import local_models
import parse_csv

_CONFIG = {}
_CLUSTER_OPTION_TEXT = "[Auto-pick colors based on the topic of the response text]"


@st.cache(allow_output_mutation=True)
def get_config():
    return local_models.get_config()


def get_questions_of_interest(columns, header="Select a question"):
    res = st.selectbox(
        "Select a question from your survey to analyze",
        [header] + columns,
        format_func=lambda x: str(x),
    )
    return (res != header) and [res] or None


def get_color_key_of_interest(categories):
    res = st.selectbox(
        "Color the points based on the respondent's answer to:",
        [_CLUSTER_OPTION_TEXT] + list(categories.keys()),
        format_func=lambda x: str(x),
        index=1,
    )
    return res


@st.cache(allow_output_mutation=True, hash_funcs={dict: (lambda _: None)})
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


def process_input_file(uploaded_file):
    df = pd.read_csv(uploaded_file, dtype=str).fillna("")
    return df


def streamlit_app():
    _CONFIG.update(get_config())
    columns_to_analyze = None

    if "analyze" in st.session_state:
        analyze_tab, summary_tab = st.tabs(["Response analysis", "Summary"])
    else:
        summary_tab_placeholder = st.empty()

    with st.sidebar:
        st.title("Feedback Map")
        uploaded_file = st.file_uploader("Upload a CSV of your Google Forms results")
        if uploaded_file:
            with st.spinner():
                df = process_input_file(uploaded_file)
                categories, text_response_columns = parse_csv.infer_column_types(df)
            if "analyze" not in st.session_state:
                with summary_tab_placeholder:
                    summary_tab = st.tabs(["Summary"])[0]

            with summary_tab:
                st.subheader("Summary:")
                st.write(
                    "Processed **%d** responses with **%d** text response questions and **%d** categorical questions"
                    % (len(df), len(text_response_columns), len(categories))
                )
                st.write(
                    "Click a text response button below to analyze the results for a specific question."
                )
                st.subheader("Text response columns:")
                buttons = {}
                for k, v in text_response_columns.items():
                    btn_col, info_col = st.columns(2)
                    with btn_col:
                        buttons[k] = st.button(k)
                    with info_col:
                        st.write(
                            "%0.1f%% response rate"
                            % (100.0 * (1.0 - (v.get("", 0.0) / len(df))))
                        )
                st.subheader("Categorical columns:")
                st.write(
                    "\n".join(
                        [
                            "- %s [%d different values]" % (c, len(v))
                            for c, v in categories.items()
                        ]
                    )
                )
            for k, b in buttons.items():
                if b:
                    columns_to_analyze = [k]
                    st.session_state["analyze"] = columns_to_analyze
                    st.experimental_rerun()

            if "analyze" in st.session_state:
                columns_to_analyze = st.session_state["analyze"]

    if columns_to_analyze:
        with analyze_tab:
            st.subheader(", ".join(columns_to_analyze))
            # Compute GPT3-based summary
            with st.spinner():
                with st.expander(
                    "Auto-generated summary of the responses", expanded=True
                ):
                    res = gpt3_model.get_summary(df, columns_to_analyze[0])
                    st.write("**%s** %s" % (res["instructions"], res["answer"]))

            # Compute embeddings
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
                with st.expander("Topic scatterplot of the responses", expanded=True):
                    color_key = get_color_key_of_interest(categories)
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


if __name__ == "__main__":
    streamlit_app()
