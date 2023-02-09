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
    summary_section = st.empty()

    with st.sidebar:
        st.title("Feedback Mirror")
        uploaded_file = st.file_uploader("Upload a CSV of your Google Forms results")
        if uploaded_file:
            with st.spinner():
                df = process_input_file(uploaded_file)
                categories, text_response_columns = parse_csv.infer_column_types(df)
            with summary_section:
                with st.container():
                    st.subheader("Summary:")
                    st.write(
                        "Processed **%d** responses with **%d** text response questions and **%d** categorical questions"
                        % (len(df), len(text_response_columns), len(categories))
                    )
                    st.write(
                        "Select a text response column on the left sidebar to analyze the results."
                    )
                    st.subheader("Text response columns:")
                    st.write("\n".join(["- " + c for c in text_response_columns]))
                    st.subheader("Categorical columns:")
                    st.write(
                        "\n".join(
                            [
                                "- %s [%d different values]" % (c, len(v))
                                for c, v in categories.items()
                            ]
                        )
                    )
            columns_to_analyze = get_questions_of_interest(text_response_columns)

    if columns_to_analyze:
        with summary_section:
            # Clear summary section
            st.write("")
        st.subheader(", ".join(columns_to_analyze))

        # Compute GPT3-based summary
        with st.spinner():
            with st.expander("Auto-generated summary of the responses", expanded=True):
                res = gpt3_model.get_summary(df, columns_to_analyze[0])
                st.write("**%s** %s" % (res["instructions"], res["answer"]))

        # Compute embeddings
        with st.spinner():
            data = []
            for q in columns_to_analyze:
                sents, embs, parent_records, full_embs = embed_responses(df, q)
                data.extend(
                    [
                        {"sentence": sents[i], "vec": embs[i], "rec": parent_records[i]}
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
