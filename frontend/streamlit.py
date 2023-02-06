#!/user/bin/env python3

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import json, spacy
from collections import defaultdict
from umap import umap_ as um

import charts
import gpt3_model
import parse_csv

_CONFIG = {}


@st.cache(allow_output_mutation=True)
def get_config():
    spacy.cli.download("en_core_web_sm")
    return {
        "nlp": spacy.load("en_core_web_sm"),
        "model": SentenceTransformer("all-MiniLM-L6-v2"),
    }


def get_questions_of_interest(columns, header="Select a question"):
    res = st.selectbox(
        "Select a question from your survey to analyze", [header] + columns
    )
    return (res != header) and [res] or None


def get_color_key_of_interest(categories):
    res = st.selectbox("Color the points based on answer to:", list(categories.keys()))
    return res


@st.cache(suppress_st_warning=True, hash_funcs={dict: (lambda _: None)})
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

    return all_sentences, all_umap_emb, parent_records


def process_input_file(uploaded_file):
    df = pd.read_csv(uploaded_file, dtype=str).fillna("")
    return df


def streamlit_app():
    _CONFIG.update(get_config())
    columns_to_analyze = None

    with st.sidebar:
        st.title("Survey Mirror")
        uploaded_file = st.file_uploader("Upload a CSV of your Google Forms results")
        if uploaded_file:
            with st.spinner():
                df = process_input_file(uploaded_file)
                categories, text_response_columns = parse_csv.infer_column_types(df)
            columns_to_analyze = get_questions_of_interest(text_response_columns)

    if columns_to_analyze:
        # Select box for how to color the points
        st.subheader(", ".join(columns_to_analyze))
        color_key = get_color_key_of_interest(categories)

        # Compute GPT3-based summary
        with st.spinner():
            with st.expander("Auto-generated summary of the responses", expanded=True):
                res = gpt3_model.get_summary(df, columns_to_analyze)
                st.write("**%s** %s" % (res["instructions"], res["answer"]))

        # Compute embeddings
        with st.spinner():
            data = []
            for q in columns_to_analyze:
                sents, embs, parent_records = embed_responses(df, q)
                data.extend(
                    [
                        {"sentence": sents[i], "vec": embs[i], "rec": parent_records[i]}
                        for i in range(len(sents))
                    ]
                )
            with st.expander("Topic scatterplot of the responses", expanded=True):
                scatterplot = charts.make_scatterplot_base(data, color_key)
                st.altair_chart(scatterplot)


if __name__ == "__main__":
    streamlit_app()
