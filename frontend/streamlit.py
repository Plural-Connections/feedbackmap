import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import json, re, spacy
from collections import defaultdict

# from umap import umap_ as um

model = SentenceTransformer("all-MiniLM-L6-v2")
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

# Do not treat these columns in the input as either categorical or free-response questions
_SKIP_COLUMNS = ["Timestamp"]

# If a row has more than this many unique values (relative to total)
# consider it to be a free-response text field
_MAX_FRACTION_FOR_CATEGORICAL = 0.2


def embed_responses(raw_responses):

    # Split raw responses into sentences and embed
    all_sentences = []
    all_embeddings = []
    for r in raw_responses:
        doc = nlp(r)
        for sent in doc.sents:
            cleaned_sent = sent.text.strip()
            if cleaned_sent:
                all_sentences.append(cleaned_sent)
                all_embeddings.append(model.encode(cleaned_sent))

    # UMAP everything
    all_umap_emb = []
    # for emb in all_embeddings:
    #     all_umap_emb.append(um.UMAP(n_components=2, metric="euclidean").fit(emb))

    return all_sentences, all_umap_emb


def val_dictionary_for_column(df, col):
    # Pull the val:count dict for a given column, accounting for comma-separated multi-values
    vals = defaultdict(lambda: 0)
    for index, row in df.iterrows():
        this_vals = [x.strip() for x in re.split(";|,", str(row[col]))]
        for val in this_vals:
            if not val or val.isnumeric():
                # Skip purely numeric values for now, as these are often scale
                # questions
                continue
            if val.lower() in [
                "very important",
                "moderately important",
                "very important;moderately important",
            ]:
                val = "Moderately or very important"
            vals[val] += 1
    return vals


def streamlit_app():
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload a CSV of your Google Forms results")

        if uploaded_file:
            df = pd.read_csv(uploaded_file, dtype=str).fillna("")

            categories = {}  # column -> val_dict
            text_response_columns = []

            for column in df.columns:
                if column not in _SKIP_COLUMNS:
                    val_dict = val_dictionary_for_column(df, column)
                    # If it's got more than _MAX_FRACTION_FOR_CATEGORICAL * numrows different vals,
                    # consider it a text response field, otherwise it's a categorical attribute
                    if len(val_dict) < _MAX_FRACTION_FOR_CATEGORICAL * len(df.index):
                        categories[column] = val_dict
                    else:
                        text_response_columns.append(column)

            # Compute embeddings for all of the open ended text
            sents_and_umap_embs_for_questions = {}
            for q in text_response_columns:
                sents, embs = embed_responses(df[q].tolist())
                sents_and_umap_embs_for_questions[q] = (sents, embs)

            print("Done embedding ...")

    st.title("Survey Mirror")


if __name__ == "__main__":
    streamlit_app()
