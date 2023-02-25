from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd

import random
from collections import defaultdict

import parse_csv

import streamlit as st


def get_config(mock_mode):
    """If mock_mode is set, don't load any NLP libraries."""
    return {
        "nlp": True and dumb_segmenter_model() or spacy_segmenter_model(),
        "model": mock_mode and DumbEmbeddingModel() or SentenceTransformersModel(),
    }


class SentenceTransformersModel:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, x):
        return self.m.encode(x)


class DumbEmbeddingModel:
    def __init__(self):
        pass

    def encode(self, x):
        return [[hash(s), hash(s)] for s in x]


def spacy_segmenter_model():
    import spacy

    spacy.cli.download("en_core_web_sm")
    return spacy.load("en_core_web_sm")


def dumb_segmenter_model():
    return lambda x: type(
        "obj",
        (object,),
        {"sents": [type("obj", (object,), {"text": y}) for y in x.split(".")]},
    )


def get_top_phrases(data, grouping_key):
    sentences = [x["sentence"] for x in data]
    sentences = [s.split() for s in sentences]

    # Determine a set of words and phrases to use
    phrases = Phrases(
        sentences, min_count=2, threshold=0.1, connector_words=ENGLISH_CONNECTOR_WORDS
    )
    res = []
    counts = defaultdict(lambda: (defaultdict(lambda: 0)))
    count_sums = defaultdict(lambda: 0)
    all_category_values = set()
    for i, sentence in enumerate(sentences):
        rec = data[i]["rec"]
        category_string = rec.get(grouping_key, "Unknown")
        categories_values = parse_csv.split_values(category_string)
        all_category_values.update(categories_values)
        for v in categories_values:
            count_sums[v] += 1
        for p in phrases[sentence]:
            if p.lower() not in STOPWORDS:
                counts[p]["Total"] += 1
                for v in categories_values:
                    counts[p][v] += 1
    table = []

    for term, c in counts.items():
        if c["Total"] >= 5:
            table.append(
                {
                    "Term": term.replace("_", " "),
                    "Total": counts[term]["Total"],
                    **{k: c[k] for k in all_category_values},
                }
            )  # Show C(word, category)
    #   **{k:round(100.0*c[k]/(count_sums[k]+1), 2) for k in all_category_values}})     # Show P(word|category)
    table.sort(key=lambda x: x["Total"], reverse=True)
    return pd.DataFrame(table)
