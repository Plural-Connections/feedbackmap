
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.parsing.preprocessing import STOPWORDS
import hdbscan
import numpy as np
import pandas as pd
from umap import umap_ as um

import random
import re
from collections import defaultdict

import app_config
import parse_csv

def get_config(mock_mode):
    """If mock_mode is set, don't load any NLP libraries."""
    return {
        "nlp": True and dumb_segmenter_model() or spacy_segmenter_model(),
        "model": mock_mode and DumbEmbeddingModel() or SentenceTransformersModel(),
    }


class SentenceTransformersModel:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(app_config.EMBEDDING_MODEL)

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


def get_highest_mi_words(df):
    # Calculate the total number of word occurrences across all clusters
    df = df.copy()
    df.set_index(df.columns[0], inplace=True)
    total_word_occurrences = df.sum().sum()

    # Calculate the probability of each word and each cluster
    word_prob = df.sum(axis=1) / total_word_occurrences
    doc_prob = df.sum(axis=0) / total_word_occurrences

    # Calculate PMI for each word-cluster pair
    pmi_df = pd.DataFrame()
    for doc in df.columns:
        pmi_df[doc] = df[doc] / total_word_occurrences / (word_prob * doc_prob[doc])
        pmi_df[doc] = pmi_df[doc].apply(lambda x: 0 if x == 0 else np.log2(x))

    # Find the top three words with the highest PMI for each cluster
    top_words = {}
    for doc in pmi_df.columns:
        top_words[doc] = pmi_df[doc].nlargest(3).index.tolist()
    return top_words


def get_top_phrases(data, grouping_key):
    sentences = [x["sentence"] for x in data]
    sentences = [re.split("\W+", s) for s in sentences]

    # Determine a set of words and phrases to use
    phrases = Phrases(
        sentences,
        min_count=2,
        threshold=0.1,
        connector_words=frozenset(
            list(ENGLISH_CONNECTOR_WORDS) + ["is", "are", "was", "I", "we", "it", "a", "000"]
        ),
    )
    res = []
    counts = defaultdict(lambda: (defaultdict(lambda: 0)))
    count_sums = defaultdict(lambda: 0)
    all_category_values = set()
    norm = {}  # downcased -> first occurrence of string
    for i, sentence in enumerate(sentences):
        rec = data[i]["rec"]
        category_string = rec.get(grouping_key, "Unknown")
        categories_values = parse_csv.split_values(category_string)
        all_category_values.update(categories_values)
        for v in categories_values:
            count_sums[v] += 1
        for p in phrases[sentence]:
            if p.lower() not in STOPWORDS:
                if p.lower() in norm:
                    # This ensures we use just one casing (the first seen) of any term
                    p = norm[p.lower()]
                else:
                    norm[p.lower()] = p
                counts[p]["Total"] += 1
                for v in categories_values:
                    counts[p][v] += 1
    table = []

    for term, c in counts.items():
        if c["Total"] >= 5 and len(term) > 2:
            table.append(
                {
                    "Term": term.replace("_", " "),
                    "Total": counts[term]["Total"],
                    **{k: c[k] for k in all_category_values},
                }
            )  # Show C(word, category)
    #   **{k:round(100.0*c[k]/(count_sums[k]+1), 2) for k in all_category_values}})     # Show P(word|category)
    table.sort(key=lambda x: x["Total"], reverse=True)
    return pd.DataFrame(table[:app_config.MAX_WORDS_AND_PHRASES])


def embed_responses(df, q, split_sentences=True, ignore_names=False, compute_2d_points=True):
    # Split raw responses into sentences and embed
    parent_records = []
    all_sentences = []
    all_embeddings = []
    for _, row in df.iterrows():
        if split_sentences:
            doc = app_config.CONFIG["nlp"](row[q])
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            sentences = [row[q].strip()]

        for cleaned_sent in sentences:
            if cleaned_sent:
                parent_records.append(dict(row))
                all_sentences.append(cleaned_sent)

    if ignore_names:
        sentences_to_encode = [re.sub(r"\b[A-Z][a-z]+\b", "", s) for s in all_sentences]
    else:
        sentences_to_encode = all_sentences
    all_embeddings = app_config.CONFIG["model"].encode(sentences_to_encode)

    if len(all_embeddings) == 0:
        st.warning("No responses found for *%s*." % (q))
        st.stop()

    # UMAP everything
    if compute_2d_points:
        all_umap_emb = um.UMAP(n_components=2, metric="euclidean").fit_transform(
            all_embeddings
        )
    else:
        all_umap_emb = None

    return all_sentences, all_umap_emb, parent_records, all_embeddings


def cluster_data(full_embs, min_cluster_size):
    mid_umap = um.UMAP(
        # Note: UMAP seems to require that k <= N-2
        n_components=min(50, len(full_embs) - 2),
        metric="euclidean",
    ).fit(full_embs)
    mid_umap_embs = mid_umap.transform(full_embs)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(min_cluster_size, len(full_embs) - 1)
    )
    clusterer.fit(mid_umap_embs)

    # Renumber clusters, most common to least common (with -1 last)
    counts = defaultdict(lambda: 0)
    for c in clusterer.labels_:
        counts[str(c)] += 1
    sorted_labels = list(counts.keys())

    sorted_labels.sort(key=lambda x: ((x == "-1" and -1) or counts[x]), reverse=True)
    final_labels = []
    for label in clusterer.labels_:
        label = str(label)
        if label == "-1":  # unknown cluster
            final_labels.append(app_config.UNCLUSTERED_NAME)
        else:
            cluster_name = "Cluster %d" % (sorted_labels.index(label) + 1)
            final_labels.append(cluster_name)
    return {"labels": final_labels, "clusterer": clusterer, "mid_umap": mid_umap}
