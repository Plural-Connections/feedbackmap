#!/usr/bin/env python3

"""
This is a standalone script that runs some FeedbackMap features in batch from the command line,
namely the auto-clustering and generation of names for the corresponding auto-clusters.

Set the filename, column to ananlyze, and maximum number of rows to use, below.
"""

import json
import pickle

import app_config
import local_models
import gpt3_model
import pandas as pd

INPUT_FILE = "localview_sentences"
NUM_TRAINING_INSTANCES = 100000
TARGET_TEXT_COLUMN = "target_text"
SPLIT_SENTENCES = False
IGNORE_NAMES = True


def set_app_config():
    app_config.CONFIG = local_models.get_config(False)
    # Use same gpt3 config as the webapp
    app_config.CONFIG.update(gpt3_model.get_config())
    # Bigger and better embedding model than the webapp
    app_config.EMBEDDING_MODEL = "all-mpnet-base-v2"
    # Tweak the prompt for this dataset
    app_config.PROMPTS[app_config.DEFAULT_PROMPT][
        "prompt"
    ] = "The above comments are from different city and school board meetings.  In no more than three words, what topics do these remarks have in common?"
    # Use gpt-4 for titling the clusters
    app_config.PROMPTS[app_config.DEFAULT_PROMPT]["model"] = "gpt-4"


def embed_sentences(df):
    _, _, _, full_embs = local_models.embed_responses(
        df, TARGET_TEXT_COLUMN, SPLIT_SENTENCES, IGNORE_NAMES, False
    )
    return full_embs


def update_df(df):
    column = df.columns[0]
    # Use pairs of adjacent sentences
    df[TARGET_TEXT_COLUMN] = df[column] + " " + df[column].shift(-1)
    df[TARGET_TEXT_COLUMN].fillna(df[column], inplace=True)
    return df


if __name__ == "__main__":
    set_app_config()

    df = pd.read_csv(INPUT_FILE)

    df = update_df(df)

    df = df.sample(NUM_TRAINING_INSTANCES, random_state=42)

    # Get cluster IDs for each point
    print("Clustering...")
    full_embs = embed_sentences(df)
    #    cluster_labels = local_models.cluster_data(full_embs, 12)   # for 10k
    cluster_result = local_models.cluster_data(full_embs, 50)

    cluster_labels = cluster_result["labels"]
    clusterer = cluster_result["clusterer"]

    # Save UMAP model from clusterer
    pickle.dump(
        cluster_result["mid_umap"], open(INPUT_FILE + ".hdbscan_umap_model", "wb")
    )

    # Generate prediction data and save hdbscan model for future use
    clusterer.generate_prediction_data()
    pickle.dump(clusterer, open(INPUT_FILE + ".hdbscan_model", "wb"))

    df["cluster_id"] = cluster_labels
    df.to_csv(INPUT_FILE + "_clustered.csv")

    sys.exit(0)

    # Get LLM-based cluster summaries
    print("Getting summaries...")
    summaries = app_config.CONFIG["llm"].get_summaries(
        df, TARGET_TEXT_COLUMN, "cluster_id", set(cluster_labels)
    )
    with open(INPUT_FILE + "_cluster_names.jsonl", "w") as fs_out:
        for x in summaries:
            print(json.dumps(x), file=fs_out)
