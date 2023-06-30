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

if __name__ == "__main__":
    app_config.CONFIG = local_models.get_config(False)
    # Use same gpt3 config as the webapp
    app_config.CONFIG.update(gpt3_model.get_config())
    # Bigger and better embedding model than the webapp
    app_config.EMBEDDING_MODEL = "all-mpnet-base-v2"
    # Tweak the prompt for this dataset
    app_config.PROMPTS[app_config.DEFAULT_PROMPT]["prompt"] = (
        "The above comments are from different city and school board meetings.  In no more than three words, what topics do these remarks have in common?"
    )
    # Use gpt-4 for titling the clusters
    app_config.PROMPTS[app_config.DEFAULT_PROMPT]["model"] = "gpt-4"

    split_sentences = False
    ignore_names = True

    df = pd.read_csv(INPUT_FILE)
    # Column to analyze
    column = df.columns[0]

    # Use pairs of adjacent sentences
    df["caption_pair"] = df[column] + " " + df[column].shift(-1)
    df["caption_pair"].fillna(df[column], inplace=True)
    column = "caption_pair"
    df = df.sample(NUM_TRAINING_INSTANCES, random_state=42)

    # Get cluster IDs for each point
    print("Clustering...")
    sents, _, parent_records, full_embs = local_models.embed_responses(
        df, column, split_sentences, ignore_names, False
    )
    #    cluster_labels = local_models.cluster_data(full_embs, 12)   # for 10k
    cluster_result = local_models.cluster_data(full_embs, 50)

    cluster_labels = cluster_result["labels"]
    clusterer = cluster_result["clusterer"]

    # Generate prediction data and save hdbscan model for future use
    clusterer.generate_prediction_data()
    pickle.dump(clusterer, open(INPUT_FILE + ".hdbscan_model", "wb"))

    df["cluster_id"] = cluster_labels
    df.to_csv(INPUT_FILE + "_clustered.csv")

    # Get LLM-based cluster summaries
    print("Getting summaries...")
    summaries = app_config.CONFIG["llm"].get_summaries(
        df, column, "cluster_id", set(cluster_labels)
    )
    with open(INPUT_FILE + "_cluster_names.jsonl", "w") as fs_out:
        for x in summaries:
            print(json.dumps(x), file=fs_out)
