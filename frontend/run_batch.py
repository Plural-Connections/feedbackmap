#!/usr/bin/env python3

"""
This is a standalone script that runs some FeedbackMap features in batch from the command line,
namely the auto-clustering and generation of names for the corresponding auto-clusters.

Set the filename, column to ananlyze, and maximum number of rows to use, below.
"""

import json

import app_config
import local_models
import gpt3_model
import pandas as pd

if __name__ == "__main__":
    app_config.CONFIG = local_models.get_config(False)
    app_config.CONFIG.update(gpt3_model.get_config())

    split_sentences = False
    ignore_names = True

    df = pd.read_csv("localview_sentences.txt")
    df = df.sample(10000, random_state=42)

    # Column to analyze
    column = df.columns[0]

    # Get cluster IDs for each point
    print("Clustering...")
    sents, embs, parent_records, full_embs = local_models.embed_responses(
        df, column, split_sentences, ignore_names
    )
    cluster_labels = local_models.cluster_data(full_embs, 10)
    df["cluster_id"] = cluster_labels
    df.to_csv("clustered_data.csv")

    # Get LLM-based cluster summaries
    # See app_config.PROMPTS for possible prompts
    print("Getting summaries...")
    summaries = app_config.CONFIG["llm"].get_summaries(
        df, column, "cluster_id", set(cluster_labels), prompt="Three words"
    )
    with open("cluster_names.jsonl", "w") as fs_out:
        for x in summaries:
            print(json.dumps(x), file=fs_out)
