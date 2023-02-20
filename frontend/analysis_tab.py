from collections import defaultdict

import hdbscan
import pandas as pd
import streamlit as st
from umap import umap_ as um

import app_config
import charts

_RESPONSE_RATE_TEXT = "Response rate for that question"

@st.cache_data(persist=True)
def embed_responses(df, q):
    # Split raw responses into sentences and embed
    parent_records = []
    all_sentences = []
    all_embeddings = []
    for _, row in df.iterrows():
        doc = app_config.CONFIG["nlp"](row[q])
        for sent in doc.sents:
            cleaned_sent = sent.text.strip()
            if cleaned_sent:
                parent_records.append(dict(row))
                all_sentences.append(cleaned_sent)
    all_embeddings = app_config.CONFIG["model"].encode(all_sentences)
    # UMAP everything
    all_umap_emb = um.UMAP(n_components=2, metric="euclidean").fit_transform(
        all_embeddings
    )

    return all_sentences, all_umap_emb, parent_records, all_embeddings


@st.cache_data(persist=True)
def cluster_data(full_embs, min_cluster_size):
    mid_umap_embs = um.UMAP(
        # Note: UMAP seems to require that k <= N-2
        n_components=min(50, len(full_embs) - 2), metric="euclidean"
    ).fit_transform(full_embs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min(min_cluster_size,
                                                     len(full_embs) - 1))
    clusterer.fit(mid_umap_embs)
    return list(clusterer.labels_)

def get_cluster_size(full_embs):
    default_cluster_size = max(5, len(full_embs) // 500)
    cluster_size_choices = list(set([5, 10, 20, 50, 100] + [default_cluster_size]))
    cluster_size_choices.sort()
    cluster_size = st.selectbox("Minimum size of auto-generated clusters",
                                cluster_size_choices,
                                cluster_size_choices.index(default_cluster_size))
    return cluster_size

def get_cluster_prompt():
    cluster_prompt = st.selectbox("",
                                  [k for k in app_config.PROMPTS])
    return cluster_prompt

def get_grouping_key_of_interest(categories):
    res = st.selectbox(
        "Choose how the responses will be clustered and colored below:",
        [app_config.CLUSTER_OPTION_TEXT] + list(categories.keys()),
        format_func=lambda x: ((x == app_config.CLUSTER_OPTION_TEXT) and x
                               or ("Group by answer to: " + str(x))),
        index=0
    )
    return res


def run(columns_to_analyze, df, categories):
    st.subheader(", ".join(columns_to_analyze))
    if len(df) > app_config.MAX_ROWS_FOR_ANALYSIS:
        st.warning("We have sampled %d random rows from the data for the following analysis"
                   % (app_config.MAX_ROWS_FOR_ANALYSIS))
        df = df.sample(app_config.MAX_ROWS_FOR_ANALYSIS, random_state=42)
    # Compute GPT3-based summary
    with st.spinner():
        # Layout expanders
        overall_summary_expander = st.expander(
            "Auto-generated summary of responses to the above question:", expanded=True
        )
        grouping_key = get_grouping_key_of_interest(categories)

        # If CLUSTER_OPTION_TEXT is selected, we'll re-set this later.
        category_values = list(categories.get(grouping_key, {}).keys())

        scatterplot_expander = st.expander(
            "Each dot represents a response sentence from the selected open-ended question.  Dots that are clustered together are likely to have similar meanings.",
            expanded=True,
        )
        value_table_expander = st.expander(
            'Summary of responses broken down by answers to the following categorical question: "%s"'
            % (grouping_key),
            expanded=True,
        )

        # Overall summary
        with overall_summary_expander:
            res = app_config.CONFIG["llm"].get_summary(df, columns_to_analyze[0])
            st.write("%s" % (res["answer"]))

        # Compute embeddings and plot scatterplot
        with scatterplot_expander:
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

                scatterplot_placeholder = st.empty()
                if grouping_key == app_config.CLUSTER_OPTION_TEXT:
                    cluster_size = get_cluster_size(full_embs)
                    cluster_result = cluster_data(full_embs, cluster_size)
                    cluster_label_counts = defaultdict(lambda: 0)
                    cluster_labels = [
                        "Cluster %s" % (cluster_result[i])
                        for i in range(len(cluster_result))
                    ]
                    for i, x in enumerate(data):
                        cluster_label_counts[cluster_labels[i]] += 1
                        x["rec"][app_config.CLUSTER_OPTION_TEXT] = cluster_labels[i]

                    # Rewrite df to be based on the sentences, rather than responses
                    df = pd.DataFrame([x["rec"] for x in data])
                    category_values = list(cluster_label_counts.keys())
                    categories[app_config.CLUSTER_OPTION_TEXT] = dict(
                        cluster_label_counts
                    )
                with scatterplot_placeholder:
                    scatterplot, color_scheme = charts.make_scatterplot_base(data, grouping_key)
                    st.altair_chart(scatterplot)
            st.markdown(
                app_config.SURVEY_CSS
                + '<p class="big-font">Was this helpful? <a href="%s" target="_blank">Share your feedback on Feedback Map!</p>'
                % (app_config.QUALTRICS_SURVEY_URL),
                unsafe_allow_html=True,
            )

        # Sort category values by popularity
        category_values.sort(key=lambda x: categories[grouping_key][x], reverse=True)


        # Per-value summary table
        with value_table_expander:
            table = []
            cluster_prompt = get_cluster_prompt()
            summaries = app_config.CONFIG["llm"].get_summaries(
                df,
                columns_to_analyze[0],
                grouping_key,
                category_values[: app_config.MAX_VALUES_TO_SUMMARIZE],
                prompt=cluster_prompt
            )

            # Color the "Response rate" column based on whether it's above or
            # below average response rate
            overall_nonempty_rate = (
                100.0 * (df[columns_to_analyze[0]] != "").sum() / len(df)
            )
            nonempty_color = (
                lambda val: float(val.replace("%", "")) >= overall_nonempty_rate
                and "background-color: lightgreen"
                or "background-color: pink"
            )
            # Color the leftmost column to coincide with the scatterplot's colors
            scatterplot_color = (
                lambda val: "font-weight: bold; background-color: %s" % (
                    color_scheme.get(val, "white")))

            # CSS to inject contained in a string
            hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            for i, res in enumerate(summaries):
                num_responses = categories.get(grouping_key, {}).get(
                    category_values[i], 0
                )
                nonempty_rate = 100.0 * res["nonempty_responses"] / num_responses
                nonempty_rate_color = (
                    (nonempty_rate > overall_nonempty_rate) and "green" or "red"
                )
                nonempty_rate = str(round(nonempty_rate, 1)) + "%"
                table.append(
                    {
                        "Categorical response": category_values[i],
                        "Number of respondees": num_responses,
                        'Auto-generated summary for their answers to "%s"'
                        % (columns_to_analyze[0]): res["answer"],
                        _RESPONSE_RATE_TEXT: nonempty_rate,
                    }
                )
            table_df = pd.DataFrame(table)
            if grouping_key == app_config.CLUSTER_OPTION_TEXT:
                # Don't show this column for auto-cluster, since it's always 100%
                table_df = table_df.drop([_RESPONSE_RATE_TEXT], axis=1)
            table_df = table_df.style.applymap(
                scatterplot_color, subset=["Categorical response"]
            )
            if grouping_key != app_config.CLUSTER_OPTION_TEXT:
                table_df = table_df.applymap(
                    nonempty_color, subset=[_RESPONSE_RATE_TEXT]
                )

            st.table(table_df)

            st.markdown(
                app_config.SURVEY_CSS
                + '<p class="big-font">Was this helpful? <a href="%s" target="_blank">Share your feedback on Feedback Map!</p>'
                % (app_config.QUALTRICS_SURVEY_URL),
                unsafe_allow_html=True,
            )
