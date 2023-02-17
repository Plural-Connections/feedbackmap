from collections import defaultdict

import hdbscan
import pandas as pd
import streamlit as st
from umap import umap_ as um

import app_config
import charts


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
def cluster_data(full_embs):
    mid_umap_embs = um.UMAP(
        n_components=min(50, len(full_embs) - 1), metric="euclidean"
    ).fit_transform(full_embs)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min(5, len(full_embs) - 1))
    clusterer.fit(mid_umap_embs)
    return list(clusterer.labels_)


def get_grouping_key_of_interest(categories):
    res = st.selectbox(
        "Color points based on responses to a categorical question OR auto-generated cluster labels:",
        [app_config.CLUSTER_OPTION_TEXT] + list(categories.keys()),
        format_func=lambda x: str(x),
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
                if grouping_key == app_config.CLUSTER_OPTION_TEXT:
                    cluster_result = cluster_data(full_embs)
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
                scatterplot = charts.make_scatterplot_base(data, grouping_key)
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
            summaries = app_config.CONFIG["llm"].get_summaries(
                df,
                columns_to_analyze[0],
                grouping_key,
                category_values[: app_config.MAX_VALUES_TO_SUMMARIZE],
                short_prompt=True,
            )

            overall_nonempty_rate = (
                100.0 * (df[columns_to_analyze[0]] != "").sum() / len(df)
            )
            nonempty_color = (
                lambda val: float(val.replace("%", "")) >= overall_nonempty_rate
                and "background-color: lightgreen"
                or "background-color: pink"
            )

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
                        "Response rate for that question": nonempty_rate,
                    }
                )
            st.table(
                pd.DataFrame(table).style.applymap(
                    nonempty_color, subset=["Response rate for that question"]
                )
            )
            st.markdown(
                app_config.SURVEY_CSS
                + '<p class="big-font">Was this helpful? <a href="%s" target="_blank">Share your feedback on Feedback Map!</p>'
                % (app_config.QUALTRICS_SURVEY_URL),
                unsafe_allow_html=True,
            )
