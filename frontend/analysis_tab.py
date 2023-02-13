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
                parent_records.append(row)
                all_sentences.append(cleaned_sent)
    all_embeddings = app_config.CONFIG["model"].encode(all_sentences)
    # UMAP everything
    all_umap_emb = um.UMAP(n_components=2, metric="euclidean").fit_transform(
        all_embeddings
    )

    return all_sentences, all_umap_emb, parent_records, all_embeddings

def cluster_data(full_embs):
    mid_umap_embs = um.UMAP(n_components=50, metric="euclidean").fit_transform(
        full_embs
    )
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    clusterer.fit(mid_umap_embs)    
    return list(clusterer.labels_)

def get_color_key_of_interest(categories):
    res = st.selectbox(
        "Group the points based on the respondent's answer to:",
        [app_config.CLUSTER_OPTION_TEXT] + list(categories.keys()),
        format_func=lambda x: str(x),
        index=1,
    )
    return res

def run(columns_to_analyze, df, categories):
    st.subheader(", ".join(columns_to_analyze))
    # Compute GPT3-based summary
    with st.spinner():
        # Layout expanders
        overall_summary_expander = st.expander(
            "Auto-generated summary of the answers", expanded=True
        )
        color_key = get_color_key_of_interest(categories)
        scatterplot_expander = st.expander(
            "Topic scatterplot of the sentences in the answers to \"%s\"" % (columns_to_analyze[0]),
            expanded=True
        )
        value_table_expander = st.expander("Summary table, broken out by answer to \"%s\"" % (color_key), expanded=True)

        # Overall summary
        with overall_summary_expander:
            res = app_config.CONFIG["llm"].get_summary(df, columns_to_analyze[0])
            st.write("**%s** %s" % (res["instructions"], res["answer"]))

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
                if color_key == app_config.CLUSTER_OPTION_TEXT:
                    cluster_labels = cluster_data(full_embs)
                    for i, x in enumerate(data):
                        x["rec"][app_config.CLUSTER_OPTION_TEXT] = "Cluster %s" % (
                            cluster_labels[i]
                        )
                    # Rewrite df to be based on the sentences, rather than responses
                    df = pd.DataFrame([x["rec"] for x in data])
                scatterplot = charts.make_scatterplot_base(data, color_key)
                st.altair_chart(scatterplot)

        # Per-value summary table
        with value_table_expander:
            table = []
            values = list(categories.get(color_key, {}).keys())
            summaries = app_config.CONFIG["llm"].get_summaries(
                df,
                columns_to_analyze[0],
                color_key,
                values[: app_config.MAX_VALUES_TO_SUMMARIZE],
            )

            overall_nonempty_rate = (
                100.0 * (df[columns_to_analyze[0]] != "").sum() / len(df)
            )
            nonempty_color = (
                lambda val: float(val.replace("%", "")) > overall_nonempty_rate
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
                num_responses = categories.get(color_key, {}).get(values[i], 0)
                nonempty_rate = 100.0 * res["nonempty_responses"] / num_responses
                nonempty_rate_color = (
                    (nonempty_rate > overall_nonempty_rate) and "green" or "red"
                )
                nonempty_rate = str(round(nonempty_rate, 1)) + "%"
                table.append(
                    {
                        'Answer to "%s"' % (color_key): values[i],
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
