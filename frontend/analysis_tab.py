from collections import defaultdict

import pandas as pd
import streamlit as st

import app_config
import charts
import local_models
import util
import logger
import random
import re

_RESPONSE_RATE_TEXT = "Response rate for that question"


@st.cache_data(persist=True)
def embed_responses(df, q, split_sentences=True, ignore_names=False):
    return local_models.embed_responses(df, q, split_sentences, ignore_names)


@st.cache_data(persist=True)
def cluster_labels(full_embs, min_cluster_size):
    return local_models.cluster_data(full_embs, min_cluster_size)["labels"]


@st.cache_data
def get_llm_summary(
    df,
    column,
    facet_column=None,
    facet_val=None,
    prompt=app_config.DEFAULT_PROMPT,
    temperature=0.0,
):
    return app_config.CONFIG["llm"].get_summary(
        df,
        column,
        facet_column=facet_column,
        facet_val=facet_val,
        prompt=prompt,
        temperature=temperature,
    )


@st.cache_data
def get_llm_summaries(
    df, question_column, facet_column, facet_values, prompt=None, temperature=0.0
):
    return app_config.CONFIG["llm"].get_summaries(
        df,
        question_column,
        facet_column,
        facet_values,
        prompt=prompt,
        temperature=temperature,
    )


def survey_teaser():
    st.markdown(
        app_config.SURVEY_CSS
        + '<p class="big-font">Was this helpful? <a href="%s" target="_blank">Share your feedback on Feedback Map!</p>'
        % (app_config.SURVEY_URL),
        unsafe_allow_html=True,
    )


def value_summary_table(
    df,
    columns_to_analyze,
    grouping_key,
    categories,
    category_values,
    color_scheme,
    split_sentences,
):
    st.write(
        'Below is a summary of %s broken down by answers to the categorical question: "%s"'
        % ((split_sentences and "sentences" or "full responses"), grouping_key)
    )

    table = []
    cluster_prompt = get_cluster_prompt()
    summaries = get_llm_summaries(
        df,
        columns_to_analyze[0],
        grouping_key,
        category_values[: app_config.MAX_VALUES_TO_SUMMARIZE],
        prompt=cluster_prompt,
    )

    # Color the "Response rate" column based on whether it's above or
    # below average response rate
    overall_nonempty_rate = 100.0 * (df[columns_to_analyze[0]] != "").sum() / len(df)
    nonempty_color = (
        lambda val: float(val.replace("%", "")) >= overall_nonempty_rate
        and "background-color: lightgreen"
        or "background-color: pink"
    )
    # Color the leftmost column to coincide with the scatterplot's colors
    scatterplot_color = lambda val: "font-weight: bold; background-color: %s" % (
        color_scheme.get(val, "white")
    )

    for i, res in enumerate(summaries):
        num_responses = categories.get(grouping_key, {}).get(category_values[i], 0)
        nonempty_rate = 100.0 * res["nonempty_responses"] / num_responses
        nonempty_rate_color = (
            (nonempty_rate > overall_nonempty_rate) and "green" or "red"
        )
        nonempty_rate = str(round(nonempty_rate, 1)) + "%"
        table.append(
            {
                "Categorical response": category_values[i],
                "Number of respondees": num_responses,
                'Auto-generated summary for answers to "%s"'
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
            table_df = table_df.applymap(nonempty_color, subset=[_RESPONSE_RATE_TEXT])

    st.table(table_df)


def top_words_table(phrase_df, grouping_key, categories):
    st.write(
        "The table below shows the key words and phrases found in the texts. "
        + 'The "Total" column shows the number of occurrences in all texts, and the other columns '
        + "show the number of occurrences in texts within each cluster.   You can change the "
        + "way clusters are chosen in the selectbox above the scatterplot above.  The maximum "
        + "clusters for each term are highlighted in green.  Click on a column header twice to sort "
        + "the column by count, and thereby see the most popular key terms in the cluster."
    )
    cols = list(phrase_df.columns.values)[2:]  # Term, total...
    if grouping_key == app_config.CLUSTER_OPTION_TEXT:
        # Sort numerically, with Unclustered at the end
        cols.sort(
            key=lambda x: int(
                x.replace("Cluster ", "").replace(app_config.UNCLUSTERED_NAME, "1000")
            )
        )
    else:
        # Sort by popularity
        cols.sort(key=lambda x: categories[grouping_key].get(x, 0), reverse=True)
    phrase_df = phrase_df[["Term", "Total"] + cols]
    # TODO:  Color the header columns according to the chart.   Hide index.

    # See:  https://issues.streamlit.app/?issue=gh-5953
    st.dataframe(
        phrase_df.style.highlight_max(
            color="lightgreen",
            subset=[c for c in cols[:50] if c != app_config.UNCLUSTERED_NAME],
            axis=1,
        )
    )


def get_cluster_size(full_embs):
    default_cluster_size = max(5, len(full_embs) // 500)
    cluster_size_choices = list(set([3, 5, 10, 20, 50, 100] + [default_cluster_size]))
    cluster_size_choices.sort()
    cluster_size = st.selectbox(
        "Minimum size of auto-generated clusters",
        cluster_size_choices,
        cluster_size_choices.index(default_cluster_size),
    )
    logger.log(
        action="SETTING_MIN_CLUSTER_SIZE",
        extra_data={"cluster_size": cluster_size},
    )
    return cluster_size


def get_cluster_prompt():
    cluster_prompt = st.selectbox(
        "You can customize what kind of summary is generated.",
        [k for k in app_config.PROMPTS],
    )
    logger.log(
        action="SETTING_CLUSTER_PROMPT",
        extra_data={"cluster_prompt": cluster_prompt},
    )
    return cluster_prompt


def get_grouping_key_of_interest(categories):
    res = st.selectbox(
        "Choose how the responses will be clustered and colored below:",
        [app_config.CLUSTER_OPTION_TEXT] + list(categories.keys()),
        format_func=lambda x: (
            (x == app_config.CLUSTER_OPTION_TEXT)
            and x
            or ("Group by answer to: " + str(x))
        ),
        key="grouping_key",
    )
    logger.log(action="SETTING_GROUPING")
    return res


def run(columns_to_analyze, df, categories):
    util.hide_table_row_index()
    st.subheader(", ".join(columns_to_analyze))

    if (
        "val_restricts" in st.session_state
        and len(set(st.session_state["val_restricts"].values())) > 1
    ):
        # We're restricting this to certain category values.  Say so, and update df
        for k, v in st.session_state["val_restricts"].items():
            if v != app_config.NO_RESTRICT_TITLE:
                st.info("Restricting to %s = %s" % (k, v))
                df = df[df[k] == v]

    if len(df) > app_config.MAX_ROWS_FOR_ANALYSIS:
        st.warning(
            "We have sampled %d random rows from the data for the following analysis"
            % (app_config.MAX_ROWS_FOR_ANALYSIS)
        )
        df = df.sample(app_config.MAX_ROWS_FOR_ANALYSIS, random_state=42)

    # Layout expanders
    overall_summary_expander = st.expander(
        "**Auto-generated summary of responses to the above question**.  Prompt: %s"
        % (app_config.PROMPTS[app_config.DEFAULT_PROMPT]["prompt"]),
        expanded=True,
    )
    with st.expander("**Configuration for the analysis below**", expanded=True):
        get_grouping_key_of_interest(categories)
        grouping_key = st.session_state["grouping_key"]
        if grouping_key == app_config.CLUSTER_OPTION_TEXT:
            st_cluster_size = st.empty()
        split_sentences = st.checkbox(
            "Treat each sentence separately",
            value=False,
            help="If this is selected, one dot will be plotted below for each *sentence* in each response.  If it's not selected, one dot will be plotted per response.",
            on_change=logger.log,
            kwargs=dict(action="TREAT_SENTENCES_SEPARATELY"),
        )
        ignore_names = st.checkbox(
            "Ignore names for scatterplot",
            value=False,
            help="If this is selected, the scatterplot will ignore the capitalized words in the responses, so that the positioning is not influenced by the names mentioned.",
            on_change=logger.log,
            kwargs=dict(action="TREAT_SENTENCES_SEPARATELY"),
        )

    # If CLUSTER_OPTION_TEXT is selected, we'll re-set this later.
    category_values = list(categories.get(grouping_key, {}).keys())

    scatterplot_expander = st.expander(
        "**Topic map**",
        expanded=True,
    )
    interesting_examples_summary_expander = st.expander(
        '**Some interesting responses (AI-selected, using the prompt: "%s")**'
        % (app_config.PROMPTS[app_config.UNUSUAL_PROMPT]["prompt"]),
        expanded=True,
    )
    value_table_expander = st.expander("**Categorical breakdown**", expanded=True)
    top_words_expander = st.expander("**Top words and phrases**", expanded=True)

    # Overall summary
    with overall_summary_expander:
        with st.spinner():
            res = get_llm_summary(df, columns_to_analyze[0])
            st.write("%s" % (res["answer"]))

    # Compute embeddings and plot scatterplot
    with scatterplot_expander:
        st.write(
            "Each dot below represents a %s from the selected open-ended question.  Dots that are close together are likely to be about similar topics."
            % (split_sentences and "sentence" or "full response")
        )
        with st.spinner():
            data = []
            for q in columns_to_analyze:
                sents, embs, parent_records, full_embs = embed_responses(
                    df, q, split_sentences, ignore_names
                )
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
                with st_cluster_size:
                    cluster_size = get_cluster_size(full_embs)
                cluster_result = cluster_labels(full_embs, cluster_size)
                cluster_label_counts = defaultdict(lambda: 0)
                for i, x in enumerate(data):
                    cluster_label_counts[cluster_result[i]] += 1
                    x["rec"][app_config.CLUSTER_OPTION_TEXT] = cluster_result[i]

                # Rewrite df to be based on the scatterplot data
                # (which might have been split by sentence.)
                df = pd.DataFrame([x["rec"] for x in data])
                # TODO:  Is this working correctly when split-by-sentences in on?
                category_values = list(cluster_label_counts.keys())
                categories[app_config.CLUSTER_OPTION_TEXT] = dict(cluster_label_counts)

            phrase_df = local_models.get_top_phrases(data, grouping_key)
            cluster_to_top_terms = local_models.get_highest_mi_words(phrase_df)

            with scatterplot_placeholder:
                scatterplot, color_scheme = charts.make_scatterplot(
                    data, grouping_key, categories.keys(), cluster_to_top_terms
                )
                st.altair_chart(scatterplot)

        # Sort category values by popularity
        category_values.sort(key=lambda x: categories[grouping_key][x], reverse=True)

    # "Interesting responses" summary
    with interesting_examples_summary_expander:
        with st.spinner():
            res = get_llm_summary(
                df,
                columns_to_analyze[0],
                prompt=app_config.UNUSUAL_PROMPT,
                temperature=(("regenerate" in st.session_state) and random.random())
                or 0.0,
            )
            st.write("%s" % (res["answer"]))
            generate_again = st.button("Pick again")
            if generate_again:
                st.session_state["regenerate"] = 1
                st.experimental_rerun()

    # Per-value summary table
    with value_table_expander:
        value_summary_table(
            df,
            columns_to_analyze,
            grouping_key,
            categories,
            category_values,
            color_scheme,
            split_sentences,
        )

    # Top words and phrases
    with top_words_expander:
        top_words_table(phrase_df, grouping_key, categories)
        survey_teaser()
