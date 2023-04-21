import app_config
import streamlit as st
import logger


def run(df, text_response_columns, categories):
    logger.log(
        action="DATA_STATS",
        extra_data={
            "num_responses": len(df),
            "num_open_ended_questions": len(text_response_columns),
            "num_categorical_questions": len(categories),
        },
    )
    st.write(
        "Processed **%d** responses with **%d** open-ended text response questions and **%d** categorical questions."
        % (len(df), len(text_response_columns), len(categories))
    )
    st.markdown(
        "#### Select an open-ended question that you'd like to analyze responses for:"
    )
    # st.subheader("Text response questions:")
    buttons = {}
    val_restricts = {}
    for k, v in text_response_columns.items():
        btn_col, info_col = st.columns(2)
        with btn_col:
            buttons[k] = st.button(
                k,
                use_container_width=True,
                type="primary",
                on_click=logger.log,
                kwargs=dict(action="SELECTING_OPEN_ENDED_QUESTION"),
            )
        with info_col:
            st.write(
                "%0.1f%% response rate" % (100.0 * (1.0 - (v.get("", 0.0) / len(df))))
            )
    st.markdown("#### Here are the categorical questions from your survey:")
    for k, v in categories.items():
        q_col, info_col, restrict_col = st.columns(3)
        with q_col:
            st.markdown(
                '<p style="background-color:%s; border-radius: 5px; padding:10px">%s</p>'
                % (app_config.CATEGORICAL_QUESTIONS_BGCOLOR, k),
                unsafe_allow_html=True,
            )
        with info_col:
            st.write(
                "%d different values seen\\\n%0.2f selections per response"
                % (len(v), sum(v.values()) / len(df))
            )
        if len(v) > 1:
            sorted_vals = list(v.keys())
            sorted_vals.sort(key = lambda x: v[x], reverse=True)
            with restrict_col:
                val_restricts[k] = st.selectbox(
                    "Restrict analysis", [app_config.NO_RESTRICT_TITLE] + sorted_vals,
                    format_func=lambda x: (
                        (x == app_config.NO_RESTRICT_TITLE)
                        and x
                        or ("Restrict to: %s (%d responses)" % (str(x), v[x]))
                    ),
                    label_visibility="collapsed")

    for k, b in buttons.items():
        if b:
            st.session_state["analyze"] = [k]
            st.session_state["val_restricts"] = val_restricts
            st.experimental_rerun()
