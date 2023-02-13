import app_config
import streamlit as st


def run(df, text_response_columns, categories):
    st.write(
        "Processed **%d** responses with **%d** text response questions and **%d** categorical questions"
        % (len(df), len(text_response_columns), len(categories))
    )
    st.write(
        "Click a text response button below to analyze the results for a specific question."
    )
    st.subheader("Text response questions:")
    buttons = {}
    for k, v in text_response_columns.items():
        btn_col, info_col = st.columns(2)
        with btn_col:
            buttons[k] = st.button(k, use_container_width=True, type="primary")
        with info_col:
            st.write(
                "%0.1f%% response rate" % (100.0 * (1.0 - (v.get("", 0.0) / len(df))))
            )
    st.subheader("Categorical questions:")
    for k, v in categories.items():
        q_col, info_col = st.columns(2)
        with q_col:
            st.markdown(
                '<p style="background-color:%s; border-radius: 5px; padding:10px">%s</p>'
                % (app_config.CATEGORICAL_QUESTIONS_BGCOLOR, k),
                unsafe_allow_html=True,
            )
        with info_col:
            st.write(
                "%d different values\\\n%0.2f selections per response"
                % (len(v), sum(v.values()) / len(df))
            )

    for k, b in buttons.items():
        if b:
            st.session_state["analyze"] = [k]
            st.experimental_rerun()
