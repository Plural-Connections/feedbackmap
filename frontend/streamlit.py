import streamlit as st
import pandas as pd


def streamlit_app():
    with st.sidebar:
        st.title("Survey Mirror")
        uploaded_file = st.file_uploader("Upload a CSV of your Google Forms results")
        df = pd.read_csv(uploaded_file)


if __name__ == "__main__":
    streamlit_app()
