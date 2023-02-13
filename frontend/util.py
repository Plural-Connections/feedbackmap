import streamlit as st


def include_markdown(fname, replace={}, highlight_numbers=False):
    text = open("static/" + fname + ".md").read()
    for k, v in replace.items():
        if highlight_numbers and v[0].isnumeric() or v[0].startswith("-"):
            v = '<mark style="background-color: #fdfd96;">' + v + "</mark>"
        text = text.replace("${%s}" % (k.upper()), v)
    st.markdown(text, unsafe_allow_html=True)
