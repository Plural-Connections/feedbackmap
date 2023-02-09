#!/user/bin/env python3

import openai
import streamlit as st

_SAMPLE_SIZE = 50

def get_summary(df, column):
    preamble = "Here are some responses to the question \"%s\"" % (column)
    instructions = "Briefly summarize these responses."
    # TODO:  nonnull/nonempty only
    df = df[df[column].str.len() > 0]
    df = df.sample(min(_SAMPLE_SIZE, len(df)), random_state=42)
    prompt = (
        preamble
        + "\n".join([("- " + s) for s in df[column]])
        + "\n\n"
        + instructions
        + "\n"
    )
    response = run_completion_query(prompt)
    answer = set([c["text"] for c in response["choices"]])
    answer = "\n".join(list(answer))
    return {"instructions": instructions, "answer": answer}


@st.cache
def run_completion_query(prompt, temperature=0.0, num_to_generate=1, echo_prompt=False):
    tries = 0
    while tries < 3:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=temperature,
                echo=echo_prompt,
                logprobs=5,
                n=num_to_generate,
                stop=["\n\n"],
                max_tokens=echo_prompt and 1 or 200,
            )
            return response
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            st.write(e)
            tries += 1
            time.sleep(10)
