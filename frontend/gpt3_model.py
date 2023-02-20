#!/user/bin/env python3

import openai
import streamlit as st

import re
import time
from itertools import repeat
from multiprocessing import Pool

import app_config

_MAX_SAMPLE_SIZE = 50

# For parallelizing value-specific summary queries.  Check GPT-3 API rate limit.
_NUM_PROCESSES = 10


class OfflineModel:
    def prompt_examples(self, df, column, facet_column, facet_val):
        df = df[df[column].str.len() > 0]
        if facet_column:
            df = df[
                df[facet_column].str.contains("(^|;)" + re.escape(facet_val) + "($|;)")
            ]
        return df[column]

    def canned_answer(self, examples):
        if len(examples) == 0:
            return "None of the respondees answered this question."
        elif len(examples) == 1:
            return 'There was just one nonempty answer: "%s"' % (list(examples)[0])
        else:
            return "No GPT-3 model available to summarize the %d answers" % (
                len(examples)
            )

    def get_summary(
        self, df, column, facet_column=None, facet_val=None, short_prompt=False
    ):
        examples = self.prompt_examples(df, column, facet_column, facet_val)
        return {
            "instructions": "No GPT-3 model available",
            "answer": self.canned_answer(examples),
            "nonempty_responses": len(examples),
            "sample_size": len(examples),
            "facet_column": facet_column,
            "facet_val": facet_val,
        }

    def get_summaries(
        self, df, question_column, facet_column, facet_values, short_prompt=False
    ):
        return [
            self.get_summary(df, question_column, facet_column, x) for x in facet_values
        ]


class LiveGptModel(OfflineModel):
    def get_summary(
            self, df, column, facet_column=None, facet_val=None, short_prompt=False
    ):
        model = (short_prompt and app_config.GPT3_MODEL_SHORT or app_config.GPT3_MODEL_LONG)
        if column == app_config.COLUMN_NAME_FOR_TEXT_FILES:
            # For single-column text files, do not label the examples
            preamble = ''
        else:
            preamble = 'Here are some responses to the question "%s":\n' % (column)
        if short_prompt:
            instructions = app_config.GPT3_PROMPT_SHORT
        else:
            instructions = app_config.GPT3_PROMPT_LONG
        nonempty_responses = self.prompt_examples(df, column, facet_column, facet_val)
        # See https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        max_words = app_config.MAX_TOKENS[model] / 1.5 - len(preamble) - len(instructions)

        examples = None

        max_sample_size = _MAX_SAMPLE_SIZE
        while (examples is None or len("\n".join(examples).split()) > max_words):
            examples = nonempty_responses.sample(
                min(max_sample_size, len(nonempty_responses)), random_state=42
            )
            max_sample_size -= 10

        if len(examples) <= 1:
            answer = self.canned_answer(examples)
        else:
            prompt = (
                preamble
                + "\n".join([("- " + s) for s in examples])
                + "\n\n"
                + instructions
                + "\n"
            )
            response = run_completion_query(prompt, model = model)
            answer = set([c["text"] for c in response["choices"]])
            answer = "\n".join(list(answer))
        return {
            "instructions": instructions,
            "answer": answer,
            "nonempty_responses": len(nonempty_responses),
            "sample_size": len(examples),
            "facet_column": facet_column,
            "facet_val": facet_val,
        }

    def get_summaries(
        self, df, question_column, facet_column, facet_values, short_prompt=False
    ):
        """Get a summary for each value of a facet."""
        if True:
            # Serial
            return [
                self.get_summary(
                    df, question_column, facet_column, x, short_prompt=short_prompt
                )
                for x in facet_values
            ]
        else:
            # Parallel
            with Pool(_NUM_PROCESSES) as pool:
                return list(
                    pool.starmap(
                        self.get_summary,
                        zip(
                            repeat(df),
                            repeat(question_column),
                            repeat(facet_column),
                            facet_values,
                            repeat(short_prompt),
                        ),
                    )
                )


def get_config():
    return {"llm": app_config.USE_GPT3 and LiveGptModel() or OfflineModel()}


@st.cache_data(persist=True)
def run_completion_query(prompt, model="text-davinci-003", num_to_generate=1):
    tries = 0
    while tries < 3:
        try:
            response = openai.Completion.create(
                model=model,
                prompt=prompt,
                temperature=0.0,
                n=num_to_generate,
                stop=["\n\n"],
                max_tokens=300,
            )
            return response
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            st.write(e)
            tries += 1
            time.sleep(10)
