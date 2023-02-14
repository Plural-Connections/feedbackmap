#!/user/bin/env python3

import openai
import streamlit as st

import re
import time
from itertools import repeat
from multiprocessing import Pool

_SAMPLE_SIZE = 50

# For parallelizing value-specific summary queries.  Check GPT-3 API rate limit.
_NUM_PROCESSES = 10


class OfflineModel:
    def prompt_examples(self, df, column, facet_column, facet_val):
        df = df[df[column].str.len() > 0]
        if facet_column:
            df = df[
                df[facet_column].str.contains('(^|;)' + re.escape(facet_val) + '($|;)')
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

    def get_summary(self, df, column, facet_column=None, facet_val=None, short_prompt=False):
        examples = self.prompt_examples(df, column, facet_column, facet_val)
        return {
            "instructions": "No GPT-3 model available",
            "answer": self.canned_answer(examples),
            "nonempty_responses": len(examples),
            "sample_size": len(examples),
            "facet_column": facet_column,
            "facet_val": facet_val,
        }

    def get_summaries(self, df, question_column, facet_column, facet_values, short_prompt=False):
        return [
            self.get_summary(df, question_column, facet_column, x) for x in facet_values
        ]


class LiveGptModel(OfflineModel):
    def get_summary(self, df, column, facet_column=None, facet_val=None, short_prompt=False):
        preamble = 'Here are some responses to the question "%s"' % (column)
        if short_prompt:
            instructions = "What 4 words describes these responses?"
        else:
            instructions = "Briefly summarize these responses."
        nonempty_responses = self.prompt_examples(df, column, facet_column, facet_val)
        examples = nonempty_responses.sample(
            min(_SAMPLE_SIZE, len(nonempty_responses)), random_state=42
        )
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
            response = run_completion_query(prompt)
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

    def get_summaries(self, df, question_column, facet_column, facet_values, short_prompt=False):
        """Get a summary for each value of a facet."""
        if True:
            # Serial
            return [
                self.get_summary(df, question_column, facet_column, x, short_prompt=short_prompt)
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
                            repeat(short_prompt)
                        ),
                    )
                )


def get_config(mock_mode):
    return {"llm": mock_mode and OfflineModel() or LiveGptModel()}


@st.cache_data(persist=True)
def run_completion_query(prompt, num_to_generate=1):
    tries = 0
    while tries < 3:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
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
