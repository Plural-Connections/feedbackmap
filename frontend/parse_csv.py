#!/user/bin/env python3

"""
Logic related to understanding the survey inputs;  no streamlit in here.
"""

from collections import defaultdict
import json
import re
from io import StringIO

import pandas as pd

import app_config

# Do not treat these columns in the input as either categorical or free-response questions
_SKIP_COLUMNS = ["Timestamp"]

# If a row has more than this many unique values (relative to total)
# consider it to be a free-response text field
_MAX_FRACTION_FOR_CATEGORICAL = 0.2

def process_file(uploaded_file):
    table = []
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # Infer file type.  If first line begins and ends in braces, assume it's
    # a JSONL file.  Otherwise, try CSV.  If that fails, read as a single text column
    # (Note that a text file without any commas will parse as a CSV and will use
    # the first line as the column name.)
    first_line = stringio.readline().strip()
    if first_line.startswith("{") and first_line.endswith("}"):
        for line in stringio:
            table.append(json.loads(line))
    else:
        try:
            return pd.read_csv(uploaded_file, dtype=str).fillna("")
        except Exception as e:
            table.append({app_config.COLUMN_NAME_FOR_TEXT_FILES: first_line})
            for line in stringio:
                table.append({app_config.COLUMN_NAME_FOR_TEXT_FILES: line})

    return pd.DataFrame(table, dtype=str)

def infer_column_types(df):
    categories = {}  # column -> val_dict
    text_responses = {}  # column -> val_dict
    for column in df.columns:
        if column not in _SKIP_COLUMNS:
            val_dict = val_dictionary_for_column(df, column)
            # If it's got more than _MAX_FRACTION_FOR_CATEGORICAL * numrows different vals,
            # consider it a text response field, otherwise it's a categorical attribute
            if len(val_dict) < _MAX_FRACTION_FOR_CATEGORICAL * len(df.index):
                categories[column] = val_dict
            else:
                text_responses[column] = val_dict
    return categories, text_responses


def split_values(s):
    return [x.strip() for x in re.split(";|,", s)]


def val_dictionary_for_column(df, col):
    # Pull the val:count dict for a given column, accounting for comma-separated multi-values
    vals = defaultdict(lambda: 0)
    for index, row in df.iterrows():
        this_vals = split_values(row[col])
        for val in this_vals:
            vals[val] += 1
    return vals
