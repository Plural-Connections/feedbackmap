#!/user/bin/env python3

from collections import defaultdict
import re

"""
Logic related to understanding the survey CSVs;  no streamlit in here.
"""


# Do not treat these columns in the input as either categorical or free-response questions
_SKIP_COLUMNS = ["Timestamp"]

# If a row has more than this many unique values (relative to total)
# consider it to be a free-response text field
_MAX_FRACTION_FOR_CATEGORICAL = 0.2


def infer_column_types(df):
    categories = {}  # column -> val_dict
    text_response_columns = []
    for column in df.columns:
        if column not in _SKIP_COLUMNS:
            val_dict = val_dictionary_for_column(df, column)
            # If it's got more than _MAX_FRACTION_FOR_CATEGORICAL * numrows different vals,
            # consider it a text response field, otherwise it's a categorical attribute
            if len(val_dict) < _MAX_FRACTION_FOR_CATEGORICAL * len(df.index):
                categories[column] = val_dict
            else:
                text_response_columns.append(column)
    return categories, text_response_columns


def split_values(s):
    return [x.strip() for x in re.split(";|,", s)]


def val_dictionary_for_column(df, col):
    # Pull the val:count dict for a given column, accounting for comma-separated multi-values
    vals = defaultdict(lambda: 0)
    for index, row in df.iterrows():
        this_vals = split_values(row[col])
        for val in this_vals:
            if not val or val.isnumeric():
                # Skip purely numeric values for now, as these are often scale
                # questions
                continue
            if val.lower() in [
                "very important",
                "moderately important",
                "very important;moderately important",
            ]:
                val = "Moderately or very important"
            vals[val] += 1
    return vals
