#!/user/bin/env python3


CONFIG = {}
CLUSTER_OPTION_TEXT = "[Auto-pick colors based on the topic of the response text]"

TITLE = "Feedback Map"
CATEGORICAL_QUESTIONS_BGCOLOR = "lightyellow"
MAX_VALUES_TO_SUMMARIZE = 20
QUALTRICS_SURVEY_URL = "https://neu.co1.qualtrics.com/jfe/form/SV_eyQf1JSVWeVhx1I"
SURVEY_CSS = "<style> .big-font { font-size:30px !important ; padding: 5px; background-color: lightgreen; } </style>"

# Set to true to run without transformers or spacy
MOCK_MODE = False

# Set to false if you don't have an OpenAI API key.  This will cause
# placeholders to be printed for the summarization features.
USE_GPT3 = True

GPT3_MODEL_SHORT = "text-curie-001"
GPT3_PROMPT_SHORT = "Briefly summarize these responses."

GPT3_MODEL_LONG = "text-davinci-003"
GPT3_PROMPT_LONG = "Briefly summarize these responses."

# See https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_TOKENS = {"text-curie-001": 2048, "text-davinci-003": 4096}

# The maximum number of rows for the sentence embedding scatterplot.
# For larger data sets, rows will be randomly sampled to select the data to plot.
MAX_ROWS_FOR_ANALYSIS = 5000
