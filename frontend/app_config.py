#!/user/bin/env python3


CONFIG = {}
CLUSTER_OPTION_TEXT = "Auto-cluster based on the topic of the response text"

TITLE = "Feedback Map"
ICON = ":pencil:"

CATEGORICAL_QUESTIONS_BGCOLOR = "lightyellow"
SURVEY_URL = "https://forms.gle/36usGykgjnR1HKma7"
SURVEY_CSS = "<style> .big-font { font-size:30px !important ; padding: 5px; background-color: lightgreen; } </style>"

# Set to true to run without transformers or spacy
MOCK_MODE = False

UNCLUSTERED_NAME = "Unclustered"

# Number of category values to generate per-value LLM-based summaries for
# in the "Categorical breakdown" section
MAX_VALUES_TO_SUMMARIZE = 20

# Number of words and phrases to show in the "Top words and phrases" table
MAX_WORDS_AND_PHRASES = 200

# Number of additional categories (other than the one of interest) to show on
# tooltips in the topic scatterplot on the analysis tab
MAX_CATEGORIES_ON_TOOLTIP = 3

# Set to false if you don't have an OpenAI API key.  This will cause
# placeholders to be printed for the summarization features.
USE_GPT3 = True

DEFAULT_PROMPT = "Paragraph summary"
UNUSUAL_PROMPT = "Most unusual responses"

PROMPTS = {
    "One sentence summary": {
        "prompt": "In one sentence, what do these responses have in common?",
        "model": "gpt-3.5-turbo-0301",
    },
    "Three words": {
        "prompt": "Summarize all of these responses in three words.",
        "model": "gpt-3.5-turbo-0301",
    },
    UNUSUAL_PROMPT: {
        "prompt": "What are 3 interesting responses and why?",
        "model": "gpt-3.5-turbo-0301",
    },
    "Three adjectives": {
        "prompt": "What three adjectives best describe these responses?",
        "model": "gpt-3.5-turbo-0301",
    },
    DEFAULT_PROMPT: {
        "prompt": "Briefly, what do these responses have in common?",
        "model": "gpt-3.5-turbo-0301",
    },
}


# See https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_TOKENS = {"text-davinci-003": 4096, "gpt-3.5-turbo-0301": 4096}

# The maximum number of rows for the sentence embedding scatterplot.
# For larger data sets, rows will be randomly sampled to select the data to plot.
MAX_ROWS_FOR_ANALYSIS = 5000

# If the user uploads a plaintext (non-CSV) file, give the column this generic name.
COLUMN_NAME_FOR_TEXT_FILES = "Text"
