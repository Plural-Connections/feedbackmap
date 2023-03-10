#!/user/bin/env python3


CONFIG = {}
CLUSTER_OPTION_TEXT = "Auto-cluster based on the topic of the response text"

TITLE = "Feedback Map"
ICON = ":pencil:"

CATEGORICAL_QUESTIONS_BGCOLOR = "lightyellow"
MAX_VALUES_TO_SUMMARIZE = 20
SURVEY_URL = "https://forms.gle/36usGykgjnR1HKma7"
SURVEY_CSS = "<style> .big-font { font-size:30px !important ; padding: 5px; background-color: lightgreen; } </style>"

# Set to true to run without transformers or spacy
MOCK_MODE = False

UNCLUSTERED_NAME = "Unclustered"

# Number of additional categories (other than the one of interest) to show on
# tooltips in the topic scatterplot on the analysis tab
MAX_CATEGORIES_ON_TOOLTIP = 3

# Set to false if you don't have an OpenAI API key.  This will cause
# placeholders to be printed for the summarization features.
USE_GPT3 = True

PROMPTS = {
    "One sentence summary": {
        "prompt": "Summarize these responses in one sentence.",
        "model": "text-davinci-003",
    },
    "Three words": {
        "prompt": "Summarize these responses in three words.",
        #        "model": "text-davinci-003"
        "model": "text-curie-001",
    },
    "Three adjectives": {
        "prompt": "What three adjectives best describe these responses?",
        "model": "text-davinci-003",
    },
    "Paragraph summary": {
        "prompt": "Briefly summarize these responses.",
        "model": "text-davinci-003",
    },
}

DEFAULT_PROMPT = "Paragraph summary"


# See https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_TOKENS = {"text-curie-001": 2048, "text-davinci-003": 4096}

# The maximum number of rows for the sentence embedding scatterplot.
# For larger data sets, rows will be randomly sampled to select the data to plot.
MAX_ROWS_FOR_ANALYSIS = 5000

# If the user uploads a plaintext (non-CSV) file, give the column this generic name.
COLUMN_NAME_FOR_TEXT_FILES = "Text"
