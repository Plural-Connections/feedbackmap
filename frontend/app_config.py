#!/user/bin/env python3


CONFIG = {}
CLUSTER_OPTION_TEXT = "[Auto-pick colors based on the topic of the response text]"

TITLE = "Feedback Map"
CATEGORICAL_QUESTIONS_BGCOLOR = "lightyellow"
MAX_VALUES_TO_SUMMARIZE = 20

# Set to true to run without transformers or spacy
MOCK_MODE = False

# Set to false if you don't have an OpenAI API key.  This will cause
# placeholders to be printed for the summarization features.
USE_GPT3 = True

GPT3_MODEL_SHORT = "text-curie-001"
GPT3_PROMPT_SHORT = "What 4 words describes these responses?"

GPT3_MODEL_LONG = "text-davinci-003"
GPT3_PROMPT_LONG = "Briefly summarize these responses."
