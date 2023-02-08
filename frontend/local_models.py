
from sentence_transformers import SentenceTransformer
import spacy

def get_config():
    spacy.cli.download("en_core_web_sm")
    return {
        "nlp": spacy.load("en_core_web_sm"),
        "model": SentenceTransformer("all-MiniLM-L6-v2"),
    }
