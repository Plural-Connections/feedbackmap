
import random

def get_config(mock_mode):
    """If mock_mode is set, don't load any NLP libraries."""
    return {
        "nlp": True and dumb_segmenter_model() or spacy_segmenter_model(),
        "model": mock_mode and DumbEmbeddingModel() or SentenceTransformersModel()
    }

class SentenceTransformersModel:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer("all-MiniLM-L6-v2")
    def encode(self, x):
        return self.m.encode(x)
        
class DumbEmbeddingModel:
    def __init__(self):
        pass
    def encode(self, x):
        return [[hash(s), hash(s)] for s in x]


def spacy_segmenter_model():
    import spacy
    spacy.cli.download("en_core_web_sm")
    return spacy.load("en_core_web_sm")

def dumb_segmenter_model():
    return lambda x: type('obj', (object,), {'sents' :
                                             [type('obj', (object,), {"text": y})
                                              for y in x.split(".")]})
