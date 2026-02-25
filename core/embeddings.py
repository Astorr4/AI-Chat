from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_PATH

class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(str(EMBED_MODEL_PATH))

    def encode(self, text):
        return self.model.encode(text, normalize_embeddings=True)
