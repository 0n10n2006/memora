from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        print(
            f"[MEMORA] Loading embedding model: "
            f"{model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        print(
            "[MEMORA] Embedding model ready."
        )


    def embed(self, text):

        vector = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return vector.tolist()


    def embed_many(self, texts):

        vectors = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return vectors.tolist()