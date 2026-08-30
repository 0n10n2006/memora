from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2",
        local_files_only=True
    ):

        print(
            f"[MEMORA] Loading embedding model: "
            f"{model_name}"
        )

        try:

            self.model = SentenceTransformer(
                model_name,
                local_files_only=local_files_only
            )

        except Exception as error:

            scope = "local cache" if local_files_only else "model source"

            raise RuntimeError(
                f"MEMORA could not load '{model_name}' from the {scope}. "
                "Download the model once in a connected setup, then run "
                "MEMORA with the cached model."
            ) from error

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
