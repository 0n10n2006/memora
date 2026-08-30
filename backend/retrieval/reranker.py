from sentence_transformers import CrossEncoder


class MemoryReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        local_files_only=True
    ):

        print(
            f"[MEMORA] Loading reranker: {model_name}"
        )

        try:

            self.model = CrossEncoder(
                model_name,
                local_files_only=local_files_only
            )

        except Exception as error:

            scope = "local cache" if local_files_only else "model source"

            raise RuntimeError(
                f"MEMORA could not load reranker '{model_name}' from the "
                f"{scope}. Download it once in a connected setup, then run "
                "MEMORA with the cached model."
            ) from error

        print(
            "[MEMORA] Reranker ready."
        )

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):

        if not documents:
            return []

        pairs = [
            [query, document]
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
