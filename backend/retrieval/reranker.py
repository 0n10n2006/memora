from sentence_transformers import CrossEncoder


class MemoryReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):

        print(
            f"[MEMORA] Loading reranker: {model_name}"
        )

        self.model = CrossEncoder(
            model_name
        )

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