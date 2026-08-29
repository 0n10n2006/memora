from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.reranker import MemoryReranker


class MemorySearch:

    def __init__(self):

        self.embedder = Embedder()
        self.store = VectorStore()
        self.reranker = MemoryReranker()

    def search(
        self,
        query,
        retrieval_k=10,
        top_k=3
    ):

        print(
            f'\n[MEMORA] Searching memory for:\n'
            f'"{query}"\n'
        )

        # -----------------------------------------
        # STEP 1 — Semantic retrieval
        # -----------------------------------------

        query_embedding = self.embedder.embed(
            query
        )

        results = self.store.search(
            query_embedding,
            n_results=retrieval_k
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        if not documents:
            return []

        # -----------------------------------------
        # STEP 2 — Reranking
        # -----------------------------------------

        reranked = self.reranker.rerank(
            query,
            documents,
            top_k=top_k
        )

        # -----------------------------------------
        # STEP 3 — Reattach metadata
        # -----------------------------------------

        final_results = []

        for document, score in reranked:

            original_index = documents.index(
                document
            )

            final_results.append({
                "document": document,
                "score": float(score),
                "metadata": metadatas[
                    original_index
                ],
                "distance": distances[
                    original_index
                ]
            })

        return final_results


if __name__ == "__main__":

    search_engine = MemorySearch()

    print(
        "\n========================================"
    )
    print(
        "          MEMORA MEMORY SEARCH"
    )
    print(
        "========================================"
    )

    while True:

        query = input(
            "\nWhat do you remember? "
        )

        if query.lower() in [
            "exit",
            "quit"
        ]:
            break

        results = search_engine.search(
            query
        )

        print(
            "\n========== MEMORIES ==========\n"
        )

        for i, result in enumerate(
            results
        ):

            metadata = result["metadata"]

            print(
                f"Result {i + 1}"
            )

            print(
                "Source:",
                metadata.get(
                    "source",
                    "Unknown"
                )
            )

            print(
                "Modality:",
                metadata.get(
                    "modality",
                    "Unknown"
                )
            )

            print(
                "Reranker score:",
                result["score"]
            )

            print(
                "Vector distance:",
                result["distance"]
            )

            print(
                "Content:",
                result["document"][:700]
            )

            print(
                "\n------------------------------"
            )