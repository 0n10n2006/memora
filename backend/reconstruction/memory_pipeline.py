from backend.retrieval.search import MemorySearch
from backend.reconstruction.reconstructor import MemoryReconstructor


class MemoryPipeline:

    def __init__(self):

        print(
            "\n[MEMORA] Starting memory pipeline..."
        )

        self.search = MemorySearch()

        self.reconstructor = (
            MemoryReconstructor()
        )

        print(
            "[MEMORA] Memory pipeline ready."
        )

    def remember(self, query):

        # -----------------------------
        # Retrieval + reranking
        # -----------------------------

        memories = self.search.search(
            query,
            retrieval_k=10,
            top_k=5
        )

        # -----------------------------
        # Reconstruction
        # -----------------------------

        reconstruction = (
            self.reconstructor.reconstruct(
                query,
                memories
            )
        )

        return {
            "query": query,
            "memories": memories,
            "reconstruction": reconstruction
        }


if __name__ == "__main__":

    pipeline = MemoryPipeline()

    print(
        "\n========================================"
    )

    print(
        "          🧠 MEMORA MEMORY"
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

        result = pipeline.remember(
            query
        )

        reconstruction = result[
            "reconstruction"
        ]

        print(
            "\n========== MEMORY ==========\n"
        )

        print(
            reconstruction["answer"]
        )

        print(
            "\nConfidence:",
            reconstruction["confidence"]
        )

        print(
            "\nEvidence:"
        )

        for source in reconstruction[
            "evidence"
        ]:

            print(
                f"  📎 {source}"
            )

        print(
            "\n============================\n"
        )