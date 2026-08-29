from backend.retrieval.reranker import MemoryReranker


reranker = MemoryReranker()


query = (
    "Find the handwritten probability "
    "assignment about defective bolts"
)


documents = [

    "Probability and statistics study notes "
    "covering distributions and formulas.",

    "A handwritten assignment containing "
    "questions about defective bolts and "
    "conditional probability.",

    "ESP32 sensor architecture with MAX30102.",

    "A textbook chapter about probability."
]


results = reranker.rerank(
    query,
    documents,
    top_k=3
)


print(
    "\n========== RERANKED RESULTS ==========\n"
)


for i, (document, score) in enumerate(results):

    print(
        f"Result {i + 1}"
    )

    print(
        "Score:",
        float(score)
    )

    print(
        "Document:",
        document
    )

    print(
        "\n--------------------------------------\n"
    )