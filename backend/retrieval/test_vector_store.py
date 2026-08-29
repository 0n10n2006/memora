from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore


embedder = Embedder()

store = VectorStore()


memories = [
    {
        "id": "battery",
        "text": "Lithium-ion battery recycling involves recovering valuable metals.",
        "metadata": {
            "source": "battery.pdf",
            "modality": "document"
        }
    },
    {
        "id": "esp32",
        "text": "An ESP32 is connected to several sensors including MAX30102.",
        "metadata": {
            "source": "sensor_diagram.jpg",
            "modality": "image"
        }
    },
    {
        "id": "football",
        "text": "The football match starts tonight at 8 PM.",
        "metadata": {
            "source": "notes.txt",
            "modality": "text"
        }
    }
]


print("\n[MEMORA] Adding memories...\n")


for memory in memories:

    embedding = embedder.embed(
        memory["text"]
    )

    store.add_memory(
        memory_id=memory["id"],
        text=memory["text"],
        embedding=embedding,
        metadata=memory["metadata"]
    )


print("[MEMORA] Memories stored.")


query = "How can valuable materials be recovered from batteries?"


print("\nQUERY:")
print(query)


query_embedding = embedder.embed(query)


results = store.search(
    query_embedding,
    n_results=3
)


print("\n========== RESULTS ==========\n")


for i, document in enumerate(
    results["documents"][0]
):

    print(
        f"Result {i + 1}:"
    )

    print(
        "Text:",
        document
    )

    print(
        "Metadata:",
        results["metadatas"][0][i]
    )

    print()