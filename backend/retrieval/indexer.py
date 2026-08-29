from backend.ingestion.batch import ingest_folder
from backend.retrieval.chunker import chunk_text
from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.memory_text import build_searchable_text

class MemoryIndexer:

    def __init__(self):

        self.embedder = Embedder()
        self.store = VectorStore()

    def index_memory(self, memory):

        # Combine everything currently available
        searchable_content = build_searchable_text(
            memory
        )

        if not searchable_content.strip():
            print(
                f"[MEMORA] No searchable content: "
                f"{memory.source}"
            )
            return 0

        chunks = chunk_text(
            searchable_content,
            chunk_size=500,
            overlap=100
        )

        embeddings = self.embedder.embed_many(
            chunks
        )

        for i, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            chunk_id = f"{memory.id}_chunk_{i}"

            metadata = {
                "source": memory.source,
                "modality": memory.modality,
                "memory_id": memory.id,
                "chunk_index": i
            }

            self.store.add_memory(
                memory_id=chunk_id,
                text=chunk,
                embedding=embedding,
                metadata=metadata
            )

        return len(chunks)

    def index_folder(self, folder_path):

        memories = ingest_folder(
            folder_path
        )

        total_chunks = 0

        print(
            "\n[MEMORA] Building semantic memory...\n"
        )

        for memory in memories:

            chunks = self.index_memory(
                memory
            )

            total_chunks += chunks

            print(
                f"[MEMORA] {memory.source} → "
                f"{chunks} chunks"
            )

        print(
            f"\n[MEMORA] Indexed "
            f"{len(memories)} memories "
            f"into {total_chunks} chunks."
        )


if __name__ == "__main__":

    folder = input(
        "Enter folder to remember: "
    )

    indexer = MemoryIndexer()

    indexer.index_folder(folder)