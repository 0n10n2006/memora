import chromadb


class VectorStore:

    def __init__(self, path="data/chroma"):

        print("[MEMORA] Initializing vector database...")

        self.client = chromadb.PersistentClient(
            path=path
        )

        self.collection = self.client.get_or_create_collection(
            name="memories"
        )

        print("[MEMORA] Vector database ready.")

    def add_memory(
        self,
        memory_id,
        text,
        embedding,
        metadata
    ):

        self.collection.upsert(
            ids=[memory_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata]
        )

    def search(
        self,
        embedding,
        n_results=5
    ):

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

        return results