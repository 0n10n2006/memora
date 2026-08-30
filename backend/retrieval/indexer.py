from datetime import datetime

from backend.ingestion.batch import ingest_folder

from backend.memory.semantic_analyzer import (
    SemanticAnalyzer
)

from backend.memory.memory_object import MemoryObject

from backend.memory.memory_store import MemoryStore

from backend.memory.topic_extractor import TopicExtractor

from backend.retrieval.chunker import chunk_text

from backend.retrieval.embedder import Embedder

from backend.retrieval.vector_store import VectorStore

from backend.retrieval.memory_text import build_searchable_text


class MemoryIndexer:

    def __init__(self):

        print(
            "[MEMORA] Initializing memory indexer..."
        )

        self.embedder = Embedder()

        self.store = VectorStore()

        self.memory_store = MemoryStore()

        self.topic_extractor = TopicExtractor()

        self.semantic_analyzer = SemanticAnalyzer()

        print(
            "[MEMORA] Memory indexer ready."
        )

    # =================================================
    # CREATE MEMORY OBJECT
    # =================================================

    def _create_memory_object(
        self,
        memory
    ):

        now = datetime.now().isoformat()

        # -----------------------------------------
        # Determine title
        # -----------------------------------------

        title = memory.metadata.get(
            "title",
            ""
        )

        if not title:

            title = memory.source.split(
                "\\"
            )[-1]

        # -----------------------------------------
        # Build raw representation
        # -----------------------------------------

        raw_content = "\n".join(
            part
            for part in [
                memory.content,
                memory.description,
                " ".join(memory.entities)
            ]
            if part
        )

        # -----------------------------------------
        # Deterministic topics
        # -----------------------------------------

        deterministic_topics = (
            self.topic_extractor.extract(
                raw_content
            )
        )

        print(
            f"[MEMORA] Deterministic topics: "
            f"{deterministic_topics}"
        )

        # -----------------------------------------
        # Semantic analysis
        # -----------------------------------------

        semantic = self.semantic_analyzer.analyze(
            raw_content,
            known_topics=deterministic_topics
        )

        semantic_topics = semantic.get(
            "topics",
            []
        )

        semantic_entities = semantic.get(
            "entities",
            []
        )

        summary = semantic.get(
            "summary",
            ""
        )

        # -----------------------------------------
        # Merge topics
        #
        # Deterministic topics are trusted.
        # Semantic topics can enrich them.
        # -----------------------------------------

        topics = self._merge_unique(
            deterministic_topics,
            semantic_topics
        )

        # -----------------------------------------
        # Merge entities
        # -----------------------------------------

        entities = self._merge_unique(
            memory.entities,
            semantic_entities
        )

        print(
            f"[MEMORA] Final topics: {topics}"
        )

        print(
            f"[MEMORA] Final entities: {entities}"
        )

        # -----------------------------------------
        # Create persistent memory
        # -----------------------------------------

        memory_object = MemoryObject(

            id=memory.id,

            source=memory.source,

            modality=memory.modality,

            title=title,

            summary=summary,

            content=memory.content,

            description=memory.description,

            entities=entities,

            topics=topics,

            metadata=memory.metadata,

            created_at=now,

            modified_at=now

        )

        return memory_object

    # =================================================
    # UNIQUE MERGE
    # =================================================

    def _merge_unique(
        self,
        first,
        second
    ):

        result = []

        seen = set()

        for value in (
            list(first or [])
            + list(second or [])
        ):

            value = str(
                value
            ).strip()

            if not value:
                continue

            key = value.lower()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                key
            )

        return result

    # =================================================
    # INDEX ONE MEMORY
    # =================================================

    def index_memory(
        self,
        memory
    ):

        print(
            f"\n[MEMORA] Creating memory object: "
            f"{memory.source}"
        )

        # -----------------------------------------
        # Create MemoryObject
        # -----------------------------------------

        memory_object = (
            self._create_memory_object(
                memory
            )
        )

        # -----------------------------------------
        # Build searchable representation
        # -----------------------------------------

        searchable_content = (
            build_searchable_text(
                memory_object
            )
        )

        if not searchable_content.strip():

            print(
                f"[MEMORA] No searchable content: "
                f"{memory.source}"
            )

            return 0

        # -----------------------------------------
        # Chunk
        # -----------------------------------------

        chunks = chunk_text(
            searchable_content,
            chunk_size=500,
            overlap=100
        )

        print(
            f"[MEMORA] Generated "
            f"{len(chunks)} chunks."
        )

        # -----------------------------------------
        # Embeddings
        # -----------------------------------------

        embeddings = self.embedder.embed_many(
            chunks
        )

        # -----------------------------------------
        # Store chunks
        # -----------------------------------------

        for i, (
            chunk,
            embedding
        ) in enumerate(
            zip(
                chunks,
                embeddings
            )
        ):

            chunk_id = (
                f"{memory_object.id}"
                f"_chunk_{i}"
            )

            metadata = {

                "source":
                    memory_object.source,

                "modality":
                    memory_object.modality,

                "memory_id":
                    memory_object.id,

                "chunk_index":
                    i,

                "title":
                    memory_object.title

            }

            self.store.add_memory(

                memory_id=chunk_id,

                text=chunk,

                embedding=embedding,

                metadata=metadata

            )

            # -----------------------------------------
            # Track chunk
            # -----------------------------------------

            memory_object.chunk_ids.append(
                chunk_id
            )

        # -----------------------------------------
        # Persist MemoryObject
        # -----------------------------------------

        self.memory_store.add(
            memory_object
        )

        print(
            f"[MEMORA] Memory saved: "
            f"{memory_object.id}"
        )

        return len(chunks)

    # =================================================
    # INDEX FOLDER
    # =================================================

    def index_folder(
        self,
        folder_path
    ):

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
                f"[MEMORA] {memory.source} "
                f"→ {chunks} chunks"
            )

        print(
            f"\n[MEMORA] Indexed "
            f"{len(memories)} memories "
            f"into {total_chunks} chunks."
        )

        print(
            f"[MEMORA] Persistent memories: "
            f"{self.memory_store.count()}"
        )


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    folder = input(
        "Enter folder to remember: "
    )

    indexer = MemoryIndexer()

    indexer.index_folder(
        folder
    )