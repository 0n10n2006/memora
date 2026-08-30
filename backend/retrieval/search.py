from backend.retrieval.embedder import Embedder
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.reranker import MemoryReranker
from backend.memory.memory_store import MemoryStore


class MemorySearch:

    def __init__(self):

        self.embedder = Embedder()
        self.store = VectorStore()
        self.reranker = MemoryReranker()
        self.memory_store = MemoryStore()

    # =====================================================
    # BASIC SEARCH
    # =====================================================

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

        # -------------------------------------------------
        # STEP 1 — Semantic retrieval
        # -------------------------------------------------

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

        # -------------------------------------------------
        # STEP 2 — Deduplicate chunks into memories
        # -------------------------------------------------

        unique_memories = {}

        for i, document in enumerate(
            documents
        ):

            metadata = metadatas[i]

            memory_id = metadata.get(
                "memory_id"
            )

            if not memory_id:

                memory_id = metadata.get(
                    "source",
                    f"memory_{i}"
                )

            if memory_id not in unique_memories:

                unique_memories[
                    memory_id
                ] = {

                    "document":
                        document,

                    "metadata":
                        metadata,

                    "distance":
                        distances[i]
                }

        unique_documents = [
            item["document"]
            for item
            in unique_memories.values()
        ]

        # -------------------------------------------------
        # STEP 3 — Reranking
        # -------------------------------------------------

        reranked = self.reranker.rerank(
            query,
            unique_documents,
            top_k=top_k
        )

        # -------------------------------------------------
        # STEP 4 — Reattach metadata
        # -------------------------------------------------

        final_results = []

        for document, score in reranked:

            matching_memory = None

            for item in unique_memories.values():

                if item["document"] == document:

                    matching_memory = item

                    break

            if matching_memory is None:
                continue

            final_results.append({

                "document":
                    document,

                "score":
                    float(score),

                "metadata":
                    matching_memory[
                        "metadata"
                    ],

                "distance":
                    matching_memory[
                        "distance"
                    ]
            })

        print(
            f"[MEMORA] Retrieved "
            f"{len(final_results)} unique memories "
            f"from {len(documents)} chunks."
        )

        return final_results

    # =====================================================
    # RELATIONSHIP EXPANSION
    # =====================================================

    def expand_relationships(
        self,
        primary_results,
        max_related=5
    ):

        if not primary_results:
            return []

        expanded = []

        seen_memory_ids = set()

        # -------------------------------------------------
        # PRIMARY MEMORIES
        # -------------------------------------------------

        for result in primary_results:

            metadata = result.get(
                "metadata",
                {}
            )

            memory_id = metadata.get(
                "memory_id"
            )

            if memory_id:

                seen_memory_ids.add(
                    memory_id
                )

            result["retrieval_type"] = (
                "primary"
            )

            expanded.append(
                result
            )

        # A relationship can be relevant even when both endpoints already
        # ranked as primary results. Keep its metadata on the result instead
        # of losing it merely because no graph expansion is needed.
        primary_ids = {
            result.get("metadata", {}).get("memory_id")
            for result in primary_results
        }

        primary_ids.discard(None)

        for result in primary_results:

            memory_id = result.get("metadata", {}).get("memory_id")
            memory = self.memory_store.get(memory_id) if memory_id else None

            if not memory:
                continue

            for relationship in getattr(memory, "relationships", []):

                target_id = getattr(relationship, "target_id", None)

                if target_id not in primary_ids:
                    continue

                result["relationship"] = {
                    "target_id": target_id,
                    "strength": getattr(relationship, "strength", "unknown"),
                    "type": getattr(relationship, "relationship_type", "semantic"),
                    "relationship_type": getattr(relationship, "relationship_type", "semantic"),
                    "score": float(getattr(relationship, "score", 0.0)),
                    "shared_topics": list(getattr(relationship, "shared_topics", [])),
                    "shared_entities": list(getattr(relationship, "shared_entities", [])),
                    "temporal_relation": getattr(relationship, "temporal_relation", ""),
                    "contradictions": list(getattr(relationship, "contradictions", [])),
                    "evidence": list(getattr(relationship, "evidence", [])),
                }
                break

        # -------------------------------------------------
        # FOLLOW MEMORY RELATIONSHIPS
        # -------------------------------------------------

        related_count = 0

        for result in primary_results:

            metadata = result.get(
                "metadata",
                {}
            )

            memory_id = metadata.get(
                "memory_id"
            )

            if not memory_id:
                continue

            memory = self.memory_store.get(
                memory_id
            )

            if not memory:
                continue

            relationships = getattr(
                memory,
                "relationships",
                []
            )

            for relationship in relationships:

                if related_count >= max_related:
                    break

                target_id = getattr(
                    relationship,
                    "target_id",
                    None
                )

                if not target_id:
                    continue

                if target_id in seen_memory_ids:
                    continue

                target_memory = (
                    self.memory_store.get(
                        target_id
                    )
                )

                if not target_memory:
                    continue

                # -------------------------------------------------
                # Relationship metadata
                # -------------------------------------------------

                relationship_score = float(
                    getattr(
                        relationship,
                        "score",
                        0.0
                    )
                )

                relationship_strength = (
                    getattr(
                        relationship,
                        "strength",
                        "unknown"
                    )
                )

                relationship_type = (
                    getattr(
                        relationship,
                        "relationship_type",
                        "semantic"
                    )
                )

                shared_topics = list(
                    getattr(
                        relationship,
                        "shared_topics",
                        []
                    )
                )

                shared_entities = list(
                    getattr(
                        relationship,
                        "shared_entities",
                        []
                    )
                )

                evidence = list(
                    getattr(
                        relationship,
                        "evidence",
                        []
                    )
                )

                # -------------------------------------------------
                # Related document
                # -------------------------------------------------

                document = (
                    target_memory.content
                    or target_memory.summary
                    or target_memory.description
                    or ""
                )

                related_result = {

                    "document":
                        document,

                    # Relationship score is kept
                    # separate from reranker score.
                    "score":
                        relationship_score,

                    "relationship_score":
                        relationship_score,

                    "metadata": {

                        "source":
                            target_memory.source,

                        "modality":
                            target_memory.modality,

                        "memory_id":
                            target_memory.id,

                        "title":
                            target_memory.title,

                        "summary":
                            target_memory.summary,

                        "description":
                            target_memory.description,

                        "topics":
                            list(
                                getattr(
                                    target_memory,
                                    "topics",
                                    []
                                )
                            ),

                        "entities":
                            list(
                                getattr(
                                    target_memory,
                                    "entities",
                                    []
                                )
                            )
                    },

                    "distance":
                        None,

                    "retrieval_type":
                        "related",

                    "relationship": {

                        "target_id":
                            target_id,

                        "strength":
                            relationship_strength,

                        "type":
                            relationship_type,

                        "relationship_type":
                            relationship_type,

                        "score":
                            relationship_score,

                        "shared_topics":
                            shared_topics,

                        "shared_entities":
                            shared_entities,

                        "temporal_relation": getattr(
                            relationship,
                            "temporal_relation",
                            ""
                        ),

                        "contradictions": list(
                            getattr(
                                relationship,
                                "contradictions",
                                []
                            )
                        ),

                        "evidence":
                            evidence
                    }
                }

                expanded.append(
                    related_result
                )

                seen_memory_ids.add(
                    target_id
                )

                related_count += 1

            if related_count >= max_related:
                break

                print(
            f"[MEMORA] Relationship expansion "
            f"added {related_count} related memories."
        )

        return expanded

    # =====================================================
    # RELATIONSHIP-AWARE SEARCH
    # =====================================================

    def search_with_relationships(
        self,
        query,
        retrieval_k=20,
        top_k=3,
        max_related=5
    ):

        print(
            "\n[MEMORA] Relationship-aware search"
        )

        primary_results = self.search(
            query,
            retrieval_k=retrieval_k,
            top_k=top_k
        )

        if not primary_results:
            return []

        return self.expand_relationships(
            primary_results,
            max_related=max_related
        )


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    search_engine = MemorySearch()

    print(
        "\n========================================"
    )

    print(
        "        MEMORA MEMORY SEARCH"
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

            metadata = result[
                "metadata"
            ]

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
