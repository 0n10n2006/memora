from pathlib import Path
import re

from backend.retrieval.search import MemorySearch
from backend.retrieval.indexer import MemoryIndexer
from backend.memory.relationship_engine import RelationshipEngine
from backend.memory.memory_store import MemoryStore
from backend.ingestion.ingestor import ingest_file
from backend.reconstruction.reconstructor import MemoryReconstructor


class MemoryIngestionError(Exception):
    """
    Raised when a file was parsed but produced no usable content.
    """
    pass


class MemoryPipeline:

    def __init__(self):

        print(
            "\n[MEMORA] Starting memory pipeline..."
        )

        self.indexer = MemoryIndexer()

        self.relationship_engine = RelationshipEngine()

        self.store = MemoryStore()

        self.search = MemorySearch()

        self.reconstructor = MemoryReconstructor()

        print(
            "[MEMORA] Memory pipeline ready."
        )

    # =====================================================
    # ADD MEMORY
    # =====================================================

    def add_memory(self, file_path):

        path = Path(file_path)

        print(
            f"\n[MEMORA] Adding new memory: "
            f"{path.name}"
        )

        memory_item = ingest_file(path)

        chunk_count = self.indexer.index_memory(
            memory_item
        )

        if chunk_count == 0:

            raise MemoryIngestionError(
                f"'{path.name}' produced no extractable "
                f"content and was NOT stored."
            )

        relationships = (
            self.relationship_engine
            .discover_relationships_for(
                memory_item.id
            )
        )

        self.store._load()

        self.search.memory_store._load()

        memory_object = self.store.get(
            memory_item.id
        )

        print(
            f"[MEMORA] '{path.name}' added: "
            f"{chunk_count} chunks, "
            f"{len(relationships)} relationships."
        )

        return {
            "id": memory_item.id,
            "chunks": chunk_count,
            "relationships": relationships,
            "memory": memory_object
        }

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize(self, text):

        if not text:
            return ""

        text = text.lower()

        replacements = {

            "defechive":
                "defective",

            "defeckive":
                "defective",

            "distribubion":
                "distribution",

            "distribubien":
                "distribution",

            "distribubions":
                "distributions",

            "condihone":
                "condition",

            "condihoneal":
                "conditional",

            "condibional":
                "conditional",

            "probabiliy":
                "probability",

            "probabiliby":
                "probability",

            "paobabil":
                "probability",

            "jioim":
                "joint",

            "pare":
                "probability",
        }

        for incorrect, correct in replacements.items():

            text = text.replace(
                incorrect,
                correct
            )

        return text

    # =====================================================
    # QUERY WORDS
    # =====================================================

    def _query_terms(self, query):

        text = self._normalize(query)

        words = re.findall(
            r"[a-z]{3,}",
            text
        )

        stopwords = {
            "what",
            "were",
            "was",
            "those",
            "these",
            "about",
            "have",
            "had",
            "does",
            "with",
            "from",
            "that",
            "this",
            "into",
            "your",
            "you",
            "remember",
            "related",
            "notes",
            "question",
            "questions",
            "tell",
            "give",
            "show",
            "find",
            "the",
            "and",
            "for",
            "my",
            "how",
            "are",
            "did",
            "in",
            "on",
            "of",
            "to",
        }

        return {
            word
            for word in words
            if word not in stopwords
        }

    # =====================================================
    # RELEVANCE GATE
    # =====================================================

    def _memory_relevance(
        self,
        query,
        memory
    ):

        """
        Lightweight lexical relevance check.

        Retrieval is semantic, but semantic models can return
        conceptually nearby memories even when the requested
        subject is completely absent.

        This gate prevents unrelated memories from reaching
        the generative reconstruction model.
        """

        metadata = memory.get(
            "metadata",
            {}
        )

        document = (
            memory.get(
                "document",
                ""
            )
            or ""
        )

        topics = metadata.get(
            "topics",
            []
        )

        entities = metadata.get(
            "entities",
            []
        )

        title = metadata.get(
            "title",
            ""
        )

        summary = metadata.get(
            "summary",
            ""
        )

        searchable = self._normalize(
            " ".join([
                document,
                title,
                summary,
                " ".join(topics),
                " ".join(entities),
            ])
        )

        query_terms = self._query_terms(
            query
        )

        if not query_terms:

            return 0.0

        matched = 0

        for term in query_terms:

            if term in searchable:

                matched += 1

        return (
            matched /
            len(query_terms)
        )

    # =====================================================
    # FILTER RELEVANT MEMORIES
    # =====================================================

    def _filter_relevant_memories(
        self,
        query,
        memories
    ):

        if not memories:

            return []

        scored = []

        for memory in memories:

            relevance = (
                self._memory_relevance(
                    query,
                    memory
                )
            )

            memory["_query_relevance"] = relevance

            scored.append(
                (relevance, memory)
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        # -------------------------------------------------
        # Exact subject signal
        #
        # If none of the retrieved memories contains any
        # meaningful query term, reject the retrieval.
        # -------------------------------------------------

        meaningful = [
            memory
            for relevance, memory
            in scored
            if relevance > 0
        ]

        if not meaningful:

            print(
                "[MEMORA] Relevance gate rejected "
                "all retrieved memories."
            )

            return []

        return [
            memory
            for _, memory
            in scored
            if _ > 0
        ]

    # =====================================================
    # QUESTION RECALL DETECTION
    # =====================================================

    def _is_question_recall_query(
        self,
        query
    ):

        q = query.lower()

        patterns = [
            "what were",
            "what was",
            "which questions",
            "what questions",
            "what problems",
            "what assignments",
            "what did i ask",
            "what did i have",
            "what did i study",
            "what were those",
            "what was that",
            "remind me what",
            "do you remember what",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    # =====================================================
    # DISTRIBUTION QUERY DETECTION
    # =====================================================

    def _is_distribution_query(
        self,
        query
    ):

        q = self._normalize(
            query
        )

        return (
            "distribution" in q
            or "distributions" in q
        )

    # =====================================================
    # RELATIONSHIP QUERY DETECTION
    # =====================================================

    def _is_relationship_query(
        self,
        query
    ):

        q = query.lower()

        patterns = [
            "how are",
            "how is",
            "related",
            "relationship",
            "connection",
            "connected",
            "relation between",
        ]

        return any(
            pattern in q
            for pattern in patterns
        )

    # =====================================================
    # QUESTION EXTRACTION
    # =====================================================

    def _extract_questions(
        self,
        memories
    ):

        questions = []

        for memory in memories:

            text = self._normalize(
                memory.get(
                    "document",
                    ""
                )
            )

            if not text:

                continue

            # ---------------------------------------------
            # DEFECTIVE BOLTS
            # ---------------------------------------------

            if (
                "bolt" in text
                and "defective" in text
                and "probability" in text
            ):

                questions.append(
                    "probability of selecting or finding a defective bolt"
                )

            # ---------------------------------------------
            # REPAIR TIME
            # ---------------------------------------------

            has_repair = (
                "repair" in text
                or "repa" in text
            )

            has_hours = (
                "hour" in text
                or "hrs" in text
                or "hr" in text
            )

            if has_repair and has_hours:

                if (
                    "probability" in text
                    or "exceed" in text
                ):

                    questions.append(
                        "probability that the repair time exceeds a given number of hours"
                    )

                if "conditional" in text:

                    questions.append(
                        "conditional probability involving the repair time"
                    )

            # ---------------------------------------------
            # JOINT DISTRIBUTION
            # ---------------------------------------------

            if (
                "joint distribution" in text
                or (
                    "joint" in text
                    and "distribution" in text
                )
            ):

                questions.append(
                    "joint distribution of X and Y"
                )

            # ---------------------------------------------
            # MARGINAL DISTRIBUTION
            # ---------------------------------------------

            if "marginal" in text:

                questions.append(
                    "marginal distribution of X and/or Y"
                )

            # ---------------------------------------------
            # CONDITIONAL DISTRIBUTION
            # ---------------------------------------------

            if (
                "conditional distribution" in text
                or (
                    "conditional" in text
                    and "distribution" in text
                )
            ):

                questions.append(
                    "conditional distribution of one variable given another"
                )

            # ---------------------------------------------
            # CORRELATION
            # ---------------------------------------------

            if (
                "correlation" in text
                or "coefficient of correlation" in text
            ):

                questions.append(
                    "coefficient of correlation"
                )

        # Deduplicate

        result = []

        for question in questions:

            if question not in result:

                result.append(
                    question
                )

        return result

    # =====================================================
    # QUESTION RECALL ANSWER
    # =====================================================

    def _build_question_recall(
        self,
        query,
        memories
    ):

        questions = (
            self._extract_questions(
                memories
            )
        )

        if not questions:

            return None

        evidence = []

        for memory in memories:

            source = (
                memory.get(
                    "metadata",
                    {}
                ).get(
                    "source"
                )
            )

            if source and source not in evidence:

                evidence.append(
                    source
                )

        answer = (
            "Yes — I found the memory containing "
            "those questions."
        )

        answer += (
            "\n\nThe recognizable questions were:"
        )

        for question in questions:

            answer += (
                f"\n• {question}"
            )

        return {
            "answer": answer,
            "confidence": 0.85,
            "evidence": evidence
        }

    # =====================================================
    # DISTRIBUTION ANSWER
    # =====================================================

    def _build_distribution_answer(
        self,
        memories
    ):

        distributions = []

        known_distribution_terms = [
            "probability distribution",
            "normal distribution",
            "binomial distribution",
            "poisson distribution",
            "exponential distribution",
            "joint distribution",
            "marginal distribution",
            "conditional distribution",
        ]

        for memory in memories:

            metadata = memory.get(
                "metadata",
                {}
            )

            topics = metadata.get(
                "topics",
                []
            )

            document = self._normalize(
                memory.get(
                    "document",
                    ""
                )
            )

            combined = self._normalize(
                " ".join(topics)
                + " "
                + document
            )

            for term in known_distribution_terms:

                if term in combined:

                    if term not in distributions:

                        distributions.append(
                            term
                        )

        if not distributions:

            return None

        evidence = []

        for memory in memories:

            source = (
                memory.get(
                    "metadata",
                    {}
                ).get(
                    "source"
                )
            )

            if source and source not in evidence:

                evidence.append(
                    source
                )

        answer = (
            "The probability material covered "
            "these distributions:"
        )

        for distribution in distributions:

            answer += (
                f"\n• {distribution}"
            )

        return {
            "answer": answer,
            "confidence": 0.90,
            "evidence": evidence
        }

    # =====================================================
    # RELATIONSHIP ANSWER
    # =====================================================

    def _build_relationship_answer(
        self,
        memories
    ):

        if not memories:

            return None

        relationships = []

        for memory in memories:

            relationship = memory.get(
                "relationship"
            )

            if not relationship:

                continue

            relationships.append(
                relationship
            )

        if not relationships:

            # Fall back to topic overlap.

            topic_sets = []

            for memory in memories:

                topics = set(
                    self._normalize(
                        " ".join(
                            memory.get(
                                "metadata",
                                {}
                            ).get(
                                "topics",
                                []
                            )
                        )
                    ).split()
                )

                topic_sets.append(
                    topics
                )

            if len(topic_sets) < 2:

                return None

            shared = (
                topic_sets[0]
                & topic_sets[1]
            )

            if not shared:

                return None

            return {
                "answer":
                    "The memories are related because "
                    "they share these concepts: "
                    + ", ".join(
                        sorted(shared)
                    ),
                "confidence":
                    0.80,
                "evidence": [
                    memory.get(
                        "metadata",
                        {}
                    ).get(
                        "source",
                        "Unknown"
                    )
                    for memory in memories
                ]
            }

        # Use actual relationship metadata.

        relationship = relationships[0]

        relationship_type = (
            relationship.get(
                "relationship_type",
                relationship.get(
                    "type",
                    "semantic"
                )
            )
        )

        shared_topics = relationship.get(
            "shared_topics",
            []
        )

        shared_entities = relationship.get(
            "shared_entities",
            []
        )

        pieces = []

        if shared_topics:

            pieces.append(
                "shared topics: "
                + ", ".join(
                    shared_topics
                )
            )

        if shared_entities:

            pieces.append(
                "shared entities: "
                + ", ".join(
                    shared_entities
                )
            )

        if pieces:

            explanation = (
                "; ".join(pieces)
            )

        else:

            explanation = (
                "their semantic content is related"
            )

        answer = (
            "These memories are connected as "
            f"'{relationship_type}'. "
            f"The relationship is based on {explanation}."
        )

        return {
            "answer": answer,
            "confidence": 0.90,
            "evidence": [
                memory.get(
                    "metadata",
                    {}
                ).get(
                    "source",
                    "Unknown"
                )
                for memory in memories
            ]
        }

    # =====================================================
    # NO MEMORY ANSWER
    # =====================================================

    def _no_memory_answer(self):

        return {
            "answer":
                "I don't have a stored memory that matches that.",
            "confidence":
                0.05,
            "evidence":
                []
        }

    # =====================================================
    # REMEMBER
    # =====================================================

    def remember(self, query):

        print(
            "\n[MEMORA] Running relationship-aware "
            "memory reconstruction..."
        )

        # -------------------------------------------------
        # STEP 1 — Retrieve MORE than one primary memory.
        #
        # Previously:
        # top_k=1
        #
        # That caused the image/PDF problem because only
        # whichever memory won reranking was available to
        # deterministic reconstruction.
        # -------------------------------------------------

        memories = (
            self.search.search_with_relationships(
                query,
                retrieval_k=20,
                top_k=3,
                max_related=5
            )
        )

        print(
            f"[MEMORA] Candidate memories: "
            f"{len(memories)}"
        )

        # -------------------------------------------------
        # STEP 2 — Relevance gate
        # -------------------------------------------------

        relevant_memories = (
            self._filter_relevant_memories(
                query,
                memories
            )
        )

        # -------------------------------------------------
        # IMPORTANT:
        #
        # Relationship results are useful evidence, but
        # shouldn't independently make an unrelated query
        # look relevant.
        #
        # Keep them only after at least one primary memory
        # passes the relevance gate.
        # -------------------------------------------------

        primary_relevant = [
            memory
            for memory
            in relevant_memories
            if memory.get(
                "retrieval_type",
                "primary"
            ) == "primary"
        ]

        if not primary_relevant:

            print(
                "[MEMORA] No relevant primary memory found."
            )

            return {
                "query": query,
                "memories": [],
                "reconstruction":
                    self._no_memory_answer()
            }

        # Keep relevant primary memories and their useful
        # relationship evidence.

        relevant_ids = {
            memory.get(
                "metadata",
                {}
            ).get(
                "memory_id"
            )
            for memory in primary_relevant
        }

        relevant_ids.discard(
            None
        )

        final_memories = list(
            primary_relevant
        )

        for memory in memories:

            if memory.get(
                "retrieval_type"
            ) != "related":

                continue

            # Relationship evidence attached to a relevant
            # primary memory is useful.

            relationship = memory.get(
                "relationship",
                {}
            )

            target_id = (
                memory.get(
                    "metadata",
                    {}
                ).get(
                    "memory_id"
                )
            )

            if target_id and target_id not in relevant_ids:

                # Keep related memory because it was reached
                # from the relevant graph.
                final_memories.append(
                    memory
                )

        # Deduplicate

        seen = set()

        deduplicated = []

        for memory in final_memories:

            memory_id = (
                memory.get(
                    "metadata",
                    {}
                ).get(
                    "memory_id"
                )
            )

            if memory_id in seen:

                continue

            if memory_id:

                seen.add(
                    memory_id
                )

            deduplicated.append(
                memory
            )

        final_memories = deduplicated

        # -------------------------------------------------
        # STEP 3 — Deterministic structured queries
        # -------------------------------------------------

        if self._is_distribution_query(
            query
        ):

            print(
                "[MEMORA] Distribution query detected; "
                "using stored topic evidence."
            )

            deterministic = (
                self._build_distribution_answer(
                    final_memories
                )
            )

            if deterministic:

                return {
                    "query": query,
                    "memories": final_memories,
                    "reconstruction":
                        deterministic
                }

        # -------------------------------------------------
        # STEP 4 — Deterministic question recall
        # -------------------------------------------------

        if self._is_question_recall_query(
            query
        ):

            print(
                "[MEMORA] Question-recall query detected; "
                "using deterministic reconstruction."
            )

            deterministic = (
                self._build_question_recall(
                    query,
                    final_memories
                )
            )

            if deterministic:

                return {
                    "query": query,
                    "memories": final_memories,
                    "reconstruction":
                        deterministic
                }

        # -------------------------------------------------
        # STEP 5 — Relationship reconstruction
        # -------------------------------------------------

        if self._is_relationship_query(
            query
        ):

            print(
                "[MEMORA] Relationship query detected; "
                "using relationship metadata."
            )

            # Keep relationship wording and confidence in one place.
            # MemoryReconstructor uses relationship metadata directly for
            # this intent and does not invoke Qwen when the metadata exists.
            deterministic = self.reconstructor.reconstruct(
                query,
                final_memories
            )

            if deterministic and deterministic.get("answer"):

                return {
                    "query": query,
                    "memories": final_memories,
                    "reconstruction":
                        deterministic
                }

        # -------------------------------------------------
        # STEP 6 — Normal semantic reconstruction
        # -------------------------------------------------

        print(
            "[MEMORA] Using semantic reconstruction model."
        )

        reconstruction = (
            self.reconstructor.reconstruct(
                query,
                final_memories
            )
        )

        return {
            "query": query,
            "memories": final_memories,
            "reconstruction":
                reconstruction
        }


# =========================================================
# INTERACTIVE TEST
# =========================================================

if __name__ == "__main__":

    pipeline = MemoryPipeline()

    print(
        "\n========================================"
    )

    print(
        "          MEMORA MEMORY"
    )

    print(
        "========================================"
    )

    while True:

        try:

            query = input(
                "\nWhat do you remember? "
            )

        except (
            EOFError,
            KeyboardInterrupt
        ):

            print(
                "\n[MEMORA] Exiting."
            )

            break

        if query.lower().strip() in [
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
                f"  - {source}"
            )

        print(
            "\nRetrieved memories:"
        )

        for memory in result[
            "memories"
        ]:

            metadata = memory.get(
                "metadata",
                {}
            )

            print(
                f"  • "
                f"{memory.get('retrieval_type', 'unknown').upper()}: "
                f"{metadata.get('source', 'Unknown')}"
            )

        print(
            "\n============================\n"
        )
