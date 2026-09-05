from itertools import combinations
from datetime import date

from backend.memory.memory_store import MemoryStore
from backend.retrieval.embedder import Embedder


class RelationshipEngine:

    def __init__(self):

        print(
            "[MEMORA] Initializing relationship engine..."
        )

        self.store = MemoryStore()
        self.embedder = Embedder()

        print(
            "[MEMORA] Relationship engine ready."
        )

    # =========================================
    # BUILD MEMORY TEXT
    # =========================================

    def _memory_text(self, memory):

        parts = []

        if getattr(memory, "title", ""):
            parts.append(memory.title)

        if getattr(memory, "summary", ""):
            parts.append(memory.summary)

        if getattr(memory, "description", ""):
            parts.append(memory.description)

        topics = getattr(memory, "topics", [])

        if topics:
            parts.append(" ".join(topics))

        entities = getattr(memory, "entities", [])

        if entities:
            parts.append(" ".join(entities))

        if getattr(memory, "content", ""):
            parts.append(memory.content[:3000])

        return " ".join(parts)

    # =========================================
    # NORMALIZE LIST
    # =========================================

    def _normalize(self, values):

        result = set()

        for value in values or []:

            value = str(value).strip().lower()

            if value:
                result.add(value)

        return result

    # =========================================
    # COSINE SIMILARITY
    # =========================================

    def _cosine_similarity(
        self,
        vector_a,
        vector_b
    ):

        if not vector_a or not vector_b:
            return 0.0

        length = min(
            len(vector_a),
            len(vector_b)
        )

        vector_a = vector_a[:length]
        vector_b = vector_b[:length]

        dot = sum(
            a * b
            for a, b in zip(
                vector_a,
                vector_b
            )
        )

        magnitude_a = (
            sum(
                a * a
                for a in vector_a
            )
            ** 0.5
        )

        magnitude_b = (
            sum(
                b * b
                for b in vector_b
            )
            ** 0.5
        )

        if (
            magnitude_a == 0
            or magnitude_b == 0
        ):
            return 0.0

        return (
            dot
            / (
                magnitude_a
                * magnitude_b
            )
        )

    # =========================================
    # GET EMBEDDING
    # =========================================

    def _get_embedding(self, memory):

        text = self._memory_text(memory)

        if not text.strip():
            return None

        return self.embedder.embed(text)

    # =========================================
    # TOPIC COVERAGE
    # =========================================

    def _topic_coverage(
        self,
        topics_a,
        topics_b
    ):

        topics_a = self._normalize(
            topics_a
        )

        topics_b = self._normalize(
            topics_b
        )

        if not topics_a or not topics_b:
            return 0.0, []

        smaller = (
            topics_a
            if len(topics_a) <= len(topics_b)
            else topics_b
        )

        larger = (
            topics_b
            if len(topics_a) <= len(topics_b)
            else topics_a
        )

        shared = smaller & larger

        coverage = (
            len(shared)
            / len(smaller)
        )

        return (
            coverage,
            sorted(shared)
        )

    # =========================================
    # ENTITY OVERLAP
    # =========================================

    def _entity_overlap(
        self,
        entities_a,
        entities_b
    ):

        entities_a = self._normalize(
            entities_a
        )

        entities_b = self._normalize(
            entities_b
        )

        if not entities_a or not entities_b:
            return 0.0, []

        shared = entities_a & entities_b

        union = entities_a | entities_b

        overlap = (
            len(shared)
            / len(union)
        )

        return (
            overlap,
            sorted(shared)
        )

    # =========================================
    # RELATIONSHIP TYPE
    # =========================================

    def _relationship_type(
        self,
        topic_coverage,
        entity_overlap,
        semantic_similarity,
        shared_topics,
        shared_entities
    ):

        # Explicit shared topics are the strongest
        # signal for memory relationships.

        if (
            len(shared_topics) >= 2
            or topic_coverage >= 0.50
        ):

            return "same_topic"

        if (
            shared_entities
            and entity_overlap >= 0.25
        ):

            return "shared_entity"

        if (
            shared_topics
            and topic_coverage > 0
        ):

            return "related_topic"

        if semantic_similarity >= 0.15:

            return "semantic"

        return None

    # =========================================
    # CONTEXTUAL EVIDENCE
    # =========================================

    def _event_date(self, memory):

        metadata = getattr(memory, "metadata", {}) or {}

        # These values are explicit event metadata. File creation time is
        # deliberately excluded because it is not necessarily when an event
        # happened.
        for key in ("event_date", "date", "occurred_at"):

            value = metadata.get(key)

            if not value:
                continue

            try:
                return date.fromisoformat(str(value)[:10])
            except ValueError:
                continue

        return None

    def _structured_facts(self, memory):

        metadata = getattr(memory, "metadata", {}) or {}
        raw_facts = metadata.get("facts", {})
        facts = {}

        if isinstance(raw_facts, dict):
            items = raw_facts.items()
        elif isinstance(raw_facts, list):
            items = [
                (item.get("key"), item.get("value"))
                for item in raw_facts
                if isinstance(item, dict)
            ]
        else:
            items = []

        for key, value in items:
            if not isinstance(key, str) or value is None:
                continue

            normalized_key = key.strip().lower()
            normalized_value = str(value).strip()

            if normalized_key and normalized_value:
                facts[normalized_key] = normalized_value

        return facts

    def _contextual_evidence(self, memory_a, memory_b):

        date_a = self._event_date(memory_a)
        date_b = self._event_date(memory_b)
        temporal_relation = ""

        if date_a and date_b:
            if date_a == date_b:
                temporal_relation = f"both memories refer to {date_a.isoformat()}"
            else:
                temporal_relation = (
                    f"the memories refer to {date_a.isoformat()} and "
                    f"{date_b.isoformat()}"
                )

        facts_a = self._structured_facts(memory_a)
        facts_b = self._structured_facts(memory_b)
        contradictions = []

        for key in sorted(set(facts_a) & set(facts_b)):
            if facts_a[key].casefold() != facts_b[key].casefold():
                contradictions.append(
                    f"{key}: '{facts_a[key]}' vs '{facts_b[key]}'"
                )

        return temporal_relation, contradictions

    # =========================================
    # ANALYZE RELATIONSHIP
    # =========================================

    def analyze_relationship(
        self,
        memory_a,
        memory_b
    ):

        topic_coverage, shared_topics = (
            self._topic_coverage(
                getattr(
                    memory_a,
                    "topics",
                    []
                ),
                getattr(
                    memory_b,
                    "topics",
                    []
                )
            )
        )

        entity_overlap, shared_entities = (
            self._entity_overlap(
                getattr(
                    memory_a,
                    "entities",
                    []
                ),
                getattr(
                    memory_b,
                    "entities",
                    []
                )
            )
        )

        embedding_a = self._get_embedding(
            memory_a
        )

        embedding_b = self._get_embedding(
            memory_b
        )

        semantic_similarity = (
            self._cosine_similarity(
                embedding_a,
                embedding_b
            )
        )

        temporal_relation, contradictions = self._contextual_evidence(
            memory_a,
            memory_b
        )

        # =====================================
        # COMBINED SCORE
        # =====================================

        combined_score = (
            (0.45 * topic_coverage)
            +
            (0.15 * entity_overlap)
            +
            (
                0.40
                * max(
                    semantic_similarity,
                    0.0
                )
            )
        )

        # =====================================
        # RELATIONSHIP TYPE
        # =====================================

        relationship_type = (
            self._relationship_type(
                topic_coverage,
                entity_overlap,
                semantic_similarity,
                shared_topics,
                shared_entities
            )
        )

        # =====================================
        # RELATIONSHIP STRENGTH
        # =====================================

        # Explicit topic evidence should be enough
        # to establish a relationship even when the
        # embedding model performs poorly on noisy OCR.

        if (
            len(shared_topics) >= 2
            or topic_coverage >= 0.50
        ):

            strength = "strong"

        elif (
            shared_topics
            or entity_overlap >= 0.25
            or semantic_similarity >= 0.25
        ):

            strength = "moderate"

        elif semantic_similarity >= 0.10:

            strength = "weak"

        else:

            strength = None

        # =====================================
        # EXPLANATION
        # =====================================

        reasons = []

        if shared_topics:

            reasons.append(
                "shared topics: "
                + ", ".join(
                    shared_topics
                )
            )

        if topic_coverage > 0:

            reasons.append(
                "topic coverage: "
                + f"{topic_coverage:.0%}"
            )

        if shared_entities:

            reasons.append(
                "shared entities: "
                + ", ".join(
                    shared_entities
                )
            )

        if semantic_similarity >= 0.30:

            reasons.append(
                "high semantic similarity"
            )

        elif semantic_similarity >= 0.15:

            reasons.append(
                "related semantic content"
            )

        if temporal_relation:
            reasons.append("temporal context: " + temporal_relation)

        if contradictions:
            reasons.append("conflicting structured facts detected")

        if not reasons:

            reasons.append(
                "insufficient shared evidence"
            )

        return {

            "memory_a":
                memory_a.id,

            "memory_b":
                memory_b.id,

            "score":
                combined_score,

            "strength":
                strength,

            "relationship_type":
                relationship_type,

            "semantic_similarity":
                semantic_similarity,

            "topic_coverage":
                topic_coverage,

            "entity_overlap":
                entity_overlap,

            "shared_topics":
                shared_topics,

            "shared_entities":
                shared_entities,

            "temporal_relation": temporal_relation,

            "contradictions": contradictions,

            "reasons":
                reasons
        }

    # =========================================
    # SAVE RELATIONSHIP
    # =========================================

    def _save_relationship(
        self,
        relationship
    ):

        self.store.add_relationship(

            memory_a=relationship[
                "memory_a"
            ],

            memory_b=relationship[
                "memory_b"
            ],

            relationship_type=relationship[
                "relationship_type"
            ],

            score=relationship[
                "score"
            ],

            strength=relationship[
                "strength"
            ],

            shared_topics=relationship[
                "shared_topics"
            ],

            shared_entities=relationship[
                "shared_entities"
            ],

            temporal_relation=relationship[
                "temporal_relation"
            ],

            contradictions=relationship[
                "contradictions"
            ],

            evidence=relationship[
                "reasons"
            ]
        )

    # =========================================
    # DISCOVER RELATIONSHIPS
    # =========================================

    def discover_relationships(self):

        memories = self.store.all()

        print(
            f"\n[MEMORA] Analyzing "
            f"{len(memories)} memories...\n"
        )

        relationships = []

        self.store.clear_relationships()

        for memory_a, memory_b in combinations(
            memories,
            2
        ):

            print(
                "----------------------------------------"
            )

            print(
                f"{memory_a.source} <-> "
                f"{memory_b.source}"
            )

            result = (
                self.analyze_relationship(
                    memory_a,
                    memory_b
                )
            )

            print(
                "Semantic similarity: "
                f"{result['semantic_similarity']:.4f}"
            )

            print(
                "Topic coverage: "
                f"{result['topic_coverage']:.4f}"
            )

            print(
                "Entity overlap: "
                f"{result['entity_overlap']:.4f}"
            )

            print(
                "Combined score: "
                f"{result['score']:.4f}"
            )

            if result["strength"]:

                print(
                    "Relationship: "
                    f"{result['strength'].upper()}"
                )

                print(
                    "Type: "
                    f"{result['relationship_type']}"
                )

                print(
                    "Reason:"
                )

                for reason in result[
                    "reasons"
                ]:

                    print(
                        f"  • {reason}"
                    )

                self._save_relationship(
                    result
                )

                relationships.append(
                    result
                )

            else:

                print(
                    "No relationship."
                )

        self.store._save()

        print(
            "\n========================================"
        )

        print(
            "       MEMORY RELATIONSHIPS"
        )

        print(
            "========================================\n"
        )

        if not relationships:

            print(
                "No relationships discovered."
            )

        else:

            for relationship in relationships:

                memory_a = self.store.get(
                    relationship[
                        "memory_a"
                    ]
                )

                memory_b = self.store.get(
                    relationship[
                        "memory_b"
                    ]
                )

                print(
                    memory_a.source
                )

                print(
                    "   ↕ "
                    + relationship[
                        "strength"
                    ].upper()
                    + " "
                    + relationship[
                        "relationship_type"
                    ].upper()
                    + " RELATIONSHIP"
                )

                print(
                    memory_b.source
                )

                print(
                    "   Score: "
                    + f"{relationship['score']:.4f}"
                )

                if relationship[
                    "shared_topics"
                ]:

                    print(
                        "   Shared topics: "
                        + ", ".join(
                            relationship[
                                "shared_topics"
                            ]
                        )
                    )

                if relationship[
                    "shared_entities"
                ]:

                    print(
                        "   Shared entities: "
                        + ", ".join(
                            relationship[
                                "shared_entities"
                            ]
                        )
                    )

                print()

        return relationships

    # =========================================
    # DISCOVER RELATIONSHIPS FOR ONE MEMORY
    # =========================================

    def discover_relationships_for(
        self,
        memory_id
    ):

        self.store._load()

        target = self.store.get(
            memory_id
        )

        if not target:

            print(
                f"[MEMORA] Cannot find memory "
                f"'{memory_id}' to relate."
            )

            return []

        others = [
            memory
            for memory in self.store.all()
            if memory.id != memory_id
        ]

        print(
            f"\n[MEMORA] Comparing '{memory_id}' "
            f"against {len(others)} existing "
            f"memories...\n"
        )

        relationships = []

        for other in others:

            result = self.analyze_relationship(
                target,
                other
            )

            if result["strength"]:

                print(
                    f"{target.source} <-> "
                    f"{other.source}: "
                    f"{result['strength'].upper()} "
                    f"{(result.get('relationship_type'))}"
                    f"({result['score']:.4f})"
                )

                self._save_relationship(
                    result
                )

                relationships.append(
                    result
                )

        self.store._save()

        if not relationships:

            print(
                "[MEMORA] No relationships "
                "discovered for this memory."
            )

        return relationships


if __name__ == "__main__":

    engine = RelationshipEngine()

    engine.discover_relationships()
