import json
from pathlib import Path
from dataclasses import asdict

from backend.memory.memory_object import (
    MemoryObject,
    MemoryRelationship
)


class MemoryStore:

    def __init__(
        self,
        storage_path="data/memories.json"
    ):

        self.storage_path = Path(
            storage_path
        )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.memories = {}

        self._load()

    # =========================================
    # LOAD
    # =========================================

    def _load(self):

        if not self.storage_path.exists():
            return

        try:

            data = json.loads(
                self.storage_path.read_text(
                    encoding="utf-8"
                )
            )

            for item in data:

                # ---------------------------------
                # Topics were added later.
                # ---------------------------------

                item.setdefault(
                    "topics",
                    []
                )

                # ---------------------------------
                # Relationships were also added
                # later. Clean old formats here.
                # ---------------------------------

                clean_relationships = []

                for relationship in item.get(
                    "relationships",
                    []
                ):

                    # Old relationship format:
                    #
                    # {
                    #   "target_id": "...",
                    #   "relationship_type": "semantic",
                    #   "confidence": 0.0,
                    #   "evidence": []
                    # }
                    #
                    # Convert it into the new format.

                    target_id = relationship.get(
                        "target_id"
                    )

                    if not target_id:
                        continue

                    clean_relationships.append(
                        MemoryRelationship(
                            target_id=target_id,
                            relationship_type=(
                                relationship.get(
                                    "relationship_type",
                                    "semantic"
                                )
                            ),
                            score=float(
                                relationship.get(
                                    "score",
                                    relationship.get(
                                        "confidence",
                                        0.0
                                    )
                                )
                            ),
                            strength=relationship.get(
                                "strength",
                                "weak"
                            ),
                            shared_topics=(
                                relationship.get(
                                    "shared_topics",
                                    []
                                )
                            ),
                            shared_entities=(
                                relationship.get(
                                    "shared_entities",
                                    []
                                )
                            ),
                            evidence=(
                                relationship.get(
                                    "evidence",
                                    relationship.get(
                                        "reasons",
                                        []
                                    )
                                )
                            ),
                            temporal_relation=relationship.get(
                                "temporal_relation",
                                ""
                            ),
                            contradictions=relationship.get(
                                "contradictions",
                                []
                            )
                        )
                    )

                item[
                    "relationships"
                ] = clean_relationships

                # ---------------------------------
                # Ensure related IDs exist.
                # ---------------------------------

                item.setdefault(
                    "related_memory_ids",
                    []
                )

                memory = MemoryObject(
                    **item
                )

                self.memories[
                    memory.id
                ] = memory

        except Exception as error:

            print(
                f"[MEMORA] Could not load "
                f"memory store: {error}"
            )

    # =========================================
    # SAVE
    # =========================================

    def _save(self):

        data = [
            asdict(memory)
            for memory
            in self.memories.values()
        ]

        self.storage_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    # =========================================
    # ADD
    # =========================================

    def add(
        self,
        memory
    ):

        self.memories[
            memory.id
        ] = memory

        self._save()

    # =========================================
    # UPDATE
    # =========================================

    def update(
        self,
        memory
    ):

        self.memories[
            memory.id
        ] = memory

        self._save()

    # =========================================
    # GET
    # =========================================

    def get(
        self,
        memory_id
    ):

        return self.memories.get(
            memory_id
        )

    # =========================================
    # ALL
    # =========================================

    def all(self):

        return list(
            self.memories.values()
        )

    # =========================================
    # COUNT
    # =========================================

    def count(self):

        return len(
            self.memories
        )

    # =========================================
    # ADD RELATIONSHIP
    # =========================================

    def add_relationship(
        self,
        memory_a,
        memory_b,
        relationship_type="semantic",
        score=0.0,
        strength="weak",
        shared_topics=None,
        shared_entities=None,
        evidence=None,
        temporal_relation="",
        contradictions=None
    ):

        if shared_topics is None:
            shared_topics = []

        if shared_entities is None:
            shared_entities = []

        if evidence is None:
            evidence = []

        if contradictions is None:
            contradictions = []

        # -----------------------------------------
        # Verify memories exist
        # -----------------------------------------

        if memory_a not in self.memories:
            return

        if memory_b not in self.memories:
            return

        first = self.memories[
            memory_a
        ]

        second = self.memories[
            memory_b
        ]

        # =========================================
        # A → B
        # =========================================

        self._add_one_way_relationship(
            source=first,
            target_id=memory_b,
            relationship_type=relationship_type,
            score=score,
            strength=strength,
            shared_topics=shared_topics,
            shared_entities=shared_entities,
            evidence=evidence,
            temporal_relation=temporal_relation,
            contradictions=contradictions
        )

        # =========================================
        # B → A
        # =========================================

        self._add_one_way_relationship(
            source=second,
            target_id=memory_a,
            relationship_type=relationship_type,
            score=score,
            strength=strength,
            shared_topics=shared_topics,
            shared_entities=shared_entities,
            evidence=evidence,
            temporal_relation=temporal_relation,
            contradictions=contradictions
        )

        # =========================================
        # Keep ID index synchronized
        # =========================================

        if memory_b not in first.related_memory_ids:

            first.related_memory_ids.append(
                memory_b
            )

        if memory_a not in second.related_memory_ids:

            second.related_memory_ids.append(
                memory_a
            )

        self._save()

    # =========================================
    # INTERNAL RELATIONSHIP HELPER
    # =========================================

    def _add_one_way_relationship(
        self,
        source,
        target_id,
        relationship_type,
        score,
        strength,
        shared_topics,
        shared_entities,
        evidence,
        temporal_relation,
        contradictions
    ):

        existing = None

        for relationship in source.relationships:

            if (
                relationship.target_id
                == target_id
                and
                relationship.relationship_type
                == relationship_type
            ):

                existing = relationship
                break

        # -----------------------------------------
        # Update existing relationship
        # -----------------------------------------

        if existing:

            existing.score = max(
                existing.score,
                score
            )

            # Prefer the strongest classification.

            strength_rank = {
                "weak": 1,
                "moderate": 2,
                "strong": 3
            }

            old_rank = strength_rank.get(
                existing.strength,
                0
            )

            new_rank = strength_rank.get(
                strength,
                0
            )

            if new_rank > old_rank:

                existing.strength = strength

            for topic in shared_topics:

                if topic not in existing.shared_topics:

                    existing.shared_topics.append(
                        topic
                    )

            for entity in shared_entities:

                if (
                    entity
                    not in existing.shared_entities
                ):

                    existing.shared_entities.append(
                        entity
                    )

            for item in evidence:

                if item not in existing.evidence:

                    existing.evidence.append(
                        item
                    )

            if temporal_relation and not existing.temporal_relation:
                existing.temporal_relation = temporal_relation

            for contradiction in contradictions:
                if contradiction not in existing.contradictions:
                    existing.contradictions.append(contradiction)

        # -----------------------------------------
        # Create new relationship
        # -----------------------------------------

        else:

            source.relationships.append(

                MemoryRelationship(

                    target_id=target_id,

                    relationship_type=(
                        relationship_type
                    ),

                    score=score,

                    strength=strength,

                    shared_topics=list(
                        shared_topics
                    ),

                    shared_entities=list(
                        shared_entities
                    ),

                    evidence=list(
                        evidence
                    ),
                    temporal_relation=temporal_relation,
                    contradictions=list(contradictions)
                )
            )

    # =========================================
    # RELATED MEMORIES
    # =========================================

    def related_to(
        self,
        memory_id
    ):

        memory = self.get(
            memory_id
        )

        if not memory:
            return []

        results = []

        for relationship in memory.relationships:

            target = self.get(
                relationship.target_id
            )

            if target:

                results.append(
                    target
                )

        return results

    # =========================================
    # GET RELATIONSHIPS
    # =========================================

    def get_relationships(
        self,
        memory_id
    ):

        memory = self.get(
            memory_id
        )

        if not memory:
            return []

        return memory.relationships

    # =========================================
    # CLEAR RELATIONSHIPS
    # =========================================

    def clear_relationships(self):

        for memory in self.memories.values():

            memory.relationships = []

            memory.related_memory_ids = []

        self._save()

    # =========================================
    # REMOVE ONE RELATIONSHIP
    # =========================================

    def remove_relationship(
        self,
        memory_a,
        memory_b
    ):

        first = self.get(
            memory_a
        )

        second = self.get(
            memory_b
        )

        if not first or not second:
            return

        first.relationships = [
            relationship
            for relationship
            in first.relationships
            if relationship.target_id
            != memory_b
        ]

        second.relationships = [
            relationship
            for relationship
            in second.relationships
            if relationship.target_id
            != memory_a
        ]

        first.related_memory_ids = [
            memory_id
            for memory_id
            in first.related_memory_ids
            if memory_id
            != memory_b
        ]

        second.related_memory_ids = [
            memory_id
            for memory_id
            in second.related_memory_ids
            if memory_id
            != memory_a
        ]

        self._save()
