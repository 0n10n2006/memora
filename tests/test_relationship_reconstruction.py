import unittest
import tempfile
from pathlib import Path

from backend.reconstruction.reconstructor import MemoryReconstructor
from backend.reconstruction.memory_pipeline import MemoryPipeline
from backend.memory.memory_object import MemoryObject, MemoryRelationship
from backend.memory.memory_store import MemoryStore
from backend.memory.relationship_engine import RelationshipEngine
from backend.retrieval.search import MemorySearch


def memory(memory_id, title, summary, retrieval_type, relationship=None):
    result = {
        "document": "probability and defective bolts",
        "retrieval_type": retrieval_type,
        "score": 0.7,
        "metadata": {
            "memory_id": memory_id,
            "title": title,
            "summary": summary,
            "source": title,
            "topics": ["probability", "statistics"],
            "entities": ["probability"],
        },
    }
    if relationship:
        result["relationship"] = relationship
    return result


class RelationshipReconstructionTests(unittest.TestCase):

    def setUp(self):
        # These deterministic tests intentionally avoid loading Qwen.
        self.reconstructor = MemoryReconstructor.__new__(MemoryReconstructor)
        self.reconstructor._build_primary_facts = lambda value: {
            "concepts": ["probability", "defective bolts"],
            "normalized_content": "probability defective bolts repair time",
        }

        self.relationship = {
            "relationship_type": "same_topic",
            "strength": "strong",
            "score": 0.4456,
            "shared_topics": ["probability", "random variable", "statistics"],
            "shared_entities": ["probability"],
        }

    def test_reconstructs_relationship_from_metadata_without_model(self):
        memories = [
            memory("assignment", "assignment.jpg", "Assignment about probability and defective bolts.", "primary"),
            memory("notes", "notes.pdf", "Study material covering probability and statistics.", "related", self.relationship),
        ]

        result = self.reconstructor.reconstruct(
            "How are the probability assignment and my probability notes related?",
            memories,
        )

        self.assertIn("The assignment and the study notes are strongly related", result["answer"])
        self.assertIn("probability, random variable, and statistics", result["answer"])
        self.assertIn("defective-bolt and repair-time questions", result["answer"])
        self.assertEqual(result["confidence"], 0.85)
        self.assertEqual(result["evidence"], ["assignment.jpg", "notes.pdf"])

    def test_missing_relationship_metadata_uses_existing_general_path(self):
        self.assertIsNone(
            self.reconstructor._build_relationship_answer(
                "How are these related?",
                [memory("assignment", "assignment.jpg", "Assignment about probability.", "primary")],
            )
        )

    def test_pipeline_delegates_relationship_queries_to_reconstructor(self):
        class SearchStub:
            def __init__(self, results):
                self.results = results

            def search_with_relationships(self, *args, **kwargs):
                return self.results

        pipeline = MemoryPipeline.__new__(MemoryPipeline)
        pipeline.search = SearchStub([
            memory("assignment", "assignment.jpg", "Assignment about probability and defective bolts.", "primary"),
            memory("notes", "notes.pdf", "Study material covering probability and statistics.", "related", self.relationship),
        ])
        pipeline.reconstructor = self.reconstructor

        result = pipeline.remember(
            "How are the probability assignment and my probability notes related?"
        )

        self.assertIn("strongly related", result["reconstruction"]["answer"])
        self.assertEqual(result["reconstruction"]["confidence"], 0.85)

    def test_pipeline_keeps_the_irrelevant_query_rejection(self):
        class SearchStub:
            def search_with_relationships(self, *args, **kwargs):
                return [
                    memory("assignment", "assignment.jpg", "Assignment about probability and defective bolts.", "primary")
                ]

        pipeline = MemoryPipeline.__new__(MemoryPipeline)
        pipeline.search = SearchStub()

        result = pipeline.remember("What do I remember about quantum computing?")

        self.assertEqual(
            result["reconstruction"],
            {
                "answer": "I don't have a stored memory that matches that.",
                "confidence": 0.05,
                "evidence": [],
            },
        )

    def test_multi_memory_synthesis_is_grounded_in_summaries_and_topics(self):
        memories = [
            memory("assignment", "assignment.jpg", "Assignment about probability and defective bolts.", "primary"),
            memory("notes", "notes.pdf", "Study material covering probability and statistics.", "related", self.relationship),
        ]

        result = self.reconstructor.reconstruct(
            "What do I remember about probability across my memories?",
            memories,
        )

        self.assertIn("common thread is probability, statistics", result["answer"])
        self.assertIn("Assignment about probability and defective bolts.", result["answer"])
        self.assertIn("Study material covering probability and statistics.", result["answer"])

    def test_contextual_evidence_keeps_dates_and_conflicts_separate_from_score(self):
        engine = RelationshipEngine.__new__(RelationshipEngine)
        first = MemoryObject(
            id="first", source="first", modality="document",
            metadata={"event_date": "2026-01-04", "facts": {"status": "open", "owner": "A"}},
        )
        second = MemoryObject(
            id="second", source="second", modality="document",
            metadata={"event_date": "2026-01-05", "facts": {"status": "closed", "owner": "A"}},
        )

        temporal, contradictions = engine._contextual_evidence(first, second)

        self.assertEqual(
            temporal,
            "the memories refer to 2026-01-04 and 2026-01-05",
        )
        self.assertEqual(contradictions, ["status: 'open' vs 'closed'"])

    def test_relationship_answer_reports_conflicts_without_resolving_them(self):
        conflicting = dict(self.relationship)
        conflicting["contradictions"] = ["status: 'open' vs 'closed'"]
        conflicting["temporal_relation"] = "both memories refer to 2026-01-04"
        memories = [
            memory("assignment", "assignment.jpg", "Assignment about probability and defective bolts.", "primary"),
            memory("notes", "notes.pdf", "Study material covering probability and statistics.", "related", conflicting),
        ]

        result = self.reconstructor.reconstruct("How are these related?", memories)

        self.assertIn("Temporal context: both memories refer to 2026-01-04.", result["answer"])
        self.assertIn("can't determine which value is correct", result["answer"])

    def test_search_keeps_relationship_metadata_when_both_memories_are_primary(self):
        assignment = MemoryObject(
            id="assignment", source="assignment.jpg", modality="image",
            relationships=[MemoryRelationship(
                target_id="notes", relationship_type="same_topic", score=0.45,
                strength="strong", shared_topics=["probability"],
            )],
        )
        notes = MemoryObject(id="notes", source="notes.pdf", modality="document")

        class StoreStub:
            def get(self, memory_id):
                return {"assignment": assignment, "notes": notes}.get(memory_id)

        search = MemorySearch.__new__(MemorySearch)
        search.memory_store = StoreStub()
        results = search.expand_relationships([
            {"document": "assignment", "metadata": {"memory_id": "assignment"}},
            {"document": "notes", "metadata": {"memory_id": "notes"}},
        ])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["relationship"]["target_id"], "notes")
        self.assertEqual(results[0]["relationship"]["strength"], "strong")

    def test_primary_to_primary_relationship_uses_relationship_confidence(self):
        memories = [
            memory("assignment", "assignment.jpg", "Assignment about probability and defective bolts.", "primary", self.relationship),
            memory("notes", "notes.pdf", "Study material covering probability and statistics.", "primary"),
        ]

        result = self.reconstructor.reconstruct("How are these related?", memories)

        self.assertIn("The assignment and the study notes", result["answer"])
        self.assertEqual(result["confidence"], 0.85)

    def test_contextual_relationship_fields_survive_store_round_trip(self):
        with tempfile.TemporaryDirectory(dir="tests") as temporary_directory:
            storage_path = Path(temporary_directory) / "memories.json"
            store = MemoryStore(storage_path=str(storage_path))
            store.add(MemoryObject(id="first", source="first", modality="document"))
            store.add(MemoryObject(id="second", source="second", modality="document"))
            store.add_relationship(
                "first", "second", relationship_type="same_topic", score=0.75,
                strength="strong", temporal_relation="both memories refer to 2026-01-04",
                contradictions=["status: 'open' vs 'closed'"],
            )

            reloaded = MemoryStore(storage_path=str(storage_path))
            relationship = reloaded.get_relationships("first")[0]

            self.assertEqual(
                relationship.temporal_relation,
                "both memories refer to 2026-01-04",
            )
            self.assertEqual(relationship.contradictions, ["status: 'open' vs 'closed'"])

    def test_multi_memory_synthesis_handles_a_larger_result_set(self):
        memories = []

        for index in range(30):
            item = memory(
                f"memory-{index}",
                f"memory-{index}.txt",
                f"Probability memory {index}.",
                "primary",
            )
            item["metadata"]["topics"] = ["probability", f"topic {index}"]
            memories.append(item)

        answer = self.reconstructor._build_multi_memory_synthesis(memories)

        self.assertIn("common thread is probability", answer)
        self.assertIn("Other covered concepts: topic 0", answer)

    def test_multi_memory_synthesis_handles_one_thousand_memories(self):
        memories = []

        for index in range(1000):
            item = memory(
                f"memory-{index}",
                f"memory-{index}.txt",
                f"Probability memory {index}.",
                "primary",
            )
            item["metadata"]["topics"] = ["probability", "statistics", f"topic {index}"]
            memories.append(item)

        answer = self.reconstructor._build_multi_memory_synthesis(memories)

        self.assertIn("common thread is probability, statistics", answer)
        self.assertIn("Probability memory 0.", answer)


if __name__ == "__main__":
    unittest.main()
