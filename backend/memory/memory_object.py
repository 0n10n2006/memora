from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MemoryRelationship:

    # ID of the connected memory
    target_id: str

    # Relationship category
    relationship_type: str = "semantic"

    # Overall relationship score
    score: float = 0.0

    # strong / moderate / weak
    strength: str = "weak"

    # Topics/entities that caused the relationship
    shared_topics: List[str] = field(
        default_factory=list
    )

    shared_entities: List[str] = field(
        default_factory=list
    )

    # Human-readable explanation
    evidence: List[str] = field(
        default_factory=list
    )

    # Context explains a relationship but does not inflate its score.
    temporal_relation: str = ""

    # Conflicting structured facts are retained for reconstruction instead
    # of silently choosing one memory's value.
    contradictions: List[str] = field(
        default_factory=list
    )


@dataclass
class MemoryObject:

    # -----------------------------------------
    # Identity
    # -----------------------------------------

    id: str

    source: str

    modality: str

    # -----------------------------------------
    # Human-readable information
    # -----------------------------------------

    title: str = ""

    summary: str = ""

    # -----------------------------------------
    # Extracted information
    # -----------------------------------------

    content: str = ""

    description: str = ""

    entities: List[str] = field(
        default_factory=list
    )

    topics: List[str] = field(
        default_factory=list
    )

    # -----------------------------------------
    # Chunks
    # -----------------------------------------

    chunk_ids: List[str] = field(
        default_factory=list
    )

    # -----------------------------------------
    # File metadata
    # -----------------------------------------

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # -----------------------------------------
    # Relationships
    # -----------------------------------------

    related_memory_ids: List[str] = field(
        default_factory=list
    )

    relationships: List[MemoryRelationship] = field(
        default_factory=list
    )

    # -----------------------------------------
    # Temporal information
    # -----------------------------------------

    created_at: Optional[str] = None

    modified_at: Optional[str] = None
