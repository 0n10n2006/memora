from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


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

    # -----------------------------------------
    # Chunks belonging to this memory
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

    # -----------------------------------------
    # Temporal information
    # -----------------------------------------

    created_at: Optional[str] = None

    modified_at: Optional[str] = None