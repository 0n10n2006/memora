from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class MemoryItem:

    # Unique identifier
    id: str

    # Original file
    source: str

    # document / image / spreadsheet / text
    modality: str

    # Raw extracted text
    content: str = ""

    # AI-generated visual/content description
    description: str = ""

    # Important concepts/entities
    entities: List[str] = field(
        default_factory=list
    )

    # Metadata about the original file
    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    # OCR confidence, if available
    ocr_confidence: Optional[float] = None

    # Semantic embedding
    embedding: Optional[List[float]] = None