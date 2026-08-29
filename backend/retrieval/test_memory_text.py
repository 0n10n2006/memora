from backend.memory.memory_item import MemoryItem
from backend.retrieval.memory_text import build_searchable_text


memory = MemoryItem(
    id="probability_assignment",
    source="PHOTO-2026-03-30-09-46-15.jpg",
    modality="image",

    content="""
    Assignment about probability.
    Determine the probability that one bolt is defective.
    Find conditional probability.
    """,

    description="""
    Handwritten probability and statistics assignment
    involving defective bolts and probability distributions.
    """,

    entities=[
        "probability",
        "statistics",
        "defective bolts",
        "conditional probability"
    ]
)


searchable_text = build_searchable_text(memory)


print(
    "\n========== SEARCHABLE MEMORY ==========\n"
)

print(searchable_text)

print(
    "\n=======================================\n"
)