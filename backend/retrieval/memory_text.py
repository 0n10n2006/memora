def build_searchable_text(memory):
    """
    Build a unified semantic representation of a MemoryItem.

    Combines:
    - OCR / extracted text
    - AI description
    - entities
    - useful metadata
    """

    sections = []

    if memory.content:
        sections.append(
            f"Extracted text:\n{memory.content}"
        )

    if memory.description:
        sections.append(
            f"Visual description:\n{memory.description}"
        )

    if memory.entities:
        sections.append(
            "Entities:\n"
            + ", ".join(memory.entities)
        )

    if memory.modality:
        sections.append(
            f"Content type: {memory.modality}"
        )

    return "\n\n".join(sections)