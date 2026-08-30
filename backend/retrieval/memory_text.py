def build_searchable_text(memory):
    """
    Build the searchable representation of a MemoryObject.

    Combines:
    - title
    - AI-generated summary
    - visual description
    - semantic topics
    - entities
    - original OCR / extracted content
    """

    parts = []

    if memory.title:
        parts.append(
            f"Title: {memory.title}"
        )

    if memory.summary:
        parts.append(
            f"Summary: {memory.summary}"
        )

    if memory.description:
        parts.append(
            f"Description: {memory.description}"
        )

    if memory.topics:
        parts.append(
            "Topics: "
            + ", ".join(memory.topics)
        )

    if memory.entities:
        parts.append(
            "Entities: "
            + ", ".join(memory.entities)
        )

    if memory.content:
        parts.append(
            f"Content:\n{memory.content}"
        )

    return "\n\n".join(parts)