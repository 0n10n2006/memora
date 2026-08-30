from backend.memory.memory_store import MemoryStore
from backend.memory.memory_understanding import (
    MemoryUnderstanding
)


store = MemoryStore()

understanding = MemoryUnderstanding()

for memory in store.all():

    print(
        "\n========================================"
    )

    print(
        "MEMORY:",
        memory.title
    )

    print(
        "\nNORMALIZED REPRESENTATION:\n"
    )

    print(
        understanding.build_normalized_text(
            memory
        )[:5000]
    )