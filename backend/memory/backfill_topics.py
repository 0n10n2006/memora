from backend.memory.memory_store import MemoryStore
from backend.memory.topic_extractor import TopicExtractor


def main():

    store = MemoryStore()

    extractor = TopicExtractor()

    memories = store.all()

    print(
        f"\n[MEMORA] Backfilling topics for "
        f"{len(memories)} memories...\n"
    )

    for memory in memories:

        raw_text = "\n".join([
            memory.title,
            memory.summary,
            memory.content,
            memory.description,
            " ".join(memory.entities)
        ])

        topics = extractor.extract(
            raw_text
        )

        memory.topics = topics

        print(
            f"{memory.source}"
        )

        print(
            f"  Topics: {topics}\n"
        )

    # Save everything back
    store._save()

    print(
        "[MEMORA] Topic backfill complete."
    )


if __name__ == "__main__":

    main()