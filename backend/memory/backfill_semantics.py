from backend.memory.memory_store import MemoryStore
from backend.memory.semantic_analyzer import SemanticAnalyzer
from backend.memory.keyword_extractor import KeywordExtractor
from backend.memory.memory_understanding import MemoryUnderstanding


def clean_list(items):

    if not items:
        return []

    cleaned = []

    bad_fragments = [

        "explanation",
        "not applicable",
        "no entities",
        "no topics",
        "therefore",
        "the memory focuses",
        "no additional",
        "should be ignored",
        "statistical distributions",
        "mathematical operations",
        "external references",
    ]

    for item in items:

        if not isinstance(
            item,
            str
        ):
            continue

        item = item.strip()

        if not item:
            continue

        item = item.lstrip(
            "•-* "
        ).strip()

        if not item:
            continue

        lowered = item.lower()

        if any(
            fragment in lowered
            for fragment in bad_fragments
        ):
            continue

        # Topics/entities should be compact.
        if len(
            item.split()
        ) > 6:
            continue

        item = item.rstrip(
            ".,;:"
        )

        if not item:
            continue

        if not any(
            existing.lower()
            == item.lower()
            for existing in cleaned
        ):

            cleaned.append(
                item.lower()
            )

    return cleaned


def clean_summary(
    summary
):

    if not isinstance(
        summary,
        str
    ):
        return ""

    summary = summary.strip()

    if not summary:
        return ""

    lowered = summary.lower()

    bad_patterns = [

        "the summary does not contain",

        "should be ignored",

        "please provide additional context",

        "random data",

        "no relevant information",

        "not applicable",

        "the memory focuses solely",

        "no additional sections",

        "the question asks",

        "to analyze the given memory",

        "i will identify",

        "as an ai",
    ]

    for pattern in bad_patterns:

        if pattern in lowered:

            return ""

    if len(
        summary
    ) > 500:

        return ""

    return summary


def merge_unique(
    *groups
):

    result = []

    for group in groups:

        if not group:
            continue

        for item in group:

            if not isinstance(
                item,
                str
            ):
                continue

            item = item.strip()

            if not item:
                continue

            if not any(
                existing.lower()
                == item.lower()
                for existing in result
            ):

                result.append(
                    item
                )

    return result


def main():

    store = MemoryStore()

    semantic_analyzer = (
        SemanticAnalyzer()
    )

    keyword_extractor = (
        KeywordExtractor()
    )

    understanding = (
        MemoryUnderstanding()
    )

    memories = store.all()

    print(
        f"\n[MEMORA] Updating "
        f"{len(memories)} memories...\n"
    )

    for memory in memories:

        print(
            "\n----------------------------------------"
        )

        print(
            f"Processing: {memory.source}"
        )

        # =================================================
        # BUILD FULL SEMANTIC INPUT
        # =================================================

        text_parts = []

        if memory.title:

            text_parts.append(
                f"Title: {memory.title}"
            )

        if memory.summary:

            text_parts.append(
                f"Summary: {memory.summary}"
            )

        if memory.description:

            text_parts.append(
                f"Description: {memory.description}"
            )

        if memory.entities:

            text_parts.append(
                "Entities: "
                + ", ".join(
                    memory.entities
                )
            )

        if memory.topics:

            text_parts.append(
                "Previously detected topics: "
                + ", ".join(
                    memory.topics
                )
            )

        if memory.content:

            text_parts.append(
                "Content:\n"
                + memory.content
            )

        raw_text = "\n\n".join(
            text_parts
        )

        # =================================================
        # NORMALIZE OCR
        # =================================================

        text = (
            understanding.normalize_text(
                raw_text
            )
        )

        # =================================================
        # KEYWORD SIGNALS
        # =================================================

        try:

            keyword_topics = (
                keyword_extractor.extract(
                    text
                )
            )

        except Exception as error:

            print(
                "[MEMORA] Keyword extraction "
                f"failed: {error}"
            )

            keyword_topics = []

        keyword_topics = clean_list(
            keyword_topics
        )

        print(
            "\n[MEMORA] Keyword signals:"
        )

        for topic in keyword_topics:

            print(
                f" • {topic}"
            )

        # =================================================
        # SEMANTIC ANALYSIS
        # =================================================

        try:

            result = (
                semantic_analyzer.analyze(
                    text,
                    known_topics=keyword_topics
                )
            )

        except Exception as error:

            print(
                "\n[MEMORA] Semantic analysis "
                f"failed: {error}"
            )

            result = {}

        # =================================================
        # SUMMARY
        # =================================================

        new_summary = clean_summary(
            result.get(
                "summary",
                ""
            )
        )

        if new_summary:

            memory.summary = (
                new_summary
            )

        else:

            print(
                "[MEMORA] Model summary rejected; "
                "keeping existing summary."
            )

        # =================================================
        # TOPICS
        # =================================================

        semantic_topics = clean_list(
            result.get(
                "topics",
                []
            )
        )

        # Existing topics are retained as well.
        existing_topics = clean_list(
            memory.topics
        )

        combined_topics = (
            merge_unique(
                semantic_topics,
                keyword_topics,
                existing_topics
            )
        )

        if combined_topics:

            memory.topics = (
                combined_topics[:20]
            )

        # =================================================
        # ENTITIES
        # =================================================

        semantic_entities = clean_list(
            result.get(
                "entities",
                []
            )
        )

        existing_entities = clean_list(
            memory.entities
        )

        combined_entities = (
            merge_unique(
                semantic_entities,
                existing_entities
            )
        )

        if combined_entities:

            memory.entities = (
                combined_entities[:20]
            )

        # =================================================
        # SAVE
        # =================================================

        store.update(
            memory
        )

        print(
            "\n[MEMORA] Updated:"
        )

        print(
            "Summary:",
            memory.summary
        )

        print(
            "Topics:",
            memory.topics
        )

        print(
            "Entities:",
            memory.entities
        )

    print(
        "\n========================================"
    )

    print(
        "[MEMORA] Semantic backfill complete."
    )

    print(
        "========================================\n"
    )


if __name__ == "__main__":

    main()