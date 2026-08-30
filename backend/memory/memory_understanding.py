from backend.memory.memory_object import MemoryObject


class MemoryUnderstanding:

    """
    Converts raw ingestion results into a
    normalized representation suitable for
    retrieval, relationships and reconstruction.
    """

    def build_memory_text(
        self,
        memory: MemoryObject
    ):

        parts = []

        # -----------------------------------------
        # Title
        # -----------------------------------------

        if memory.title:

            parts.append(
                f"Title: {memory.title}"
            )

        # -----------------------------------------
        # Summary
        # -----------------------------------------

        if memory.summary:

            parts.append(
                f"Summary: {memory.summary}"
            )

        # -----------------------------------------
        # Description
        # -----------------------------------------

        if memory.description:

            parts.append(
                f"Description: {memory.description}"
            )

        # -----------------------------------------
        # Entities
        # -----------------------------------------

        if memory.entities:

            parts.append(
                "Entities: "
                + ", ".join(
                    memory.entities
                )
            )

        # -----------------------------------------
        # Content
        # -----------------------------------------

        if memory.content:

            parts.append(
                f"Content:\n{memory.content}"
            )

        return "\n\n".join(
            parts
        )

    # -----------------------------------------
    # Normalize noisy OCR
    # -----------------------------------------

    def normalize_text(
        self,
        text
    ):

        if not text:

            return ""

        replacements = {

            "probabiliby":
                "probability",

            "probabiliy":
                "probability",

            "defechive":
                "defective",

            "defeckive":
                "defective",

            "parobei":
                "probability",

            "paobabil":
                "probability",

            "distribubion":
                "distribution",

            "distribubions":
                "distributions",

            "condihone":
                "condition",

            "condihoneal":
                "conditional",

            "condibional":
                "conditional",

            "srequived":
                "required",

            "Jokes":
                "bolts",
        }

        normalized = text

        for (
            incorrect,
            correct
        ) in replacements.items():

            normalized = normalized.replace(
                incorrect,
                correct
            )

        return normalized

    # -----------------------------------------
    # Build normalized text
    # -----------------------------------------

    def build_normalized_text(
        self,
        memory
    ):

        raw_text = self.build_memory_text(
            memory
        )

        normalized = self.normalize_text(
            raw_text
        )

        return normalized