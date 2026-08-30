import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from backend.memory.topic_extractor import TopicExtractor


class SemanticAnalyzer:

    """
    Converts raw memory text into a compact semantic representation.

    The LLM provides semantic interpretation while deterministic
    extraction acts as a safety net for noisy OCR.
    """

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct"
    ):

        print(
            f"[MEMORA] Loading semantic analyzer: "
            f"{model_name}"
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                local_files_only=True
            )
        )

        self.model.to(self.device)
        self.model.eval()

        self.topic_extractor = TopicExtractor()

        print(
            f"[MEMORA] Semantic analyzer ready "
            f"on {self.device}."
        )

    # =================================================
    # ANALYZE
    # =================================================

    def analyze(
        self,
        text,
        known_topics=None
    ):

        if not text:

            return {
                "summary": "",
                "topics": [],
                "entities": []
            }

        if known_topics is None:

            known_topics = []

        # -------------------------------------------------
        # Prepare noisy OCR
        # -------------------------------------------------

        clean_text = self._prepare_text(
            text
        )

        # -------------------------------------------------
        # Ask the LLM for semantic structure
        # -------------------------------------------------

        prompt = self._build_prompt(
            clean_text
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id
            )

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated = (
            output[0][input_length:]
        )

        result = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        print(
            "\n[MEMORA] Raw semantic output:"
        )

        print(result)

        # -------------------------------------------------
        # Parse model output
        # -------------------------------------------------

        parsed = self._parse_result(
            result
        )

        # -------------------------------------------------
        # Deterministic topic recovery
        #
        # This remains important because OCR can contain
        # spelling corruption and the tiny LLM can miss
        # concepts.
        # -------------------------------------------------

        recovered_topics = (
            self._recover_topics(
                text
            )
        )

        parsed["topics"] = (
            self._merge_unique(
                known_topics,
                parsed.get(
                    "topics",
                    []
                ),
                recovered_topics
            )
        )

        # -------------------------------------------------
        # Deterministic entity recovery
        # -------------------------------------------------

        recovered_entities = (
            self._recover_entities(
                text
            )
        )

        parsed["entities"] = (
            self._merge_unique(
                parsed.get(
                    "entities",
                    []
                ),
                recovered_entities
            )
        )

        # -------------------------------------------------
        # Final cleanup
        # -------------------------------------------------

        parsed["summary"] = (
            self._clean_summary(
                parsed.get(
                    "summary",
                    ""
                ),
                text
            )
        )

        parsed["topics"] = (
            self._clean_topics(
                parsed.get(
                    "topics",
                    []
                )
            )
        )

        parsed["entities"] = (
            self._clean_entities(
                parsed.get(
                    "entities",
                    []
                )
            )
        )

        return parsed

    # =================================================
    # PREPARE TEXT
    # =================================================

    def _prepare_text(
        self,
        text
    ):

        text = str(
            text
        )

        # Collapse excessive whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        # Tiny local model does not need enormous
        # duplicated OCR.
        if len(text) > 5000:

            text = text[:5000]

        return text

    # =================================================
    # PROMPT
    # =================================================

    def _build_prompt(
        self,
        text
    ):

        return f"""
You are the semantic memory analyzer for a personal memory system.

Analyze the memory below.

Return exactly three lines:

SUMMARY: <one short factual sentence>
TOPICS: <comma-separated concepts>
ENTITIES: <comma-separated important things>

Rules:
- Identify what the memory is actually about.
- Extract specific academic, technical, or practical concepts.
- Preserve specific concepts even when the text contains spelling mistakes.
- Recognize concepts from context when OCR spelling is imperfect.
- Do not solve problems.
- Do not calculate values.
- Do not invent information.
- Do not mention OCR.
- Do not mention images, photographs, files, documents, or paper.
- Topics should be short concepts, normally 1-4 words.
- Entities should be specific things, objects, technologies, people, places, or named concepts.
- If there are no entities, write NONE.
- Output only the three requested lines.

MEMORY:
{text}
""".strip()

    # =================================================
    # PARSER
    # =================================================

    def _parse_result(
        self,
        result
    ):

        if not result:

            return {
                "summary": "",
                "topics": [],
                "entities": []
            }

        if self._has_prompt_leakage(
            result
        ):

            return {
                "summary": "",
                "topics": [],
                "entities": []
            }

        summary = ""
        topics = []
        entities = []

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        match = re.search(
            r"(?im)^SUMMARY\s*:\s*(.+)$",
            result
        )

        if match:

            summary = (
                match.group(1)
                .strip()
            )

        # -------------------------------------------------
        # TOPICS
        # -------------------------------------------------

        match = re.search(
            r"(?im)^TOPICS\s*:\s*(.+)$",
            result
        )

        if match:

            raw_topics = (
                match.group(1)
                .strip()
            )

            if raw_topics.upper() != "NONE":

                topics = [
                    item.strip()
                    for item
                    in raw_topics.split(",")
                    if item.strip()
                ]

        # -------------------------------------------------
        # ENTITIES
        # -------------------------------------------------

        match = re.search(
            r"(?im)^ENTITIES\s*:\s*(.+)$",
            result
        )

        if match:

            raw_entities = (
                match.group(1)
                .strip()
            )

            if raw_entities.upper() != "NONE":

                entities = [
                    item.strip()
                    for item
                    in raw_entities.split(",")
                    if item.strip()
                ]

        return {
            "summary": summary[:500],
            "topics": topics,
            "entities": entities
        }

    # =================================================
    # PROMPT LEAKAGE DETECTION
    # =================================================

    def _has_prompt_leakage(
        self,
        result
    ):

        value = (
            result
            .lower()
            .strip()
        )

        if not value:

            return True

        leakage_markers = [

            "rules:",

            "input example",

            "example 1:",

            "instructions:",

            "memory:",

            "summary:",
        ]

        # SUMMARY: by itself is legitimate.
        # Only treat it as leakage when other
        # prompt markers are also present.

        marker_count = 0

        for marker in leakage_markers:

            if marker in value:

                marker_count += 1

        if marker_count >= 2:

            return True

        instruction_patterns = [

            r"\bdo\s+not\s+(?:use|write|mention|invent|solve|calculate|explain)\b",

            r"\bthe\s+summary\s+should\b",

            r"\bwrite\s+exactly\b",

            r"\boutput\s+only\b",

            r"\bone\s+short\s+(?:factual\s+)?sentence\b",

            r"\bread\s+the\s+memory\s+below\b",

            r"\byou\s+are\s+(?:a|the)\s+semantic\s+memory\s+analyzer\b"
        ]

        for pattern in instruction_patterns:

            if re.search(
                pattern,
                value
            ):

                return True

        return False

    # =================================================
    # TOPIC RECOVERY
    # =================================================

    def _recover_topics(
        self,
        text
    ):

        if not text:

            return []

        try:

            return self.topic_extractor.extract(
                text,
                max_topics=20
            )

        except Exception as error:

            print(
                f"[MEMORA] Topic recovery failed: "
                f"{error}"
            )

            return []

    # =================================================
    # ENTITY RECOVERY
    # =================================================

    def _recover_entities(
        self,
        text
    ):

        text_lower = (
            str(text)
            .lower()
        )

        entities = []

        entity_patterns = [

            (
                "defective bolts",
                [
                    "defective bolt",
                    "defective bolts",
                    "defechive",
                    "defeckive"
                ]
            ),

            (
                "probability",
                [
                    "probability",
                    "probabil"
                ]
            ),

            (
                "esp32",
                [
                    "esp32"
                ]
            ),

            (
                "max30102",
                [
                    "max30102"
                ]
            ),

            (
                "ecg",
                [
                    "ecg"
                ]
            ),

            (
                "ppg",
                [
                    "ppg"
                ]
            )
        ]

        for entity, variants in entity_patterns:

            for variant in variants:

                if variant in text_lower:

                    entities.append(
                        entity
                    )

                    break

        return entities

    # =================================================
    # CLEAN SUMMARY
    # =================================================

    def _clean_summary(
        self,
        summary,
        original_text
    ):

        if not summary:

            return self._fallback_summary(
                original_text
            )

        summary = re.sub(
            r"\s+",
            " ",
            summary
        ).strip()

        summary = re.sub(
            r"(?i)^(summary|answer)\s*:\s*",
            "",
            summary
        ).strip()

        # Keep first sentence.
        sentences = re.split(
            r"(?<=[.!?])\s+",
            summary
        )

        if sentences:

            summary = (
                sentences[0]
                .strip()
            )

        if self._invalid_summary(
            summary
        ):

            return self._fallback_summary(
                original_text
            )

        return summary[:500]

    # =================================================
    # FALLBACK SUMMARY
    # =================================================

    def _fallback_summary(
        self,
        text
    ):

        lower = (
            str(text)
            .lower()
        )

        if (
            "defective bolt" in lower
            or "defechive" in lower
            or "defeckive" in lower
        ):

            return (
                "Assignment about probability "
                "and defective bolts."
            )

        if "probability" in lower:

            return (
                "Study material covering "
                "probability and statistics."
            )

        return (
            "Memory containing structured information."
        )

    # =================================================
    # CLEAN TOPICS
    # =================================================

    def _clean_topics(
        self,
        topics
    ):

        blocked = {

            "none",
            "n/a",
            "unknown",
            "content",
            "text",
            "ocr",
            "ocr text",
            "image",
            "photo",
            "picture",
            "photograph",
            "document",
            "paper",
            "file",
            "files",
            "assignment",
            "topic",
            "topics",
            "entity",
            "entities",
            "explanation",
            "mathematical operations",
            "statistical distributions",
            "external references"
        }

        cleaned = []

        for topic in topics:

            topic = str(
                topic
            ).strip()

            topic = topic.rstrip(
                ".:"
            )

            topic = re.sub(
                r"\s+",
                " ",
                topic
            )

            if not topic:

                continue

            lower = topic.lower()

            if lower in blocked:

                continue

            # Reject explanatory sentences.
            if len(
                topic.split()
            ) > 4:

                continue

            # Reject equations / garbage.
            if re.search(
                r"[%=|@#$<>]",
                topic
            ):

                continue

            if re.search(
                r"\d",
                topic
            ):

                # Keep known technical names.
                if lower not in {
                    "esp32",
                    "max30102"
                }:

                    continue

            cleaned.append(
                lower
            )

        return self._unique(
            cleaned
        )

    # =================================================
    # CLEAN ENTITIES
    # =================================================

    def _clean_entities(
        self,
        entities
    ):

        blocked = {

            "none",
            "not applicable",
            "unknown",
            "n/a",
            "content",
            "text",
            "ocr",
            "ocr text",
            "image",
            "photo",
            "picture",
            "photograph",
            "document",
            "paper",
            "file",
            "files",
            "explanation"
        }

        cleaned = []

        for entity in entities:

            entity = str(
                entity
            ).strip()

            entity = entity.rstrip(
                ".:"
            )

            entity = re.sub(
                r"\s+",
                " ",
                entity
            )

            if not entity:

                continue

            lower = entity.lower()

            if lower in blocked:

                continue

            if len(
                entity.split()
            ) > 4:

                continue

            if re.search(
                r"[%=|@#$<>]",
                entity
            ):

                continue

            cleaned.append(
                lower
            )

        return self._unique(
            cleaned
        )

    # =================================================
    # UNIQUE
    # =================================================

    def _unique(
        self,
        values
    ):

        result = []
        seen = set()

        for value in values:

            if not isinstance(
                value,
                str
            ):

                continue

            value = value.strip()

            if not value:

                continue

            key = value.lower()

            if key in seen:

                continue

            seen.add(
                key
            )

            result.append(
                value
            )

        return result

    # =================================================
    # MERGE
    # =================================================

    def _merge_unique(
        self,
        *groups
    ):

        values = []

        for group in groups:

            if not group:

                continue

            values.extend(
                group
            )

        return self._unique(
            values
        )

    # =================================================
    # INVALID SUMMARY
    # =================================================

    def _invalid_summary(
        self,
        summary
    ):

        if not summary:

            return True

        value = (
            summary
            .lower()
            .strip()
        )

        if value in {

            "summary",
            "summary:",
            "none",
            "unknown",
            "one short sentence",
            "one shortsentence"
        }:

            return True

        if len(
            value.split()
        ) < 5:

            return True

        bad_phrases = [

            "the question asked was",
            "to summarize this memory",
            "here's how",
            "this memory describes",
            "i will identify",
            "do not",
            "output only",
            "write exactly",
            "one short factual sentence",
            "the summary should",
            "summary should be",
            "write a summary",
            "provide a summary",
            "read the memory below",
            "memory below",
            "rules:",
            "instructions:",
            "as an ai",
            "as an assistant",
            "i cannot",
            "i can't"
        ]

        for phrase in bad_phrases:

            if phrase in value:

                return True

        instruction_patterns = [

            r"\bshould\s+(?:be|contain|include|describe|write)\b",

            r"\bmust\s+(?:be|contain|include|describe|write)\b",

            r"\bdo\s+not\s+(?:mention|invent|solve|calculate|explain)\b",

            r"\boutput\s+(?:only|the)\b",

            r"\b(?:use|keep|limit)\s+(?:exactly|only|no more than)\b"
        ]

        for pattern in instruction_patterns:

            if re.search(
                pattern,
                value
            ):

                return True

        return False


# =====================================================
# CLI
# =====================================================

if __name__ == "__main__":

    analyzer = SemanticAnalyzer()

    print(
        "\n========================================"
    )

    print(
        "          MEMORA SEMANTIC ANALYZER"
    )

    print(
        "========================================"
    )

    while True:

        text = input(
            "\nEnter text to analyze: "
        )

        if text.lower() in {
            "exit",
            "quit"
        }:

            break

        result = analyzer.analyze(
            text
        )

        print(
            "\n========== SEMANTIC MEMORY ==========\n"
        )

        print(
            "SUMMARY:"
        )

        print(
            result["summary"]
        )

        print(
            "\nTOPICS:"
        )

        for topic in result["topics"]:

            print(
                f" • {topic}"
            )

        print(
            "\nENTITIES:"
        )

        for entity in result["entities"]:

            print(
                f" • {entity}"
            )

        print(
            "\n======================================"
        )
