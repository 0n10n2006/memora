import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from backend.memory.memory_understanding import (
    MemoryUnderstanding
)

from backend.memory.topic_extractor import (
    TopicExtractor
)


class MemoryReconstructor:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B-Instruct"
    ):

        print(
            f"[MEMORA] Loading reconstruction model: "
            f"{model_name}"
        )

        self.model_name = model_name

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

        self.model.to(
            self.device
        )

        self.model.eval()

        self.understanding = (
            MemoryUnderstanding()
        )

        self.topic_extractor = (
            TopicExtractor()
        )

        print(
            f"[MEMORA] Reconstruction model "
            f"ready on {self.device}."
        )

    # =====================================================
    # PRIMARY FACT EXTRACTION
    # =====================================================

    def _build_primary_facts(
        self,
        memory
    ):

        metadata = memory.get(
            "metadata",
            {}
        )

        document = memory.get(
            "document",
            ""
        )

        normalized = (
            self.understanding.normalize_text(
                document
            )
        )

        stored_topics = metadata.get(
            "topics",
            []
        )

        stored_entities = metadata.get(
            "entities",
            []
        )

        concepts = []

        for item in (
            stored_topics
            + stored_entities
        ):

            if not isinstance(
                item,
                str
            ):
                continue

            item = item.strip()

            if not item:
                continue

            if item.lower() not in [
                x.lower()
                for x in concepts
            ]:

                concepts.append(
                    item
                )

        try:

            recovered_topics = (
                self.topic_extractor.extract(
                    normalized,
                    max_topics=20
                )
            )

        except Exception:

            recovered_topics = []

        for item in recovered_topics:

            if not isinstance(
                item,
                str
            ):
                continue

            if item.lower() not in [
                x.lower()
                for x in concepts
            ]:

                concepts.append(
                    item
                )

        canonical_concepts = [

            "conditional probability",
            "probability distribution",
            "exponential distribution",
            "marginal distribution",
            "joint distribution",
            "binomial distribution",
            "poisson distribution",
            "normal distribution",
            "random variable",
            "expected value",
            "expectation",
            "variance",
            "Bayes theorem",
            "coefficient of correlation",
            "correlation",
            "independence",
            "defective bolts",
            "hypothesis testing",
            "confidence interval",
            "central limit theorem",
            "statistics",
            "sampling",
        ]

        final_concepts = []

        lower_text = normalized.lower()

        for concept in canonical_concepts:

            if concept.lower() in lower_text:

                final_concepts.append(
                    concept
                )

        for concept in concepts:

            if concept.lower() not in [
                x.lower()
                for x in final_concepts
            ]:

                final_concepts.append(
                    concept
                )

        question_types = []

        def add_question(value):

            if value.lower() not in [
                x.lower()
                for x in question_types
            ]:

                question_types.append(
                    value
                )

        if (
            "defective bolts"
            in [
                x.lower()
                for x in final_concepts
            ]
        ):

            add_question(
                "probability involving defective bolts"
            )

        if (
            "conditional probability"
            in [
                x.lower()
                for x in final_concepts
            ]
        ):

            add_question(
                "conditional probability"
            )

        if (
            "exponential distribution"
            in [
                x.lower()
                for x in final_concepts
            ]
        ):

            add_question(
                "exponential distribution and repair time"
            )

        if (
            "marginal distribution"
            in [
                x.lower()
                for x in final_concepts
            ]
        ):

            add_question(
                "marginal distribution"
            )

        if (
            "joint distribution"
            in [
                x.lower()
                for x in final_concepts
            ]
        ):

            add_question(
                "joint distribution"
            )

        if (
            "correlation"
            in [
                x.lower()
                for x in final_concepts
            ]
            or
            "coefficient of correlation"
            in [
                x.lower()
                for x in final_concepts
            ]
        ):

            add_question(
                "coefficient of correlation"
            )

        if (
            "exceed"
            in lower_text
            and (
                "hrs"
                in lower_text
                or "hours"
                in lower_text
            )
        ):

            add_question(
                "probability involving repair time"
            )

        return {

            "concepts":
                final_concepts[:20],

            "question_types":
                question_types[:10],

            "normalized_content":
                normalized

        }

    # =====================================================
    # QUERY INTENT
    # =====================================================

    def _query_is_question_recall(
        self,
        query
    ):

        query = query.lower()

        patterns = [

            "what were",
            "what was",
            "which questions",
            "what questions",
            "questions i had",
            "questions did i",
            "what did i ask",
            "what was that question",
            "what were those questions",
            "remind me what",
            "do you remember the questions",

        ]

        return any(
            pattern in query
            for pattern in patterns
        )

    # =====================================================
    # RELATIONSHIP QUERY DETECTION
    # =====================================================

    def _query_is_relationship(
        self,
        query
    ):

        query = query.lower().strip()

        patterns = [

            "how are",
            "how is",
            "how do",
            "how does",
            "what is the relationship",
            "what's the relationship",
            "what is the connection",
            "what's the connection",
            "how are they related",
            "how are these related",
            "how are those related",
            "are they related",
            "are these related",
            "are those related",
            "related",
            "relationship",
            "connection",
            "connected",
            "link between",
            "linked",
            "similar",
            "in common",
            "what do they have in common",

        ]

        return any(
            pattern in query
            for pattern in patterns
        )

    # =====================================================
    # EXTRACT QUESTION DETAILS
    # =====================================================

    def _extract_question_details(
        self,
        content,
        concepts
    ):

        text = content.lower()

        questions = []

        def add(value):

            value = value.strip()

            if not value:
                return

            if value.lower() not in [
                x.lower()
                for x in questions
            ]:

                questions.append(
                    value
                )

        if (
            "defective"
            in text
            and "bolt"
            in text
            and "probab"
            in text
        ):

            add(
                "probability of selecting or finding a defective bolt"
            )

        if (
            "conditional"
            in text
            and "probab"
            in text
        ):

            add(
                "conditional probability"
            )

        if (
            "exponential"
            in text
            or (
                "distribution"
                in text
                and (
                    "hrs"
                    in text
                    or "hours"
                    in text
                )
            )
        ):

            add(
                "exponential distribution and repair time"
            )

        if (
            "exceed"
            in text
            and (
                "hrs"
                in text
                or "hours"
                in text
            )
        ):

            add(
                "probability that the repair time exceeds a given number of hours"
            )

        if (
            "given"
            in text
            and "exceed"
            in text
            and (
                "hrs"
                in text
                or "hours"
                in text
            )
        ):

            add(
                "conditional probability involving repair times"
            )

        if (
            "joint"
            in text
            and "distribution"
            in text
        ):

            add(
                "joint distribution of X and Y"
            )

        if (
            "marginal"
            in text
            and "distribution"
            in text
        ):

            add(
                "marginal distribution"
            )

        if (
            "conditional"
            in text
            and "distribution"
            in text
        ):

            add(
                "conditional distribution"
            )

        if (
            "correlation"
            in text
            or (
                "coef"
                in text
                and "correlation"
                in text
            )
        ):

            add(
                "coefficient of correlation"
            )

        return questions[:10]

    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    def _build_context(
        self,
        memories
    ):

        blocks = []

        for index, memory in enumerate(
            memories,
            start=1
        ):

            metadata = memory.get(
                "metadata",
                {}
            )

            retrieval_type = memory.get(
                "retrieval_type",
                "unknown"
            )

            source = metadata.get(
                "source",
                "unknown"
            )

            title = metadata.get(
                "title",
                ""
            )

            topics = metadata.get(
                "topics",
                []
            )

            entities = metadata.get(
                "entities",
                []
            )

            block = [

                f"MEMORY {index}",

                f"TYPE: {retrieval_type}",

                f"SOURCE: {source}",

                f"TITLE: {title}",

            ]

            if topics:

                block.append(
                    "TOPICS: "
                    + ", ".join(
                        str(x)
                        for x in topics
                    )
                )

            if entities:

                block.append(
                    "ENTITIES: "
                    + ", ".join(
                        str(x)
                        for x in entities
                    )
                )

            if retrieval_type == "primary":

                facts = (
                    self._build_primary_facts(
                        memory
                    )
                )

                block.append(
                    "PRIMARY CONCEPTS: "
                    + ", ".join(
                        facts["concepts"]
                    )
                )

                questions = (
                    self._extract_question_details(
                        facts[
                            "normalized_content"
                        ],
                        facts["concepts"]
                    )
                )

                if questions:

                    block.append(
                        "RECOGNIZED QUESTIONS: "
                        + " | ".join(
                            questions
                        )
                    )

                block.append(
                    "CONTENT:\n"
                    + facts[
                        "normalized_content"
                    ][:7000]
                )

            blocks.append(
                "\n".join(
                    block
                )
            )

        return "\n\n".join(
            blocks
        )

    # =====================================================
    # DETERMINISTIC QUESTION ANSWER
    # =====================================================

    def _build_question_recall(
        self,
        query,
        memories
    ):

        primary = None

        for memory in memories:

            if memory.get(
                "retrieval_type"
            ) == "primary":

                primary = memory
                break

        if primary is None:

            primary = memories[0]

        metadata = primary.get(
            "metadata",
            {}
        )

        title = metadata.get(
            "title",
            ""
        )

        source = metadata.get(
            "source",
            ""
        )

        facts = (
            self._build_primary_facts(
                primary
            )
        )

        content = facts[
            "normalized_content"
        ]

        questions = (
            self._extract_question_details(
                content,
                facts["concepts"]
            )
        )

        if questions:

            name = (
                title
                if title
                else source
            )

            answer = (
                f"Yes — you're remembering "
                f"{name}. The recognizable questions "
                f"were:"
            )

            for question in questions:

                answer += (
                    "\n• "
                    + question
                )

            return answer

        if facts["question_types"]:

            answer = (
                "You're remembering a probability "
                "assignment. The recognizable topics "
                "were:"
            )

            for question in facts[
                "question_types"
            ]:

                answer += (
                    "\n• "
                    + question
                )

            return answer

        return (
            "I found the probability memory, "
            "but the exact question details "
            "could not be reliably reconstructed."
        )

    # =====================================================
    # DETERMINISTIC RELATIONSHIP ANSWER
    # =====================================================

    def _build_relationship_answer(
        self,
        query,
        memories
    ):

        relationships = []

        for memory in memories:

            relationship = memory.get(
                "relationship",
                {}
            )

            if (
                not relationship
                or not isinstance(
                    relationship,
                    dict
                )
            ):
                continue

            score = relationship.get(
                "score"
            )

            try:

                score = (
                    float(score)
                    if score is not None
                    else None
                )

            except (
                TypeError,
                ValueError
            ):

                score = None

            strength = str(
                relationship.get(
                    "strength",
                    ""
                )
                or ""
            ).strip().lower()

            relationships.append({

                "memory": memory,

                "score": score,

                "type": (
                    relationship.get(
                        "type"
                    )
                    or relationship.get(
                        "relationship_type"
                    )
                    or "related"
                ),

                "topics": (
                    relationship.get(
                        "shared_topics",
                        []
                    )
                    or []
                ),

                "entities": (
                    relationship.get(
                        "shared_entities",
                        []
                    )
                    or []
                ),

                "strength": strength,

                "temporal_relation": str(
                    relationship.get(
                        "temporal_relation",
                        ""
                    )
                    or ""
                ).strip(),

                "contradictions": list(
                    relationship.get(
                        "contradictions",
                        []
                    )
                    or []
                ),

            })

        if not relationships:

            return None

        relationships.sort(
            key=lambda item: (
                item["score"]
                if item["score"] is not None
                else 0.0
            ),
            reverse=True
        )

        best = relationships[0]

        relationship_type = str(
            best["type"]
        ).strip()

        score = best["score"]

        shared_topics = []

        for topic in best["topics"]:

            if not isinstance(
                topic,
                str
            ):
                continue

            topic = topic.strip()

            if (
                topic
                and topic.lower()
                not in [
                    x.lower()
                    for x in shared_topics
                ]
            ):

                shared_topics.append(
                    topic
                )

        shared_entities = []

        for entity in best["entities"]:

            if not isinstance(
                entity,
                str
            ):
                continue

            entity = entity.strip()

            if (
                entity
                and entity.lower()
                not in [
                    x.lower()
                    for x in shared_entities
                ]
            ):

                shared_entities.append(
                    entity
                )

        # The relationship engine explicitly classifies strength from
        # its evidence. Prefer that classification to reinterpreting
        # the combined score here: a strong topic-overlap relationship
        # can legitimately have a sub-0.50 combined score.
        if best["strength"] in {
            "strong",
            "moderate",
            "weak"
        }:

            strength = {
                "strong": "strongly",
                "moderate": "moderately",
                "weak": "loosely",
            }[best["strength"]]

        elif score is not None:

            if score >= 0.50:

                strength = "strongly"

            elif score >= 0.30:

                strength = "moderately"

            elif score >= 0.15:

                strength = "loosely"

            else:

                strength = "weakly"

        else:

            strength = "directly"

        normalized_type = (
            relationship_type
            .lower()
            .replace("_", " ")
            .strip()
        )

        if normalized_type == "same topic":

            relation_phrase = (
                "share the same broader topic"
            )

        elif normalized_type == "same_topic":

            relation_phrase = (
                "share the same broader topic"
            )

        elif normalized_type == "semantic":

            relation_phrase = (
                "are semantically related"
            )

        elif normalized_type == "related":

            relation_phrase = (
                "are related"
            )

        else:

            relation_phrase = (
                f"have a {normalized_type} relationship"
            )

        primary_memories = [
            memory
            for memory in memories
            if memory.get("retrieval_type") == "primary"
        ]

        related_memories = [
            memory
            for memory in memories
            if memory.get("retrieval_type") == "related"
        ]

        def memory_role(memory):

            summary = str(
                memory.get("metadata", {}).get(
                    "summary",
                    ""
                )
                or ""
            ).lower()

            if "assignment" in summary:
                return "the assignment"

            if "study material" in summary or "study notes" in summary:
                return "the study notes"

            title_lower = str(
                memory.get("metadata", {}).get("title", "") or ""
            ).lower()

            if "notes" in title_lower:
                return "the study notes"

            document_lower = str(memory.get("document", "") or "").lower()

            if (
                "assignment" in document_lower
                or ("defective" in document_lower and "bolt" in document_lower)
            ):
                return "the assignment"

            title = str(
                memory.get("metadata", {}).get(
                    "title",
                    ""
                )
                or ""
            ).strip()

            return title or "the memory"

        if primary_memories and related_memories:

            subject = (
                f"{memory_role(primary_memories[0]).capitalize()} "
                f"and {memory_role(related_memories[0])}"
            )

        elif len(primary_memories) >= 2:

            subject = (
                f"{memory_role(primary_memories[0]).capitalize()} "
                f"and {memory_role(primary_memories[1])}"
            )

        else:

            subject = "The two memories"

        answer = (
            f"{subject} are {strength} related because "
            f"they {relation_phrase}."
        )

        if shared_topics:

            if len(shared_topics) == 1:

                topic_text = (
                    shared_topics[0]
                )

            elif len(shared_topics) == 2:

                topic_text = (
                    f"{shared_topics[0]} "
                    f"and {shared_topics[1]}"
                )

            else:

                topic_text = (
                    ", ".join(
                        shared_topics[:-1]
                    )
                    + ", and "
                    + shared_topics[-1]
                )

            answer += (
                f" They share concepts including "
                f"{topic_text}."
            )

        if shared_entities:

            if len(shared_entities) == 1:

                entity_text = (
                    shared_entities[0]
                )

            elif len(shared_entities) == 2:

                entity_text = (
                    f"{shared_entities[0]} "
                    f"and {shared_entities[1]}"
                )

            else:

                entity_text = (
                    ", ".join(
                        shared_entities[:-1]
                    )
                    + ", and "
                    + shared_entities[-1]
                )

            answer += (
                f" They also share the entity "
                f"{entity_text}."
            )

        if best["temporal_relation"]:

            answer += (
                " Temporal context: "
                + best["temporal_relation"]
                + "."
            )

        contradictions = [
            item.strip()
            for item in best["contradictions"]
            if isinstance(item, str) and item.strip()
        ]

        if contradictions:

            answer += (
                " I found conflicting stored facts ("
                + "; ".join(contradictions[:3])
                + "), so I can't determine which value is correct."
            )

        candidate_memories = primary_memories + related_memories

        for candidate in candidate_memories:

            primary_facts = self._build_primary_facts(candidate)

            concepts = [
                str(x)
                for x in primary_facts.get(
                    "concepts",
                    []
                )
            ]

            lower_concepts = [
                x.lower()
                for x in concepts
            ]

            if (
                "defective bolts"
                in lower_concepts
                and (
                    "probability"
                    in lower_concepts
                    or "conditional probability"
                    in lower_concepts
                )
            ):

                answer += (
                    " The assignment applies these "
                    "probability concepts to defective-bolt "
                    "and repair-time questions."
                )

                break

        return answer

    # =====================================================
    # DETERMINISTIC MULTI-MEMORY SYNTHESIS
    # =====================================================

    def _query_needs_multi_memory_synthesis(self, query, memories):

        if len(memories) < 2:
            return False

        query = query.lower()

        patterns = [
            "what do i remember",
            "summarize",
            "summary",
            "overview",
            "across my memories",
            "combine",
            "together",
            "compare",
        ]

        return any(pattern in query for pattern in patterns)

    def _build_multi_memory_synthesis(self, memories):

        topics_by_memory = []
        topic_counts = {}
        summaries = []
        summary_keys = set()

        for memory in memories:

            metadata = memory.get("metadata", {})
            topics = []
            topic_keys = set()

            for topic in metadata.get("topics", []) or []:
                if not isinstance(topic, str):
                    continue

                topic = topic.strip()

                topic_key = topic.casefold()

                if topic and topic_key not in topic_keys:
                    topics.append(topic)
                    topic_keys.add(topic_key)
                    topic_counts[topic_key] = (
                        topic_counts.get(topic_key, 0) + 1
                    )

            topics_by_memory.append(topics)

            summary = str(metadata.get("summary", "") or "").strip()

            summary_key = summary.casefold()

            if summary and summary_key not in summary_keys:
                summaries.append(summary)
                summary_keys.add(summary_key)

        shared_topics = [
            topic
            for topics in topics_by_memory
            for topic in topics
            if topic_counts.get(topic.casefold(), 0) >= 2
        ]

        shared_topics = list(dict.fromkeys(shared_topics))

        all_topics = []
        all_topic_keys = set()
        for topics in topics_by_memory:
            for topic in topics:
                topic_key = topic.casefold()

                if topic_key not in all_topic_keys:
                    all_topics.append(topic)
                    all_topic_keys.add(topic_key)

        if not all_topics and not summaries:
            return None

        if shared_topics:
            answer = (
                "Across these memories, the common thread is "
                + ", ".join(shared_topics[:6])
                + "."
            )
        else:
            answer = "Across these memories, I found:"

        for summary in summaries[:4]:
            answer += "\n• " + summary

        shared_topic_keys = {
            item.casefold() for item in shared_topics
        }

        remaining_topics = [
            topic for topic in all_topics
            if topic.casefold() not in shared_topic_keys
        ]

        if remaining_topics:
            answer += (
                "\nOther covered concepts: "
                + ", ".join(remaining_topics[:10])
                + "."
            )

        contradictions = []
        for memory in memories:
            relationship = memory.get("relationship", {}) or {}
            for item in relationship.get("contradictions", []) or []:
                if isinstance(item, str) and item not in contradictions:
                    contradictions.append(item)

        if contradictions:
            answer += (
                "\nConflicting stored facts: "
                + "; ".join(contradictions[:3])
                + ". I can't determine which value is correct."
            )

        return answer

    # =====================================================
    # GENERAL LLM RECONSTRUCTION
    # =====================================================

    def _generate_general_answer(
        self,
        query,
        memories
    ):

        context = (
            self._build_context(
                memories
            )
        )

        prompt = f"""
You are MEMORA, a personal memory assistant.

Answer the user's question using ONLY the supplied
memory evidence.

USER:
{query}

MEMORY:
{context}

Rules:

- Do not invent facts.
- Do not invent exact wording.
- Do not mention OCR.
- Do not mention retrieval.
- Do not mention models or databases.
- Do not say "the user asked".
- Be concise.
- If information is uncertain, say that it is uncertain.
- Prefer concrete remembered facts over generic summaries.

Return only the answer.
"""

        messages = [

            {
                "role": "system",
                "content": (
                    "You answer questions from "
                    "stored memory evidence only."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

        formatted = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=8192
        )

        inputs = {
            key: value.to(
                self.device
            )
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                )
            )

        generated = (
            outputs[0][
                inputs["input_ids"].shape[1]:
            ]
        )

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        return self._clean_answer(
            answer
        )

    # =====================================================
    # CLEAN ANSWER
    # =====================================================

    def _clean_answer(
        self,
        answer
    ):

        if not answer:

            return ""

        answer = re.sub(
            r"^(Answer:\s*)+",
            "",
            answer,
            flags=re.IGNORECASE
        )

        answer = re.sub(
            r"^(Sure[,!]\s*)+",
            "",
            answer,
            flags=re.IGNORECASE
        )

        return answer.strip()

    # =====================================================
    # QUALITY CHECK
    # =====================================================

    def _is_low_quality_answer(
        self,
        answer
    ):

        if not answer:

            return True

        lowered = answer.lower()

        bad_patterns = [

            "i don't have enough information",

            "i cannot determine",

            "i can't determine",

            "the user asked",

            "the question asks",

            "the answer should",

            "based on the evidence",

            "retrieved evidence",

            "memory evidence",

            "as an ai",

            "i am an ai",

            "i'm an ai",

            "i cannot answer",

            "i can't answer",

            "the task requires",

            "i will identify",

            "to analyze the memory",

        ]

        if any(
            pattern in lowered
            for pattern in bad_patterns
        ):

            return True

        if len(
            answer.split()
        ) < 8:

            return True

        return False

    # =====================================================
    # FALLBACK
    # =====================================================

    def _build_fallback_answer(
        self,
        query,
        memories
    ):

        if self._query_is_relationship(
            query
        ):

            relationship_answer = (
                self._build_relationship_answer(
                    query,
                    memories
                )
            )

            if relationship_answer:

                return relationship_answer

        primary = None

        for memory in memories:

            if memory.get(
                "retrieval_type"
            ) == "primary":

                primary = memory
                break

        if primary is None:

            primary = memories[0]

        facts = (
            self._build_primary_facts(
                primary
            )
        )

        questions = (
            self._extract_question_details(
                facts[
                    "normalized_content"
                ],
                facts["concepts"]
            )
        )

        if questions:

            return self._build_question_recall(
                query,
                memories
            )

        if facts["concepts"]:

            return (
                "I found a memory about "
                + ", ".join(
                    facts["concepts"][:8]
                )
                + "."
            )

        return (
            "I found a related memory, "
            "but its contents could not "
            "be reconstructed reliably."
        )

    # =====================================================
    # CONFIDENCE
    # =====================================================

    def _calculate_confidence(
        self,
        memories
    ):

        if not memories:

            return 0.0

        primary_scores = []

        related_scores = []

        for memory in memories:

            retrieval_type = memory.get(
                "retrieval_type"
            )

            if retrieval_type == "primary":

                score = memory.get(
                    "score"
                )

                if score is not None:

                    primary_scores.append(
                        float(score)
                    )

            elif retrieval_type == "related":

                relationship = memory.get(
                    "relationship",
                    {}
                )

                score = relationship.get(
                    "score"
                )

                if score is not None:

                    related_scores.append(
                        float(score)
                    )

        if primary_scores:

            best = max(
                primary_scores
            )

            normalized = (
                1.0
                /
                (
                    1.0
                    +
                    pow(
                        2.718281828,
                        -best
                    )
                )
            )

            confidence = (
                0.35
                +
                normalized
                * 0.50
            )

        else:

            confidence = 0.25

        if related_scores:

            best_related = max(
                related_scores
            )

            if best_related >= 0.50:

                confidence += 0.08

            elif best_related >= 0.30:

                confidence += 0.05

            elif best_related >= 0.15:

                confidence += 0.03

        return round(
            max(
                0.0,
                min(
                    confidence,
                    0.95
                )
            ),
            2
        )

    # =====================================================
    # RELATIONSHIP-SPECIFIC CONFIDENCE
    # =====================================================

    def _calculate_relationship_confidence(
        self,
        memories
    ):

        scores = []
        strengths = []

        for memory in memories:

            relationship = memory.get(
                "relationship",
                {}
            )

            if not isinstance(
                relationship,
                dict
            ):

                continue

            score = relationship.get(
                "score"
            )

            strength = str(
                relationship.get(
                    "strength",
                    ""
                )
                or ""
            ).lower()

            if strength:
                strengths.append(strength)

            try:

                if score is not None:

                    scores.append(
                        float(score)
                    )

            except (
                TypeError,
                ValueError
            ):

                continue

        if not scores and not strengths:

            return None

        best = max(scores) if scores else 0.0

        if best >= 0.60:

            confidence = 0.90

        elif best >= 0.50:

            confidence = 0.85

        elif best >= 0.40:

            confidence = 0.80

        elif best >= 0.30:

            confidence = 0.72

        elif best >= 0.20:

            confidence = 0.62

        elif best >= 0.15:

            confidence = 0.52

        else:

            confidence = 0.35

        strength_floor = {
            "strong": 0.85,
            "moderate": 0.70,
            "weak": 0.45,
        }

        for strength in strengths:

            confidence = max(
                confidence,
                strength_floor.get(strength, 0.0)
            )

        return round(confidence, 2)

    # =====================================================
    # MAIN RECONSTRUCTION
    # =====================================================

    def reconstruct(
        self,
        query,
        memories
    ):

        if not memories:

            return {

                "answer":
                    "I couldn't find a relevant memory.",

                "confidence":
                    0.0,

                "evidence":
                    []

            }

        evidence = []

        for memory in memories:

            source = (
                memory
                .get(
                    "metadata",
                    {}
                )
                .get(
                    "source"
                )
            )

            if (
                source
                and source not in evidence
            ):

                evidence.append(
                    source
                )

        # =================================================
        # QUESTION RECALL
        # =================================================

        if self._query_is_question_recall(
            query
        ):

            print(
                "[MEMORA] Question-recall query "
                "detected; using deterministic "
                "question reconstruction."
            )

            answer = (
                self._build_question_recall(
                    query,
                    memories
                )
            )

            confidence = (
                self._calculate_confidence(
                    memories
                )
            )

        # =================================================
        # RELATIONSHIP RECONSTRUCTION
        # =================================================

        elif self._query_is_relationship(
            query
        ):

            print(
                "[MEMORA] Relationship query "
                "detected; using deterministic "
                "relationship reconstruction."
            )

            answer = (
                self._build_relationship_answer(
                    query,
                    memories
                )
            )

            relationship_confidence = (
                self._calculate_relationship_confidence(
                    memories
                )
            )

            if answer is not None:

                confidence = (
                    relationship_confidence
                    if relationship_confidence is not None
                    else self._calculate_confidence(
                        memories
                    )
                )

            else:

                print(
                    "[MEMORA] No usable relationship "
                    "metadata found; using general "
                    "reconstruction."
                )

                answer = (
                    self._generate_general_answer(
                        query,
                        memories
                    )
                )

                if self._is_low_quality_answer(
                    answer
                ):

                    print(
                        "[MEMORA] General model answer "
                        "was low quality; using "
                        "structured fallback."
                    )

                    answer = (
                        self._build_fallback_answer(
                            query,
                            memories
                        )
                    )

                confidence = (
                    self._calculate_confidence(
                        memories
                    )
                )

        # =================================================
        # MULTI-MEMORY SYNTHESIS
        # =================================================

        elif self._query_needs_multi_memory_synthesis(
            query,
            memories
        ):

            print(
                "[MEMORA] Multi-memory query detected; "
                "using deterministic synthesis."
            )

            answer = self._build_multi_memory_synthesis(
                memories
            )

            if answer is None:

                answer = self._generate_general_answer(
                    query,
                    memories
                )

            confidence = self._calculate_confidence(
                memories
            )

        # =================================================
        # GENERAL RECONSTRUCTION
        # =================================================

        else:

            answer = (
                self._generate_general_answer(
                    query,
                    memories
                )
            )

            if self._is_low_quality_answer(
                answer
            ):

                print(
                    "[MEMORA] General model answer "
                    "was low quality; using "
                    "structured fallback."
                )

                answer = (
                    self._build_fallback_answer(
                        query,
                        memories
                    )
                )

            confidence = (
                self._calculate_confidence(
                    memories
                )
            )

        return {

            "answer":
                answer,

            "confidence":
                confidence,

            "evidence":
                evidence

        }


# =====================================================
# CLI TEST
# =====================================================

if __name__ == "__main__":

    from backend.memory.memory_store import (
        MemoryStore
    )

    store = MemoryStore()

    reconstructor = (
        MemoryReconstructor()
    )

    print(
        "\n========================================"
    )

    print(
        "          MEMORA RECONSTRUCTOR"
    )

    print(
        "========================================"
    )

    while True:

        query = input(
            "\nWhat do you remember? "
        )

        if query.lower() in {
            "exit",
            "quit"
        }:

            break

        memories = []

        for memory in store.all():

            memories.append({

                "document":
                    memory.content,

                "metadata": {

                    "source":
                        memory.source,

                    "title":
                        memory.title,

                    "topics":
                        memory.topics,

                    "entities":
                        memory.entities,

                },

                "retrieval_type":
                    "primary",

            })

        result = (
            reconstructor.reconstruct(
                query,
                memories
            )
        )

        print(
            "\n========== MEMORY ==========\n"
        )

        print(
            result["answer"]
        )

        print(
            "\nConfidence:",
            result["confidence"]
        )

        print(
            "\n============================"
        )
