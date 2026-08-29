import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        self.model.to(self.device)

        print(
            f"[MEMORA] Reconstruction model "
            f"ready on {self.device}."
        )

    def _calculate_confidence(self, memories):
        """
        Estimate confidence from retrieval evidence.

        This is NOT a probability produced by the LLM.
        It is a conservative heuristic based on the
        reranker score and the gap between the top results.
        """

        if not memories:
            return 0.0

        scores = [
            memory.get("score")
            for memory in memories
            if memory.get("score") is not None
        ]

        if not scores:
            return 0.0

        best = scores[0]

        if len(scores) > 1:
            second = scores[1]
        else:
            second = best - 1.0

        margin = best - second

        if best >= 5 and margin >= 2:
            return 0.90

        if best >= 2 and margin >= 1:
            return 0.75

        if best >= 0 and margin >= 0.5:
            return 0.60

        if best >= 0:
            return 0.45

        return 0.25

    def reconstruct(
        self,
        query,
        memories
    ):

        # -----------------------------------------
        # No memories
        # -----------------------------------------

        if not memories:

            return {
                "answer": (
                    "I couldn't find a relevant "
                    "memory."
                ),
                "confidence": 0.0,
                "evidence": []
            }

        # -----------------------------------------
        # Build evidence
        # -----------------------------------------

        context_parts = []
        evidence = []

        for index, memory in enumerate(memories):

            document = memory.get(
                "document",
                ""
            )

            metadata = memory.get(
                "metadata",
                {}
            )

            source = metadata.get(
                "source",
                "Unknown"
            )

            modality = metadata.get(
                "modality",
                "unknown"
            )

            score = memory.get(
                "score",
                "unknown"
            )

            description = metadata.get(
                "description",
                ""
            )

            context_parts.append(
                f"""
MEMORY {index + 1}

Source:
{source}

Type:
{modality}

Relevance score:
{score}

Extracted evidence:
{document}

Visual description:
{description}
"""
            )

            if source not in evidence:
                evidence.append(source)

        context = "\n".join(
            context_parts
        )

        # -----------------------------------------
        # Reconstruction prompt
        # -----------------------------------------

        prompt = f"""
You are MEMORA, a private digital memory assistant.

Your job is to help the user remember something from
their stored digital memories.

USER QUERY:
{query}

RETRIEVED MEMORIES:
{context}

IMPORTANT RULES:

1. Use ONLY the evidence provided above.
2. Do NOT use outside knowledge.
3. Do NOT invent facts.
4. Do NOT claim something is visible unless the evidence
   explicitly supports it.
5. OCR may contain mistakes because it can come from
   handwritten or low-quality images.
6. The memories are ordered by relevance.
7. MEMORY 1 is the highest-ranked result.
8. Do not replace the highest-ranked result with a
   different source unless the evidence clearly shows
   that it is incorrect.
9. Prefer concrete evidence over assumptions.
10. If evidence is insufficient, say that you are unsure.
11. Keep the response concise.

The user wants to know what they were probably remembering.

Respond using exactly this structure:

I think you're remembering:
[short description]

Best match:
[filename]

Why:
[one or two concrete reasons based ONLY on the evidence]
"""

        # -----------------------------------------
        # LLM messages
        # -----------------------------------------

        messages = [
            {
                "role": "system",
                "content": (
                    "You are MEMORA. "
                    "You are precise, grounded, "
                    "and never invent evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        formatted_prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        # -----------------------------------------
        # Tokenize
        # -----------------------------------------

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # -----------------------------------------
        # Generate
        # -----------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False
            )

        # -----------------------------------------
        # Remove prompt tokens
        # -----------------------------------------

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        # -----------------------------------------
        # Confidence
        # -----------------------------------------

        confidence = self._calculate_confidence(
            memories
        )

        # -----------------------------------------
        # Final response
        # -----------------------------------------

        return {
            "answer": answer,
            "confidence": confidence,
            "evidence": evidence
        }