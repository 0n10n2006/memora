import re


class KeywordExtractor:

    """
    Deterministic concept extraction for MEMORA.

    This layer provides reliable semantic signals without
    depending on an LLM. These signals are later used by
    semantic analysis, retrieval and relationship discovery.
    """

    KNOWN_CONCEPTS = {

        "probability": [
            "probability",
            "probabilistic",
            "probabiliby",
            "probabiliy",
            "probabil"
        ],

        "conditional probability": [
            "conditional probability",
            "conditional",
            "conditionary",
            "condibional",
            "condihoneal",
            "conditional distribution"
        ],

        "defective bolts": [
            "defective bolt",
            "defective bolts",
            "defect bolt",
            "defect bolts",
            "defective",
            "defechive",
            "defeckive"
        ],

        "probability distributions": [
            "probability distribution",
            "probability distributions"
        ],

        "exponential distribution": [
            "exponential distribution",
            "exponential"
        ],

        "marginal distribution": [
            "marginal distribution",
            "marginal"
        ],

        "random variables": [
            "random variable",
            "random variables"
        ],

        "statistics": [
            "statistics",
            "statistical"
        ],

        "normal distribution": [
            "normal distribution"
        ],

        "binomial distribution": [
            "binomial distribution",
            "binomial"
        ],

        "poisson distribution": [
            "poisson distribution",
            "poisson"
        ],

        "hypothesis testing": [
            "hypothesis testing",
            "hypothesis test"
        ],

        "confidence intervals": [
            "confidence interval",
            "confidence intervals"
        ],

        "central limit theorem": [
            "central limit theorem"
        ],

        "bayes theorem": [
            "bayes theorem",
            "bayes' theorem",
            "bayes"
        ],

        "independence": [
            "independence of events",
            "independent events"
        ],

        "expectation": [
            "expectation",
            "expected value"
        ],

        "variance": [
            "variance"
        ],

        "sampling": [
            "sampling distribution",
            "sampling distributions",
            "sampling"
        ],

        "hypothesis errors": [
            "type i error",
            "type ii error",
            "type 1 error",
            "type 2 error"
        ],
    }

    def extract(self, text):

        if not text:
            return []

        normalized = self._normalize(
            text
        )

        found = []

        for concept, variants in self.KNOWN_CONCEPTS.items():

            for variant in variants:

                if self._contains_concept(
                    normalized,
                    variant
                ):

                    found.append(
                        concept
                    )

                    break

        # Remove overly generic concepts when a
        # more specific concept already exists.

        found = self._remove_redundant_topics(
            found
        )

        return self._unique(
            found
        )

    # =================================================
    # NORMALIZATION
    # =================================================

    def _normalize(self, text):

        text = str(text).lower()

        # Fix common OCR errors.
        replacements = {

            "probabiliby":
                "probability",

            "probabiliy":
                "probability",

            "defechive":
                "defective",

            "defeckive":
                "defective",

            "condihoneal":
                "conditional",

            "condibional":
                "conditional",

            "condihone":
                "condition",

            "distribubion":
                "distribution",

            "distribubions":
                "distributions",

            "srequived":
                "required",
        }

        for incorrect, correct in replacements.items():

            text = text.replace(
                incorrect,
                correct
            )

        # Normalize whitespace.
        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # =================================================
    # CONCEPT MATCHING
    # =================================================

    def _contains_concept(
        self,
        text,
        variant
    ):

        variant = variant.lower().strip()

        if not variant:
            return False

        # Multi-word concepts should be matched
        # as phrases.

        if " " in variant:

            return variant in text

        # Single words need word boundaries.
        # This prevents things like:
        #
        # "normalization" → "normal"
        #
        # from becoming a false match.

        pattern = (
            r"\b"
            + re.escape(variant)
            + r"\b"
        )

        return bool(
            re.search(
                pattern,
                text
            )
        )

    # =================================================
    # REDUNDANT TOPICS
    # =================================================

    def _remove_redundant_topics(
        self,
        topics
    ):

        topics = self._unique(
            topics
        )

        # If a specific distribution is present,
        # don't additionally label it simply as
        # "probability distributions" unless the text
        # explicitly contains that phrase.

        specific_distributions = {
            "exponential distribution",
            "normal distribution",
            "binomial distribution",
            "poisson distribution",
            "marginal distribution"
        }

        if (
            "probability distributions" in topics
            and any(
                topic in topics
                for topic
                in specific_distributions
            )
        ):

            # Keep it only when explicitly supported
            # by the source; since extract() only adds
            # explicit matches, don't remove it blindly.
            pass

        return topics

    # =================================================
    # UNIQUE
    # =================================================

    def _unique(self, values):

        result = []

        seen = set()

        for value in values:

            key = value.lower().strip()

            if key in seen:
                continue

            seen.add(key)

            result.append(
                value
            )

        return result


if __name__ == "__main__":

    extractor = KeywordExtractor()

    print(
        "\n========================================"
    )

    print(
        "          MEMORA KEYWORD EXTRACTOR"
    )

    print(
        "========================================"
    )

    while True:

        text = input(
            "\nEnter text: "
        )

        if text.lower() in {
            "exit",
            "quit"
        }:

            break

        topics = extractor.extract(
            text
        )

        print(
            "\n========== KEYWORD SIGNALS ==========\n"
        )

        if not topics:

            print(
                "No known concepts detected."
            )

        else:

            for topic in topics:

                print(
                    f" • {topic}"
                )

        print(
            "\n======================================"
        )