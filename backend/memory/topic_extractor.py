import re
from collections import Counter


class TopicExtractor:

    STOPWORDS = {
        "this", "that", "with", "from", "have", "been", "were",
        "what", "when", "where", "which", "their", "there",
        "about", "into", "then", "than", "also", "some", "such",
        "more", "most", "used", "using", "given", "find",
        "following", "question", "questions", "assignment",
        "notes", "chapter", "unit", "piece", "writing", "visual",
        "document", "paper", "photo", "photograph", "picture"
    }

    DOMAIN_PHRASES = [
        "probability",
        "conditional probability",
        "bayes theorem",
        "random variable",
        "probability distribution",
        "normal distribution",
        "binomial distribution",
        "poisson distribution",
        "exponential distribution",
        "marginal distribution",
        "joint distribution",
        "hypothesis testing",
        "confidence interval",
        "central limit theorem",
        "correlation",
        "coefficient of correlation",

        "battery recycling",
        "lithium battery",
        "battery recycling process",
        "valuable metals",

        "esp32",
        "max30102",
        "heart rate",
        "pulse sensor",
        "ecg",
        "ppg",
        "sensor",
    ]

    OCR_VARIANTS = {
        "probability": [
            "probabil",
            "probabih",
            "probabii",
            "parole",
        ],

        "conditional probability": [
            "conditional pare",
            "conditional probab",
            "condihonal probability",
            "condihoncl probability",
            "condihional probability",
        ],

        "probability distribution": [
            "probability distribution",
            "probability distributions",
            "probabil distribution",
            "probability distribubion",
            "probabihity distribution",
        ],

        "exponential distribution": [
            "exponential distribution",
            "exponential distribubion",
        ],

        "marginal distribution": [
            "marginal distribution",
            "marginal distribubion",
            "marginal distribubien",
        ],

        "joint distribution": [
            "joint distribution",
            "jioim distribution",
            "jioim distribubion",
            "jioim distribubien",
        ],

        "defective bolts": [
            "defective bolt",
            "defective bolts",
            "defechive",
            "defeckive",
            "delechtive",
            "defechive bolts",
        ],

        "random variable": [
            "random variable",
            "random variables",
            "vandom",
        ],

        "correlation": [
            "correlation",
            "coefeyotens",
            "coefficient",
        ],
    }

    # =================================================
    # MAIN EXTRACTION
    # =================================================

    def extract(
        self,
        text,
        max_topics=20
    ):

        if not text:
            return []

        text = str(text).lower()

        topics = []

        # -------------------------------------------------
        # 1. Exact domain phrases
        # -------------------------------------------------

        for phrase in self.DOMAIN_PHRASES:

            if phrase in text:

                topics.append(
                    phrase
                )

        # -------------------------------------------------
        # 2. Known OCR variants
        # -------------------------------------------------

        for canonical, variants in self.OCR_VARIANTS.items():

            for variant in variants:

                if variant in text:

                    topics.append(
                        canonical
                    )

                    break

        # -------------------------------------------------
        # 3. Proximity matching
        #
        # This handles OCR where the two words of a concept
        # are damaged or separated.
        #
        # Example:
        #
        # probability ... distribubion
        #
        # -> probability distribution
        # -------------------------------------------------

        proximity_topics = self._extract_by_proximity(
            text
        )

        topics.extend(
            proximity_topics
        )

        # -------------------------------------------------
        # 4. Repeated useful words
        #
        # Fallback only. Known concepts always have priority.
        # -------------------------------------------------

        words = re.findall(
            r"[a-zA-Z]{5,}",
            text
        )

        words = [
            word
            for word in words
            if word not in self.STOPWORDS
        ]

        counts = Counter(
            words
        )

        garbage = {
            "balts",
            "disken",
            "distri",
            "distribubien",
            "distribubion",
            "jioim",
            "achin",
            "parole",
        }

        for word, count in counts.most_common():

            if count < 2:
                continue

            if len(word) < 5:
                continue

            if word in garbage:
                continue

            if word not in topics:

                topics.append(
                    word
                )

            if len(topics) >= max_topics:

                break

        return self._unique(
            topics
        )[:max_topics]

    # =================================================
    # PROXIMITY MATCHING
    # =================================================

    def _extract_by_proximity(
        self,
        text,
        max_distance=5
    ):

        topics = []

        # Tokenize while retaining approximate word positions.
        words = re.findall(
            r"[a-zA-Z]{4,}",
            text.lower()
        )

        if not words:
            return []

        # -------------------------------------------------
        # Canonical concept -> possible component words.
        #
        # These are semantic anchors, not OCR corrections.
        # -------------------------------------------------

        concepts = {
            "probability distribution": [
                (
                    ["probability", "probabil", "probabih"],
                    ["distribution", "distribubion", "distribubien"]
                )
            ],

            "conditional probability": [
                (
                    ["conditional", "condihonal", "condihoncl", "condihional"],
                    ["probability", "probabil", "probabih"]
                )
            ],

            "exponential distribution": [
                (
                    ["exponential"],
                    ["distribution", "distribubion", "distribubien"]
                )
            ],

            "marginal distribution": [
                (
                    ["marginal"],
                    ["distribution", "distribubion", "distribubien"]
                )
            ],

            "joint distribution": [
                (
                    ["joint", "jioim"],
                    ["distribution", "distribubion", "distribubien"]
                )
            ],

            "random variable": [
                (
                    ["random", "vandom"],
                    ["variable", "variables"]
                )
            ],

            "defective bolts": [
                (
                    ["defective", "defechive", "defeckive", "delechtive"],
                    ["bolt", "bolts", "balts"]
                )
            ],

            "coefficient of correlation": [
                (
                    ["coefficient", "coefeyotens"],
                    ["correlation", "correlation"]
                )
            ],
        }

        for canonical, patterns in concepts.items():

            for left_words, right_words in patterns:

                left_positions = []
                right_positions = []

                for index, word in enumerate(words):

                    if word in left_words:
                        left_positions.append(index)

                    if word in right_words:
                        right_positions.append(index)

                found = False

                for left in left_positions:

                    for right in right_positions:

                        if abs(left - right) <= max_distance:

                            topics.append(
                                canonical
                            )

                            found = True
                            break

                    if found:
                        break

                if found:
                    break

        return topics

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

            value = str(
                value
            ).strip()

            key = value.lower()

            if not value:
                continue

            if key in seen:
                continue

            seen.add(key)

            result.append(
                value
            )

        return result