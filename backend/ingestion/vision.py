import torch

from PIL import Image
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)


class VisionAnalyzer:

    def __init__(
        self,
        model_name="Salesforce/blip-image-captioning-base"
    ):

        print(
            f"[MEMORA] Loading vision model: "
            f"{model_name}"
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = (
            BlipProcessor.from_pretrained(
                model_name
            )
        )

        self.model = (
            BlipForConditionalGeneration
            .from_pretrained(
                model_name
            )
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            f"[MEMORA] Vision model ready "
            f"on {self.device}."
        )

    # =====================================================
    # GENERATE VISUAL OBSERVATION
    # =====================================================

    def _ask(
        self,
        image,
        question
    ):

        inputs = self.processor(
            images=image,
            text=question,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=3
            )

        answer = self.processor.decode(
            output[0],
            skip_special_tokens=True
        )

        return answer.strip()

    # =====================================================
    # ANALYZE IMAGE
    # =====================================================

    def analyze(
        self,
        image_path
    ):

        image = Image.open(
            image_path
        ).convert("RGB")

        # -----------------------------------------
        # Focused visual observations
        # -----------------------------------------

        observations = []

        questions = [

            "a photograph of",

            "a photo of a",

            "a picture showing"
        ]

        for question in questions:

            try:

                result = self._ask(
                    image,
                    question
                )

                if result:

                    observations.append(
                        result
                    )

            except Exception as error:

                print(
                    "[MEMORA] Vision observation "
                    f"failed: {error}"
                )

        # -----------------------------------------
        # Remove duplicate observations
        # -----------------------------------------

        unique_observations = []

        for observation in observations:

            normalized = observation.lower()

            if normalized not in [
                item.lower()
                for item in unique_observations
            ]:

                unique_observations.append(
                    observation
                )

        # -----------------------------------------
        # Build visual description
        # -----------------------------------------

        if unique_observations:

            description = (
                "Visual observations: "
                + " | ".join(
                    unique_observations
                )
            )

        else:

            description = (
                "No reliable visual description "
                "was generated."
            )

        # -----------------------------------------
        # Image metadata
        # -----------------------------------------

        width, height = image.size

        metadata = {
            "width": width,
            "height": height,
            "format": image.format or "unknown"
        }

        return {
            "description": description,
            "entities": [],
            "topics": [],
            "text": "",
            "source": str(image_path),
            "metadata": metadata
        }


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    image_path = input(
        "Enter image path: "
    )

    analyzer = VisionAnalyzer()

    result = analyzer.analyze(
        image_path
    )

    print(
        "\n========== VISION RESULT ==========\n"
    )

    print(
        "Description:",
        result["description"]
    )

    print(
        "\nMetadata:",
        result["metadata"]
    )

    print(
        "\n===================================="
    )