from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class VisionAnalyzer:

    def __init__(
        self,
        model_name="microsoft/git-base"
    ):
        print(
            f"[MEMORA] Loading vision model: {model_name}"
        )

        self.model_name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        self.model.to(self.device)

        print(
            f"[MEMORA] Vision model ready on "
            f"{self.device}."
        )

    def analyze(
        self,
        image_path: str,
        prompt: str = (
            "Describe this image in detail. "
            "Focus on information that would help "
            "someone remember and find this image later."
        )
    ) -> dict:

        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {path}"
            )

        image = Image.open(path).convert("RGB")

        # ------------------------------------------------
        # Current GIT model
        #
        # GIT is an image-captioning model and does not
        # currently use the prompt. The parameter exists
        # so we can replace GIT with a proper VLM later
        # without changing the rest of MEMORA.
        # ------------------------------------------------

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=80
            )

        description = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return {
            "description": description,
            "entities": [],
            "text": "",
            "source": str(path),
            "model": self.model_name
        }


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
        "\nEntities:",
        result["entities"]
    )

    print(
        "\nModel:",
        result["model"]
    )

    print(
        "\n====================================\n"
    )