from pathlib import Path

from PIL import Image
import pytesseract

from backend.memory.memory_item import MemoryItem
from backend.ingestion.vision import VisionAnalyzer


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_image_text(path):

    image = Image.open(path)

    text = pytesseract.image_to_string(image)

    return text


def parse_image(path, vision_analyzer=None):

    path = Path(path)

    image = Image.open(path)

    # -------------------------
    # OCR
    # -------------------------

    text = extract_image_text(path)

    # -------------------------
    # Vision
    # -------------------------

    description = ""
    entities = []

    if vision_analyzer is not None:

        vision_result = vision_analyzer.analyze(
            str(path)
        )

        description = vision_result.get(
            "description",
            ""
        )

        entities = vision_result.get(
            "entities",
            []
        )

    # -------------------------
    # Metadata
    # -------------------------

    metadata = {
        "width": image.width,
        "height": image.height,
        "format": image.format
    }

    # -------------------------
    # Memory
    # -------------------------

    return MemoryItem(
        id=path.stem,
        source=str(path),
        modality="image",
        content=text,
        description=description,
        entities=entities,
        metadata=metadata
    )


if __name__ == "__main__":

    file_path = input(
        "Enter image path: "
    )

    print(
        "\n[MEMORA] Initializing vision..."
    )

    analyzer = VisionAnalyzer()

    memory = parse_image(
        file_path,
        analyzer
    )

    print(
        "\n========== IMAGE MEMORY ==========\n"
    )

    print("ID:", memory.id)
    print("Source:", memory.source)
    print("Modality:", memory.modality)
    print("Metadata:", memory.metadata)

    print(
        "\n---------- OCR ----------\n"
    )

    print(memory.content)

    print(
        "\n---------- VISION ----------\n"
    )

    print(memory.description)

    print(
        "\n---------- ENTITIES ----------\n"
    )

    print(memory.entities)

    print(
        "\n=================================\n"
    )