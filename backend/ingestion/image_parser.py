from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

from backend.memory.memory_item import MemoryItem
from backend.ingestion.vision import VisionAnalyzer
from backend.memory.semantic_analyzer import SemanticAnalyzer


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# OCR
# =========================================================

def _ocr_image(image):
    """
    Run Tesseract with a couple of lightweight configurations.

    Different preprocessing/configurations can recover different
    parts of noisy documents, especially photographed paper.
    """

    results = []

    configs = [
        "--psm 6",
        "--psm 11"
    ]

    for config in configs:

        try:

            text = pytesseract.image_to_string(
                image,
                config=config
            ).strip()

            if text:
                results.append(text)

        except Exception as error:

            print(
                "[MEMORA] OCR pass failed: "
                f"{error}"
            )

    return results


def _enhance_for_ocr(image):
    """
    Prepare a photographed document for OCR.

    This is intentionally lightweight so ingestion remains fast.
    """

    image = image.convert("L")

    # Upscale small handwriting.
    width, height = image.size

    if width < 1800:

        scale = 1800 / width

        image = image.resize(
            (
                int(width * scale),
                int(height * scale)
            ),
            Image.Resampling.LANCZOS
        )

    # Improve contrast.
    image = ImageEnhance.Contrast(
        image
    ).enhance(1.8)

    # Mild sharpening.
    image = image.filter(
        ImageFilter.SHARPEN
    )

    # Automatic contrast normalization.
    image = ImageOps.autocontrast(
        image
    )

    return image


def extract_image_text(path):

    image = Image.open(path)

    all_results = []

    # -----------------------------------------------------
    # Pass 1: original image
    # -----------------------------------------------------

    all_results.extend(
        _ocr_image(image)
    )

    # -----------------------------------------------------
    # Pass 2: enhanced image
    # -----------------------------------------------------

    enhanced = _enhance_for_ocr(
        image
    )

    all_results.extend(
        _ocr_image(enhanced)
    )

    # -----------------------------------------------------
    # Remove duplicate OCR output
    # -----------------------------------------------------

    unique_results = []

    seen = set()

    for text in all_results:

        normalized = " ".join(
            text.lower().split()
        )

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(normalized)

        unique_results.append(
            text
        )

    # -----------------------------------------------------
    # Combine OCR passes.
    #
    # We keep the passes separate so the semantic layer can
    # see alternative interpretations rather than assuming
    # one OCR pass is correct.
    # -----------------------------------------------------

    return "\n\n--- OCR PASS ---\n\n".join(
        unique_results
    ).strip()


# =========================================================
# IMAGE PARSER
# =========================================================

def parse_image(
    path,
    vision_analyzer=None,
    semantic_analyzer=None
):

    path = Path(path)

    image = Image.open(path)

    # =====================================================
    # OCR
    # =====================================================

    print(
        "[MEMORA] Extracting OCR..."
    )

    text = extract_image_text(
        path
    )

    # =====================================================
    # VISION
    # =====================================================

    description = ""
    vision_entities = []

    if vision_analyzer is not None:

        print(
            "[MEMORA] Analyzing image..."
        )

        vision_result = (
            vision_analyzer.analyze(
                str(path)
            )
        )

        description = vision_result.get(
            "description",
            ""
        )

        vision_entities = (
            vision_result.get(
                "entities",
                []
            )
        )

    # =====================================================
    # SEMANTIC ANALYSIS
    # =====================================================

    summary = ""
    topics = []
    semantic_entities = []

    if semantic_analyzer is not None:

        print(
            "[MEMORA] Extracting semantic meaning..."
        )

        # OCR is the primary source of textual
        # information. Vision provides additional
        # contextual information.

        semantic_input = ""

        if text:

            semantic_input += (
                "OCR TEXT:\n"
                + text
                + "\n\n"
            )

        if description:

            semantic_input += (
                "VISUAL DESCRIPTION:\n"
                + description
            )

        if semantic_input.strip():

            semantic_result = (
                semantic_analyzer.analyze(
                    semantic_input
                )
            )

            summary = (
                semantic_result.get(
                    "summary",
                    ""
                )
            )

            topics = (
                semantic_result.get(
                    "topics",
                    []
                )
            )

            semantic_entities = (
                semantic_result.get(
                    "entities",
                    []
                )
            )

    # =====================================================
    # MERGE ENTITIES
    # =====================================================

    entities = []

    for entity in (
        vision_entities
        + semantic_entities
    ):

        if entity and entity not in entities:

            entities.append(
                entity
            )

    # =====================================================
    # METADATA
    # =====================================================

    metadata = {

        "width": image.width,

        "height": image.height,

        "format": image.format

    }

    # =====================================================
    # MEMORY
    # =====================================================

    memory = MemoryItem(

        id=path.stem,

        source=str(path),

        modality="image",

        content=text,

        description=(
            description
        ),

        entities=entities,

        metadata=metadata
    )

    # =====================================================
    # Attach semantic information
    # =====================================================

    memory.summary = summary

    memory.topics = topics

    return memory


# =========================================================
# COMMAND-LINE TEST
# =========================================================

if __name__ == "__main__":

    file_path = input(
        "Enter image path: "
    )

    print(
        "\n[MEMORA] Initializing vision..."
    )

    vision_analyzer = (
        VisionAnalyzer()
    )

    print(
        "\n[MEMORA] Initializing semantic analyzer..."
    )

    semantic_analyzer = (
        SemanticAnalyzer()
    )

    memory = parse_image(

        file_path,

        vision_analyzer,

        semantic_analyzer
    )

    print(
        "\n========== IMAGE MEMORY ==========\n"
    )

    print(
        "ID:",
        memory.id
    )

    print(
        "Source:",
        memory.source
    )

    print(
        "Modality:",
        memory.modality
    )

    print(
        "Metadata:",
        memory.metadata
    )

    print(
        "\n---------- OCR ----------\n"
    )

    print(
        memory.content
    )

    print(
        "\n---------- VISION ----------\n"
    )

    print(
        memory.description
    )

    print(
        "\n---------- SUMMARY ----------\n"
    )

    print(
        memory.summary
    )

    print(
        "\n---------- TOPICS ----------\n"
    )

    print(
        memory.topics
    )

    print(
        "\n---------- ENTITIES ----------\n"
    )

    print(
        memory.entities
    )

    print(
        "\n=================================\n"
    )