import os
from pathlib import Path

from backend.ingestion.parser import parse_file
from backend.ingestion.image_parser import parse_image


SUPPORTED_DOCUMENTS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".xlsx",
}


SUPPORTED_IMAGES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# =========================================================
# VISION ANALYZER — LAZY SINGLETON
#
# Previously, ingest_file() called parse_image(path) with no
# analyzer arguments, so BLIP vision analysis silently never
# ran for any image going through the real pipeline (batch.py,
# the API, etc.) — it only ran if you manually instantiated
# VisionAnalyzer and called image_parser.py directly as a CLI
# script. This wires it into the actual pipeline.
#
# It's a lazy singleton (loaded on first image, then reused)
# rather than loaded eagerly for every call, since loading BLIP
# is expensive and most callers process many files per process
# lifetime.
#
# Set MEMORA_SKIP_VISION=1 to skip vision analysis entirely
# (OCR-only) — useful while iterating quickly on non-vision
# parts of the pipeline, since BLIP adds real time per image.
# =========================================================

_vision_analyzer = None


def _get_vision_analyzer():

    global _vision_analyzer

    if os.environ.get("MEMORA_SKIP_VISION") == "1":
        return None

    if _vision_analyzer is None:

        from backend.ingestion.vision import VisionAnalyzer

        print(
            "[MEMORA] Loading vision analyzer "
            "(first image of this process)..."
        )

        _vision_analyzer = VisionAnalyzer()

    return _vision_analyzer


def ingest_file(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"File does not exist: {path}"
        )

    extension = path.suffix.lower()

    print(f"\n[MEMORA] Processing: {path.name}")

    # Documents / text / spreadsheets
    if extension in SUPPORTED_DOCUMENTS:

        memory = parse_file(path)

    # Images
    elif extension in SUPPORTED_IMAGES:

        memory = parse_image(
            path,
            vision_analyzer=_get_vision_analyzer()
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    print(
        f"[MEMORA] Successfully processed "
        f"{path.name}"
    )

    return memory


if __name__ == "__main__":

    file_path = input(
        "Enter a file path: "
    )

    try:

        memory = ingest_file(file_path)

        print("\n========== MEMORY ==========\n")

        print("ID:", memory.id)
        print("Source:", memory.source)
        print("Modality:", memory.modality)
        print("Metadata:", memory.metadata)

        print("\n---------- CONTENT ----------\n")

        print(memory.content[:3000])

        if memory.description:

            print(
                "\n---------- DESCRIPTION ----------\n"
            )

            print(memory.description)

        print("\n============================\n")

    except Exception as error:

        print(
            f"\n[MEMORA ERROR] {error}\n"
        )