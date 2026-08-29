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

        memory = parse_image(path)

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