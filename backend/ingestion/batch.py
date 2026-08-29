from pathlib import Path

from backend.ingestion.ingestor import ingest_file


def ingest_folder(folder_path):

    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {folder}"
        )

    memories = []

    files = [
        file for file in folder.rglob("*")
        if file.is_file()
    ]

    print(
        f"\n[MEMORA] Found {len(files)} files.\n"
    )

    for file in files:

        try:

            memory = ingest_file(file)

            memories.append(memory)

        except Exception as error:

            print(
                f"[MEMORA] Skipping {file.name}: {error}"
            )

    print(
        f"\n[MEMORA] Successfully processed "
        f"{len(memories)}/{len(files)} files."
    )

    return memories


if __name__ == "__main__":

    folder_path = input(
        "Enter folder path: "
    )

    memories = ingest_folder(folder_path)

    print("\n========== SUMMARY ==========\n")

    for memory in memories:

        print(
            f"{memory.modality:12} | "
            f"{memory.source}"
        )

    print("\n=============================\n")