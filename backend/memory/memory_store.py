import json
from pathlib import Path
from dataclasses import asdict

from backend.memory.memory_object import MemoryObject


class MemoryStore:

    def __init__(
        self,
        storage_path="data/memories.json"
    ):

        self.storage_path = Path(
            storage_path
        )

        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.memories = {}

        self._load()

    def _load(self):

        if not self.storage_path.exists():
            return

        try:

            data = json.loads(
                self.storage_path.read_text(
                    encoding="utf-8"
                )
            )

            for item in data:

                memory = MemoryObject(
                    **item
                )

                self.memories[
                    memory.id
                ] = memory

        except Exception as e:

            print(
                f"[MEMORA] Could not load memory store: {e}"
            )

    def _save(self):

        data = [
            asdict(memory)
            for memory in self.memories.values()
        ]

        self.storage_path.write_text(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    def add(self, memory):

        self.memories[
            memory.id
        ] = memory

        self._save()

    def get(self, memory_id):

        return self.memories.get(
            memory_id
        )

    def all(self):

        return list(
            self.memories.values()
        )

    def add_relationship(
        self,
        memory_a,
        memory_b
    ):

        if memory_a in self.memories:

            if memory_b not in self.memories[
                memory_a
            ].related_memory_ids:

                self.memories[
                    memory_a
                ].related_memory_ids.append(
                    memory_b
                )

        if memory_b in self.memories:

            if memory_a not in self.memories[
                memory_b
            ].related_memory_ids:

                self.memories[
                    memory_b
                ].related_memory_ids.append(
                    memory_a
                )

        self._save()