r"""Quick local benchmark for MEMORA's deterministic reconstruction path.

Run from the repository root:
    .\venv\Scripts\python.exe -m backend.reconstruction.benchmark_deterministic
"""

from time import perf_counter

from backend.reconstruction.reconstructor import MemoryReconstructor


def build_memories(count):
    return [
        {
            "document": "",
            "retrieval_type": "primary",
            "metadata": {
                "memory_id": f"benchmark-{index}",
                "source": f"benchmark-{index}.txt",
                "summary": f"Probability study memory {index}.",
                "topics": ["probability", "statistics", f"topic {index}"],
            },
        }
        for index in range(count)
    ]


def main():
    # Synthesis is deterministic and does not need the Qwen model.
    reconstructor = MemoryReconstructor.__new__(MemoryReconstructor)

    for count in (10, 100, 500, 1000):
        memories = build_memories(count)
        started = perf_counter()
        answer = reconstructor._build_multi_memory_synthesis(memories)
        elapsed_ms = (perf_counter() - started) * 1000

        print(
            f"{count:4d} memories: {elapsed_ms:8.2f} ms "
            f"({len(answer)} characters)"
        )


if __name__ == "__main__":
    main()
