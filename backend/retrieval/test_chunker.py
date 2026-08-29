from backend.retrieval.chunker import chunk_text


text = """
Lithium-ion batteries are increasingly being recycled.
Battery recycling involves several processes.

Hydrometallurgy is one important recycling technique.
It uses chemical solutions to recover valuable metals.

Pyrometallurgy uses high temperatures to process
battery materials and recover metals.

Mechanical recycling involves shredding and sorting
battery components before further processing.
"""


chunks = chunk_text(
    text,
    chunk_size=20,
    overlap=5
)


for i, chunk in enumerate(chunks):

    print(f"\n--- CHUNK {i} ---")
    print(chunk)