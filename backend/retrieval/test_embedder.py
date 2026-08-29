from backend.retrieval.embedder import Embedder


embedder = Embedder()


texts = [
    "lithium battery recycling",
    "recovering valuable metals from batteries",
    "football match tonight"
]


vectors = embedder.embed_many(texts)


for text, vector in zip(texts, vectors):

    print("\nTEXT:")
    print(text)

    print("\nVECTOR DIMENSIONS:")
    print(len(vector))

    print("\nFIRST 10 VALUES:")
    print(vector[:10])