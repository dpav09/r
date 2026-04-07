import os
import sys

from qdrant_client import QdrantClient, models

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "contracts_chunks")
FRIDA_VECTOR_SIZE = int(os.getenv("FRIDA_VECTOR_SIZE", "1536"))


def main() -> None:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(QDRANT_COLLECTION):
        print(f"Collection '{QDRANT_COLLECTION}' already exists")
        sys.exit(0)

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "frida_dense": models.VectorParams(
                size=FRIDA_VECTOR_SIZE,
                distance=models.Distance.COSINE,
                on_disk=False,
            )
        },
        sparse_vectors_config={
            "bm25_sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
                index=models.SparseIndexParams(on_disk=True),
            )
        },
    )

    print(f"Collection '{QDRANT_COLLECTION}' created successfully")


if __name__ == "__main__":
    main()
