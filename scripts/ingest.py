
from pathlib import Path
import hashlib

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

from backend.config import settings


DATA_DIR = Path("data/documents")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION = 384


def parse_document(path: Path):
    raw = path.read_text(encoding="utf-8")
    metadata = {}

    mapping = {
        "Title": "title",
        "Department": "department",
        "Document Type": "document_type",
        "Access Level": "access_level",
        "Created Date": "created_date",
    }

    for line in raw.splitlines():
        for prefix, key in mapping.items():
            if line.startswith(prefix + ":"):
                metadata[key] = line.split(":", 1)[1].strip()

    metadata["source"] = path.name
    return raw, metadata


def chunk_text(text: str, size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


def main():
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=settings.pinecone_api_key)

    if not pc.indexes.exists(settings.pinecone_index_name):
        print("Creating Pinecone index...")

        pc.indexes.create(
            name=settings.pinecone_index_name,
            schema={
                "fields": {
                    "embedding": {
                        "type": "dense_vector",
                        "dimension": DIMENSION,
                        "metric": "cosine",
                    }
                }
            },
            deployment={
                "deployment_type": "managed",
                "cloud": "aws",
                "region": "us-east-1",
            },
        )

    index = pc.index(settings.pinecone_index_name)
    model = SentenceTransformer(EMBEDDING_MODEL)

    documents = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        text, metadata = parse_document(path)

        for number, chunk in enumerate(chunk_text(text)):
            chunk_id = hashlib.sha256(
                f"{path.name}:{number}".encode()
            ).hexdigest()[:24]

            documents.append({
                "_id": chunk_id,
                "text": chunk,
                **metadata,
            })

    if not documents:
        raise ValueError("No documents found")

    embeddings = model.encode(
        [doc["text"] for doc in documents],
        normalize_embeddings=True,
    )

    for doc, embedding in zip(documents, embeddings):
        doc["embedding"] = embedding.tolist()

    # Small batches keep requests manageable.
    for start in range(0, len(documents), 20):
        batch = documents[start:start + 20]

        index.documents.upsert(
            namespace=settings.pinecone_namespace,
            documents=batch,
        )

    print(f"Indexed {len(documents)} chunks successfully")


if __name__ == "__main__":
    main()
