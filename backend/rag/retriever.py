
import asyncio
import hashlib
import re
from pathlib import Path

from pinecone import Pinecone, DenseVectorQuery
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from backend.config import settings
from scripts.ingest import parse_document, chunk_text


class HybridRetriever:
    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.pc = Pinecone(
            api_key=settings.pinecone_api_key
        )

        self.index = self.pc.index(
            settings.pinecone_index_name
        )

        self.local_docs = []

        for path in sorted(
            Path("data/documents").glob("*.txt")
        ):
            text, metadata = parse_document(path)

            for number, chunk in enumerate(
                chunk_text(text)
            ):
                chunk_id = hashlib.sha256(
                    f"{path.name}:{number}".encode()
                ).hexdigest()[:24]

                self.local_docs.append({
                    "id": chunk_id,
                    "text": chunk,
                    **metadata,
                })

        tokenized = [
            self.tokenize(doc["text"])
            for doc in self.local_docs
        ]

        self.bm25 = (
            BM25Okapi(tokenized) if tokenized else None
        )

    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    def search_sync(
        self,
        query: str,
        top_k: int = 4,
        department: str | None = None,
    ):
        query_vector = self.model.encode(
            query,
            normalize_embeddings=True,
        ).tolist()

        metadata_filter = (
            {"department": {"$eq": department}}
            if department else None
        )

        dense_results = self.index.documents.search(
            namespace=settings.pinecone_namespace,
            top_k=10,
            score_by=[
                DenseVectorQuery(
                    field="embedding",
                    values=query_vector,
                )
            ],
            include_fields=[
                "text",
                "title",
                "source",
                "department",
                "document_type",
                "access_level",
                "created_date",
            ],
            filter=metadata_filter,
        )

        if dense_results.matches:
            match = dense_results.matches[0]
            print("MATCH TYPE:", type(match))
            print("MATCH DATA:", match.to_dict())

        print("PINECONE MATCHES:", dense_results.matches[:1])
        
        dense_ids = []
        dense_docs = {}

        for match in dense_results.matches:
            # Convert Pinecone Document to a dictionary
            doc_data = match.to_dict()

            # Extract the document ID
            doc_id = doc_data.get("_id") or doc_data.get("id")

            if not doc_id:
                continue

            dense_ids.append(doc_id)

            # Pinecone may return document attributes
            # directly rather than under "fields"
            fields = doc_data.get("fields")

            if fields is None:
                fields = {
                    key: value
                    for key, value in doc_data.items()
                    if key not in {
                        "_id",
                        "id",
                        "_score",
                        "score",
                        "embedding",
                    }
                }

            dense_docs[doc_id] = {
                "id": doc_id,
                **fields,
            }


        # Local BM25 keyword search
        sparse_ids = []

        if self.bm25 is not None:
            scores = self.bm25.get_scores(
                self.tokenize(query)
            )

            ranked = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True,
            )

            for i in ranked:
                doc = self.local_docs[i]

                if department and (
                    doc.get("department") != department
                ):
                    continue

                if scores[i] <= 0:
                    continue

                sparse_ids.append(doc["id"])

                if len(sparse_ids) >= 10:
                    break

        # Reciprocal Rank Fusion
        rrf_scores = {}

        for ranking in [dense_ids, sparse_ids]:
            for rank, doc_id in enumerate(
                ranking, start=1
            ):
                rrf_scores[doc_id] = (
                    rrf_scores.get(doc_id, 0)
                    + 1 / (60 + rank)
                )

        local_lookup = {
            doc["id"]: doc
            for doc in self.local_docs
        }

        results = []

        for doc_id in sorted(
            rrf_scores,
            key=rrf_scores.get,
            reverse=True,
        )[:top_k]:
            doc = (
                dense_docs.get(doc_id)
                or local_lookup.get(doc_id)
            )

            if doc:
                results.append({
                    **doc,
                    "score": rrf_scores[doc_id],
                })

        return results

    async def search(
        self,
        query: str,
        top_k: int = 4,
        department: str | None = None,
    ):
        return await asyncio.to_thread(
            self.search_sync,
            query,
            top_k,
            department,
        )


retriever = HybridRetriever()
