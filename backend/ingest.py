import os
import uuid
import requests
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    OLLAMA_URL,
    EMBEDDING_MODEL,
    PDF_FOLDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def get_embedding(text: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def extract_pdf_text(pdf_path: str):
    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                }
            )

    return pages


def recreate_collection(client: QdrantClient, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        client.delete_collection(collection_name=COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )


def ingest_pdfs():
    client = QdrantClient(url=QDRANT_URL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    pdf_files = [
        file for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {PDF_FOLDER}")

    print(f"Found {len(pdf_files)} PDF files.")

    sample_embedding = get_embedding("test embedding")
    recreate_collection(client, vector_size=len(sample_embedding))

    points = []
    chunk_counter = 0

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        print(f"Processing: {pdf_file}")

        pages = extract_pdf_text(pdf_path)

        for page in pages:
            chunks = splitter.split_text(page["text"])

            for chunk_index, chunk_text in enumerate(chunks):
                embedding = get_embedding(chunk_text)

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "paper_name": pdf_file,
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                        "text": chunk_text,
                    },
                )

                points.append(point)
                chunk_counter += 1

                if len(points) >= 50:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points,
                    )
                    points = []

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    print(f"Ingestion complete. Total chunks stored: {chunk_counter}")


if __name__ == "__main__":
    ingest_pdfs()