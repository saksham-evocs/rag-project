import requests
from qdrant_client import QdrantClient

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    OLLAMA_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
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


def search_qdrant(query: str):
    client = QdrantClient(url=QDRANT_URL)

    query_vector = get_embedding(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
        with_payload=True,
    )

    results = response.points

    chunks = []

    for result in results:
        payload = result.payload

        chunks.append(
            {
                "score": result.score,
                "paper_name": payload.get("paper_name"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
            }
        )

    return chunks


def build_prompt(query: str, chunks: list):
    context_blocks = []

    for i, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"""
SOURCE {i}
Paper: {chunk["paper_name"]}
Page: {chunk["page_number"]}
Similarity Score: {chunk["score"]}

Text:
{chunk["text"]}
"""
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are a research assistant.

Answer the user's question using ONLY the provided research paper context.

Rules:
1. If the answer is not found in the context, say:
   "I could not find enough information in the provided papers."
2. Do not make up information.
3. Mention source paper names and page numbers in the answer.
4. Keep the answer clear and structured.

Research Context:
{context}

User Question:
{query}

Final Answer:
"""

    return prompt


def ask_ollama(prompt: str):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["response"]


def answer_question(query: str):
    chunks = search_qdrant(query)

    if not chunks:
        return {
            "answer": "No relevant chunks found.",
            "sources": [],
        }

    prompt = build_prompt(query, chunks)
    answer = ask_ollama(prompt)

    return {
        "answer": answer,
        "sources": chunks,
    }