import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from config import (
    QDRANT_URL,
    COLLECTION_NAME,
    OLLAMA_URL,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K_TEXT,
    TOP_K_FIGURES,
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


def search_qdrant_by_type(query: str, source_type: str, limit: int):
    client = QdrantClient(url=QDRANT_URL)
    query_vector = get_embedding(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value=source_type),
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )

    results = response.points
    chunks = []

    for result in results:
        payload = result.payload

        chunks.append(
            {
                "type": payload.get("type", "text"),
                "score": result.score,
                "paper_name": payload.get("paper_name"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "text": payload.get("text"),
                "caption": payload.get("caption"),
                "image_path": payload.get("image_path"),
                "figure_index": payload.get("figure_index"),
            }
        )

    return chunks


def search_qdrant(query: str):
    text_results = search_qdrant_by_type(
        query=query,
        source_type="text",
        limit=TOP_K_TEXT,
    )

    figure_results = search_qdrant_by_type(
        query=query,
        source_type="figure",
        limit=TOP_K_FIGURES,
    )

    return text_results + figure_results


def build_prompt(query: str, chunks: list):
    context_blocks = []

    for i, chunk in enumerate(chunks, start=1):
        if chunk["type"] == "figure":
            context_blocks.append(
                f"""
SOURCE {i}
Type: Figure
Paper: {chunk["paper_name"]}
Page: {chunk["page_number"]}
Figure Index: {chunk["figure_index"]}
Similarity Score: {chunk["score"]}

Caption:
{chunk["caption"]}
"""
            )
        else:
            context_blocks.append(
                f"""
SOURCE {i}
Type: Text
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
5. Be as detailed as possible.
6. If a figure source is relevant, mention it in the answer.

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