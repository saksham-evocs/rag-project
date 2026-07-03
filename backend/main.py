from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer_question

app = FastAPI(title="Research Paper RAG API")


class QueryRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Research Paper RAG API is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    result = answer_question(request.question)
    return result