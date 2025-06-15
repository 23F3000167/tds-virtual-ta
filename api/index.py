from fastapi import FastAPI, Request
from backend.retriever import get_answer
from pydantic import BaseModel
from typing import Optional
from backend.main import app

app = FastAPI()

class QARequest(BaseModel):
    question: str
    image: Optional[str] = None

@app.post("/")
async def handle(request: Request):
    payload = await request.json()
    query = payload.get("question")
    image = payload.get("image")

    if not query:
        return {"error": "Missing 'question' in request body."}

    # You can use image later if OCR is implemented
    answer = get_answer(query)
    return answer
