from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from backend.retriever import get_answer
app = FastAPI()
from backend.utils import chunk_text
big_md_text = "This is a long markdown file content that needs to be chunked..." * 50
chunks = chunk_text(big_md_text)
print(chunks)


class QARequest(BaseModel):
    question: str
    image: Optional[str] = None  # Placeholder for future image OCR

@app.get("/")
def root():
    return {"message": "TDS Virtual TA API is running. Use POST /api/ to ask questions."}

@app.post("/api/")
async def answer_question(payload: QARequest):
    image_text = "OCR not implemented." if payload.image else ""
    full_query = payload.question + ("\n\n" + image_text if image_text else "")
    response = get_answer(full_query)
    return response  # must be a dict



