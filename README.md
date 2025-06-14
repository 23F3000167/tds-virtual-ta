# TDS Virtual TA

This is a FastAPI-based virtual teaching assistant for the IITM Data Science course. It uses Hybrid RAG over Discourse + Markdown content to answer questions with source citations.

## API Endpoint

- `POST /api/`  
- Body:
```json
{
  "question": "What is Hybrid RAG?"
}
