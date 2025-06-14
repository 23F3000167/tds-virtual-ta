from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import os
from pathlib import Path
import json
from backend.utils import extract_discourse_post_info

model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

corpus = []
corpus_embeddings = []
post_metadata = []  # store source info for links

from pathlib import Path
import json


def build_index():
    base_dir = Path("backend/data")

    # Load markdown files
    md_dir = base_dir / "tds_pages_md"
    for md_file in md_dir.glob("*.md"):
        corpus.append(md_file.read_text(encoding="utf-8").strip())

    # Load JSON files
    json_dir = base_dir / "discourse_json"
    for json_file in json_dir.glob("*.json"):
        posts = json.loads(json_file.read_text(encoding="utf-8"))
        for post in posts:
            if isinstance(post, dict) and "content" in post:
                corpus.append(post["content"].strip())

    
    max_docs = 200
    if len(corpus) > max_docs:
        print(f"[Warning] Trimming corpus to {max_docs} documents to reduce memory.")
        corpus[:] = corpus[:max_docs]  # in-place trimming

    print(f" Loaded {len(corpus)} documents. Building embeddings...")

    embeddings = model.encode(corpus, convert_to_tensor=True)
    corpus_embeddings.extend(embeddings)

    print(f" Index built with {len(corpus_embeddings)} vectors.")



    embeddings = model.encode(corpus, convert_to_tensor=True)
    corpus_embeddings.extend(embeddings)


def get_answer(query: str) -> dict:
    if not corpus_embeddings:
        build_index()

    query_embedding = model.encode(query, convert_to_tensor=True)
    results = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)[0]

    top_contexts = []
    top_links = []
    for r in results:
        idx = r["corpus_id"]
        text = corpus[idx]
        top_contexts.append(text)

        # Add citation if it's from discourse
        meta = post_metadata[idx] if idx < len(post_metadata) else None
        if meta and meta.get("url"):
            top_links.append({"url": meta["url"], "text": meta["text"]})

    context = "\n\n".join(top_contexts)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",  # or "gpt-4o-mini"
            messages=[
                {"role": "system", "content": "You are a helpful teaching assistant."},
                {"role": "user", "content": f"Based on the following:\n\n{context}\n\nAnswer this:\n{query}"}
            ]
        )
        return {
            "answer": response.choices[0].message.content.strip(),
            "links": top_links
        }
    except Exception as e:
        return {
            "answer": f"[OpenRouter Error] {str(e)}",
            "links": []
        }


