import base64
import textwrap
import re
from typing import List, Dict, Optional


# ─────────────────────────────────────────────────────────────
# 🧼 TEXT CLEANING / CHUNKING
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove excess whitespace and normalize markdown content."""
    return re.sub(r'\s+', ' ', text.strip())


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split a long text into overlapping chunks for embedding.
    Example: chunk_size=512, overlap=50 → chunks of 512 with 50-token overlap
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ─────────────────────────────────────────────────────────────
# 🖼️ IMAGE HANDLING (OCR PLACEHOLDER)
# ─────────────────────────────────────────────────────────────

def decode_base64_image(base64_str: str) -> bytes:
    """Decode a base64 image string to raw bytes."""
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        print("Error decoding image:", e)
        return b""


def ocr_image_placeholder(image_bytes: bytes) -> str:
    """Placeholder for OCR — add Tesseract or OpenAI Vision later."""
    return "[OCR not implemented yet]"

def extract_discourse_post_info(post: dict) -> Optional[dict]:
    try:
        content = post.get("content", "").strip()
        slug = post.get("topic_slug")
        topic_id = post.get("topic_id")
        post_num = post.get("post_number")
        url = f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{post_num}"
        short_text = content.split("\n")[0][:120] + "..."
        return {"url": url, "text": short_text, "content": content}
    except:
        return None




def generate_citation_links(posts: List[Dict]) -> List[Dict[str, str]]:
    """
    Given a list of post metadata (from extract_discourse_post_info),
    return a list of {"url", "text"} entries for citation in answer.
    """
    return [{"url": post["url"], "text": post["text"]} for post in posts if "url" in post and "text" in post]
