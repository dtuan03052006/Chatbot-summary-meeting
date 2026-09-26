"""
Module tìm kiếm ngữ cảnh trong Qdrant và trả lời câu hỏi bằng Groq LLM
----------------------------------------------------------------------
Không dùng class, chỉ dùng các hàm chức năng.
"""

import os
import requests
from qdrant_client import QdrantClient
from Rag_indexing import get_embedding_model, COLLECTION_NAME, DEFAULT_DB_PATH

from dotenv import load_dotenv
load_dotenv()

# Cấu hình Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME   = "qwen/qwen3.8-27b"


def search_relevant_context(
    question: str,
    top_k: int = 4,
    db_path: str = DEFAULT_DB_PATH
) -> list[dict]:
    """Tìm các đoạn ngữ cảnh liên quan nhất trong Qdrant"""
    if not os.path.exists(db_path):
        print(f"Chưa tìm thấy cơ sở dữ liệu tại '{db_path}'. Hãy nạp dữ liệu trước!")
        return []

    client = QdrantClient(path=db_path)
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        return []

    encoder = get_embedding_model()
    query_vector = encoder.encode(question).tolist()

    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    ).points

    return [p.payload for p in search_result if p.payload]


def build_rag_prompt(question: str, contexts: list[dict]) -> str:
    """Ghép ngữ cảnh và câu hỏi thành prompt hoàn chỉnh cho AI"""
    context_parts = []
    for ctx in contexts:
        time_tag = ""
        if ctx.get("type") == "transcript":
            time_tag = f" (Thời gian: {ctx.get('start')} - {ctx.get('end')})"
        context_parts.append(f"--- ĐOẠN TRÍCH{time_tag} ---\n{ctx.get('text')}")

    context_str = "\n\n".join(context_parts)

    return f"""Bạn là trợ lý AI chuyên gia phân tích biên bản cuộc họp.
Hãy trả lời câu hỏi dưới đây DỰA HOÀN TOÀN vào các thông tin trong phần NGỮ CẢNH CUỘC HỌP.

YÊU CẦU:
1. Trả lời rõ ràng, chính xác, khách quan bằng Tiếng Việt.
2. Trích dẫn rõ người nói (SPEAKER_XX) và mốc thời gian (nếu có trong ngữ cảnh).
3. Nếu không có thông tin trong ngữ cảnh, hãy nói thật là "Biên bản cuộc họp không đề cập đến thông tin này", không tự suy đoán.

NGỮ CẢNH CUỘC HỌP:
{context_str}

CÂU HỎI: {question}
CÂU TRẢ LỜI:""" 


def ask_meeting(
    question: str,
    top_k: int = 4,
    db_path: str = DEFAULT_DB_PATH
) -> str:
    """
    Hàm chính: Nhận câu hỏi, tìm trong Qdrant và trả lời bằng Groq LLM.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if not api_key:
        return "l Lỗi: Chưa tìm thấy biến môi trường GROQ_API_KEY."

    contexts = search_relevant_context(question, top_k=top_k, db_path=db_path)
    if not contexts:
        return "Không tìm thấy thông tin phù hợp trong dữ liệu cuộc họp."

    prompt = build_rag_prompt(question, contexts)

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 800,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return f" Lỗi Groq API ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f" Lỗi kết nối LLM: {e}"


def interactive_chat(db_path: str = DEFAULT_DB_PATH):
    """Vòng lặp hỏi đáp liên tục qua Terminal / Console"""
    print("\n" + "="*50)
    print(" CHATBOT HỎI ĐÁP CUỘC HỌP (QDRANT + GROQ)")
    print("Gõ câu hỏi của bạn (hoặc 'exit' để thoát):")
    print("="*50)

    while True:
        try:
            q = input("\n Bạn hỏi: ").strip()
            if not q or q.lower() in ("exit", "quit", "q"):
                print("Tạm biệt!")
                break
            ans = ask_meeting(q, db_path=db_path)
            print(f"\n Trợ lý: {ans}")
        except KeyboardInterrupt:
            print("\nĐã dừng.")
            break


if __name__ == "__main__":
    interactive_chat()

