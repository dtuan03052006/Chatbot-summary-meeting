"""
Module nạp dữ liệu cuộc họp vào Qdrant Vector Database
------------------------------------------------------
Không dùng class, chỉ dùng các hàm chức năng.
Đọc 2 file JSON (formatted_transcript.json & meeting_summary.json),
tạo vector embedding và lưu vào Qdrant.
"""

from importlib import metadata
import os
import json
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Cấu hình mặc định
DEFAULT_DB_PATH     = "./qdrant_db"
COLLECTION_NAME     = "meeting_knowledge"
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# Biến cache toàn cục tránh load lại model nhiều lần
_ENCODER = None


def get_embedding_model():
    """Tải hoặc lấy lại model embedding đa ngôn ngữ từ cache"""
    global _ENCODER
    if _ENCODER is None:
        print(" Đang tải mô hình Embedding đa ngôn ngữ...")
        _ENCODER = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _ENCODER


def get_qdrant_client(db_path: str = DEFAULT_DB_PATH) -> QdrantClient:
    """Khởi tạo hoặc kết nối Qdrant client chế độ embedded local"""
    return QdrantClient(path=db_path)

def format_segment(seg):
    start = seg.get("start","");
    speaker = seg.get("speaker","UNKNOWN");
    trans = seg.get("text_translated","").strip();
    orgin = seg.get("text_original","").strip();
    return f"[{start} {speaker}: {trans}, Gốc{orgin}]"
def prepare_documents(summary_path: str, transcript_path: str) -> list[dict]:
    docs = []

    # 1. Bóc tách file meeting_summary.json
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            sum_data = json.load(f)

            if overall := sum_data.get("overall_summary", "").strip():
                docs.append(
                    Document(
                        page_content=f"TỔNG QUAN VÀ KẾT LUẬN CUỘC HỌP:\n{overall}",
                        metadata={"type": "overall_summary", "speaker": "ALL"}
                    )
                )

        per_speaker = sum_data.get("per_speaker", {})
        for sp, sp_summary in per_speaker.items():
            if text := str(sp_summary).strip():
                docs.append(
                        Document(
                            page_content=f"TÓM TẮT Ý KIẾN CỦA {sp}:\n{text}",
                            metadata={"type": "speaker_summary", "speaker": sp}
                        )
                )

    # 2. Bóc tách file formatted_transcript.json
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            trans_data = json.load(f)

        documents = []

        if(trans_data):
            formatted_lines = [format_segment(seg) for seg in trans_data]
            full_transcripts_text = "\n".join(formatted_lines)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " "]
            )
            raw_chunks = splitter.split_text(full_transcripts_text)
            for chunk in raw_chunks:
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={"type" : "transcript"}
                    )
                )
    return docs


def index_meeting_to_qdrant(
    summary_path: str = "meeting_summary.json",
    transcript_path: str = "formatted_transcript.json",
    db_path: str = DEFAULT_DB_PATH
) -> int:
    """
    Hàm chính: Tự động nạp 2 file JSON vào Qdrant Vector DB.
    Trả về số lượng chunks đã nạp.
    """
    docs = prepare_documents(summary_path, transcript_path)
    if not docs:
        print("CẢNH BÁO: Không có dữ liệu để nạp vào Qdrant!")
        return 0

    encoder = get_embedding_model()
    dim = encoder.get_sentence_embedding_dimension()
    client = get_qdrant_client(db_path)

    # Chuyển text sang vector
    print(f" Đang mã hóa {len(docs)} chunks sang vector...")
    texts = [d["text"] for d in docs]
    vectors = encoder.encode(texts, show_progress_bar=False)

    # Đóng gói points
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v.tolist(),
            payload=d
        )
        for d, v in zip(docs, vectors)
    ]

    # Làm mới collection và lưu points
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Đã nạp tự động {len(points)} chunks vào Qdrant DB tại '{db_path}'!")
    return len(points)


if __name__ == "__main__":
    index_meeting_to_qdrant()

