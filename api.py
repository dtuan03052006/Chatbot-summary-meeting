"""
FastAPI Server cho Trợ lý Cuộc họp AI (Meeting AI Assistant)
-----------------------------------------------------------
Đúng 5 chức năng cốt lõi:
1. POST /api/upload-audio        : Upload file âm thanh có sẵn -> xử lý trọn gói
2. POST /api/record/system-only  : Thu âm cuộc họp KHÔNG dùng micro (chỉ thu tiếng hệ thống/loa)
3. POST /api/record/with-mic     : Thu âm cuộc họp CÓ dùng micro (thu cả mic + loa hệ thống)
4. GET  /api/export-pdf/{type}   : Xuất và tải file PDF (tóm tắt hoặc chi tiết)
5. POST /api/chat                : Chatbot hỏi đáp thông minh về cuộc họp (RAG Qdrant + Groq)
"""

import os
import shutil
import json
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import các hàm chức năng trong dự án
from Gather_and_processing_sound import process_audio
from Speaker_Diarization import speaker_diarization
from Transform_speed_to_text import transcribe_audio
from Translation_and_formating import translate_and_format_transcript
from Summarization import summarize_meeting
from Generate_pdf import export_summary_to_pdf, export_transcript_to_pdf
from Rag_indexing import index_meeting_to_qdrant
from Rag_query import ask_meeting

# -------------------------------------------------
# Khởi tạo App & Thư mục
# -------------------------------------------------
app = FastAPI(
    title="Meeting AI Assistant API",
    description="Hệ thống 5 chức năng: Upload âm thanh, Thu âm không mic, Thu âm có mic, Xuất PDF và Chatbot hỏi đáp",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



def execute_meeting_pipeline(audio_wav_path: str) -> dict:
    """
    Chạy tự động các bước tiếp theo sau khi có file âm thanh:
    Phân tách người nói -> Bóc text Whisper -> Dịch -> Tóm tắt LLM -> Xuất 2 PDF -> Nạp vào Qdrant RAG
    """
    print("\n" + "="*50)
    print(" BẮT ĐẦU XỬ LÝ TOÀN BỘ CUỘC HỌP...")
    print("="*50)

    # 1. Phân dạng người nói (Diarization)
    print(" [1/5] Đang phân tách người nói...")
    diar_result = speaker_diarization(audio_wav_path)

    # 2. Bóc giọng nói thành chữ (Whisper)
    print(" [2/5] Đang chuyển giọng nói thành văn bản...")
    transcriptions = transcribe_audio(audio_wav_path, diar_result, model_size="medium")
    with open("final_transcriptions.json", "w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=4)

    # 3. Dịch và chuẩn hóa transcript
    print(" [3/5] Đang dịch sang Tiếng Việt...")
    translate_and_format_transcript(input_json="final_transcriptions.json", target_language="Tiếng Việt")

    # 4. Tóm tắt cuộc họp bằng LLM
    print(" [4/5] Đang tóm tắt cuộc họp bằng AI...")
    summary_result = summarize_meeting(input_json="formatted_transcript.json", target_lang="Tiếng Việt")

    # 5. Xuất 2 file PDF
    print(" [5/5] Đang xuất file PDF...")
    pdf_summary = export_summary_to_pdf(input_json="meeting_summary.json", output_pdf="Meeting_Summary.pdf")
    pdf_transcript = export_transcript_to_pdf(input_json="formatted_transcript.json", output_pdf="Transcript.pdf")

    # 6. Tự động nạp dữ liệu vào Qdrant Vector DB
    print(" [RAG] Đang nạp dữ liệu vào Qdrant...")
    chunks_indexed = index_meeting_to_qdrant(
        summary_path="meeting_summary.json",
        transcript_path="formatted_transcript.json"
    )

    print(" XỬ LÝ HOÀN TẤT!")
    return {
        "audio_path": audio_wav_path,
        "pdf_summary": pdf_summary,
        "pdf_transcript": pdf_transcript,
        "qdrant_chunks": chunks_indexed
    }



@app.post("/api/upload-audio", tags=["1. Upload file âm thanh"])
async def api_upload_audio(file: UploadFile = File(...)):
    """
    Nhận file âm thanh (mp3, wav, xZ  m4a...), tự động chuẩn hóa và chạy toàn bộ pipeline
    tạo ra bản tóm tắt, 2 file PDF và nạp vào RAG.
    """
    raw_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(raw_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Chuẩn hóa âm thanh về WAV 16kHz mono và lọc nhiễu
        processed_wav = process_audio(
            input_source=raw_path,
            output_final_path="output.wav",
            is_live=False
        )

        pipeline_result = execute_meeting_pipeline(processed_wav)
        return {
            "status": "success",
            "message": "Đã xử lý xong file âm thanh và sẵn sàng hỏi đáp!",
            "data": pipeline_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/record/system-only", tags=["2. Thu âm không dùng micro"])
def api_record_system_only():
    """
    Thu âm thanh TRỰC TIẾP từ hệ thống (Zoom, Google Meet, Teams, YouTube...).
    KHÔNG ghi lại tiếng từ micro của bạn.
    Sau khi thu âm xong, hệ thống tự động xử lý và nạp RAG.
    """
    try:
        print(" Bắt đầu thu âm hệ thống (không micro)...")
        processed_wav = process_audio(
            input_source=None,
            output_final_path="output.wav",
            is_live=True,
            use_system_audio=True,
            both_sides=False
        )

        pipeline_result = execute_meeting_pipeline(processed_wav)
        return {
            "status": "success",
            "message": "Đã thu âm hệ thống và xử lý hoàn tất!",
            "data": pipeline_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/record/with-mic", tags=["3. Thu âm có dùng micro"])
def api_record_with_mic(mic_device_id: Optional[int] = Query(None, description="ID microphone nếu có")):
    """
    Ghi âm 2 PHÍA đồng thời:
    - Tiếng của bạn nói qua Micro.
    - Tiếng của người khác trong cuộc họp qua Loa/Hệ thống.
    Sau khi thu âm xong, hệ thống tự động xử lý và nạp RAG.
    """
    try:
        print(" Bắt đầu thu âm 2 phía (Micro + Hệ thống)...")
        processed_wav = process_audio(
            input_source=None,
            output_final_path="output.wav",
            is_live=True,
            device_id=mic_device_id,
            use_system_audio=True,
            both_sides=True
        )

        pipeline_result = execute_meeting_pipeline(processed_wav)
        return {
            "status": "success",
            "message": "Đã thu âm 2 phía và xử lý hoàn tất!",
            "data": pipeline_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/export-pdf/{pdf_type}", tags=["4. Xuất file PDF"])
def api_export_pdf(pdf_type: str):
    """
    Xuất và tải file PDF về máy:
    - `pdf_type = 'summary'`   : Tải file 'Meeting_Summary.pdf' (Biên bản tóm tắt tổng thể)
    - `pdf_type = 'transcript'`: Tải file 'Transcript.pdf' (Biên bản chi tiết theo người nói)
    """
    if pdf_type == "summary":
        file_name = "Meeting_Summary.pdf"
        # Đảm bảo file tồn tại hoặc tạo mới
        if not os.path.exists(file_name) and os.path.exists("meeting_summary.json"):
            export_summary_to_pdf(input_json="meeting_summary.json", output_pdf=file_name)
    elif pdf_type == "transcript":
        file_name = "Transcript.pdf"
        if not os.path.exists(file_name) and os.path.exists("formatted_transcript.json"):
            export_transcript_to_pdf(input_json="formatted_transcript.json", output_pdf=file_name)
    else:
        raise HTTPException(status_code=400, detail="pdf_type không hợp lệ! Hãy chọn 'summary' hoặc 'transcript'.")

    if not os.path.exists(file_name):
        raise HTTPException(status_code=404, detail=f"Chưa có file '{file_name}'. Hãy chạy bước xử lý cuộc họp trước.")

    return FileResponse(file_name, media_type="application/pdf", filename=file_name)


class ChatQuery(BaseModel):
    question: str
    top_k: int = 4

@app.post("/api/chat", tags=["5. Chatbot hỏi đáp RAG"])
def api_chat(req: ChatQuery):
    """
    Hỏi đáp thông minh về nội dung cuộc họp:
    Truy vấn tri thức đã lưu trong Qdrant Vector DB và trả lời chính xác bằng Groq LLM.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống!")

    try:
        answer = ask_meeting(question=req.question, top_k=req.top_k)
        return {
            "status": "success",
            "question": req.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------
# Khởi động Server
# -------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

