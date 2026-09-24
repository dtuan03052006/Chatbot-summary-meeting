from Gather_and_processing_sound import process_audio
from Speaker_Diarization import speaker_diarization
from Transform_speed_to_text import transcribe_audio
from Translation_and_formating import translate_and_format_transcript
from Summarization import summarize_meeting
from Generate_pdf import export_summary_to_pdf, export_transcript_to_pdf
from Rag_indexing import index_meeting_to_qdrant
import os
import json
from pydub import AudioSegment



# ----- Bước 1: Thu âm / Xử lý âm thanh -----
audio_process = process_audio(
    input_source=None,
    output_final_path="output.wav",
    is_live=True,
    device_id=None,
    use_system_audio=True,
    both_sides=False,
)
print("Processed audio saved at:", audio_process)

# ----- Bước 2: Phân dạng người nói (Speaker Diarization) -----
diarization_result = speaker_diarization(audio_process)
print("Diarization result:", diarization_result)

unique_speakers = set()
for segment in diarization_result:
    unique_speakers.add(segment["speaker"])
print(f" Số người nói: {len(unique_speakers)}")

# ----- Bước 3: Chuyển giọng nói thành văn bản -----
transcriptions = transcribe_audio(audio_process, diarization_result, model_size="medium")
with open("final_transcriptions.json", "w", encoding="utf-8") as json_file:
    json.dump(transcriptions, json_file, ensure_ascii=False, indent=4)
print("Saved: final_transcriptions.json")

# ----- Bước 4: Dịch và định dạng transcript -----
text, data = translate_and_format_transcript()
print("Completed: formatted_transcript.json")

# ----- Bước 5: Tóm tắt cuộc họp (LLM) -----
summary_result = summarize_meeting(
    input_json="formatted_transcript.json",
    output_txt="meeting_summary.txt",
    output_json="meeting_summary.json",
)
print("Saved: meeting_summary.json")

# ----- Bước 6: Xuất file PDF từ file JSON -----
pdf_summary = export_summary_to_pdf(
    input_json="meeting_summary.json",       # ← file JSON đầu vào
    output_pdf="Meeting_Summary.pdf",        # ← file PDF đầu ra
    meeting_title="BIÊN BẢN CUỘC HỌP TỔNG HỢP",
)
pdf_transcript = export_transcript_to_pdf(
    input_json="formatted_transcript.json",       # ← file JSON đầu vào
    output_pdf="Transcript.pdf",                  # ← file PDF đầu ra
    meeting_title="BIÊN BẢN CUỘC HỌP CHIA TỪNG NGƯỜI NÓI",
)
print(f"Xuất PDF thành công: {pdf_summary,pdf_transcript}")

# ----- Bước 7: Tự động nạp 2 file JSON vào Qdrant Vector DB (RAG) -----
num_chunks = index_meeting_to_qdrant(
    summary_path="meeting_summary.json",
    transcript_path="formatted_transcript.json",
    db_path="./qdrant_db"
)
print(f"Bước 7 ✅ Đã tự động nạp {num_chunks} chunks vào Qdrant RAG DB!")

print("\n" + "=" * 50)
print(" 🎉 TOÀN BỘ PIPELINE & RAG ĐÃ HOÀN THÀNH!")
print("=" * 50)

