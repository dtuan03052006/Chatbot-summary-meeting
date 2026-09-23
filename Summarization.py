import re
import json, os, time, requests
from typing import List, Dict

from torch import chunk

from Speaker_Diarization import speaker_diarization

GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
GROQ_URL         = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME       = "openai/gpt-oss-120b"
TARGET_LANGUAGE  = "Tiếng Việt"
CHUNK_WORD_LIMIT = 500          # số từ mỗi chunk MAP
INPUT_JSON       = "formatted_transcript.json"
OUTPUT_TXT       = "meeting_summary.txt"
OUTPUT_JSON      = "meeting_summary.json"

def load_formatted_script(json_path):
    with open(json_path,"r",encoding="utf-8") as f:
        data=json.load(f)
    return data

def split_into_chunks(segments, word_limit=CHUNK_WORD_LIMIT):
    chunks = []
    curr_chunk = []
    curr_wc = 0
    for seg in segments:
        text = seg.get("text_translated", seg.get("text", ""))
        wc = len(text.split())
        if curr_wc + wc > word_limit and curr_chunk:
            chunks.append(curr_chunk)
            curr_wc = 0
            curr_chunk = []
        curr_wc += wc
        curr_chunk.append(seg)
    
    if curr_chunk:
        chunks.append(curr_chunk)
    return chunks

def chunk_to_text(chunk: List[Dict]) -> str:
    """
    Chuyển list segment thành chuỗi dễ đọc cho LLM:
    [HH:MM:SS] SPEAKER_XX: <text>
    """
    lines = []
    for seg in chunk:
        start   = seg.get("start", "??:??")
        speaker = seg.get("speaker", "UNKNOWN")
        text    = seg.get("text_translated", seg.get("text", "")).strip()
        if text:
            if isinstance(start, (int, float)):
                m, s = int(start) // 60, int(start) % 60
                ts = f"{m:02d}:{s:02d}"
            else:
                ts = str(start)
            lines.append(f"[{ts}] {speaker}: {text}")
    return "\n".join(lines)

def call_llm(prompt, timeout=600):
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if not api_key:
        raise RuntimeError(
            "❌ Chưa tìm thấy GROQ_API_KEY!\n"
            "   Nguyên nhân: Giá trị key lấy từ UserSecretsClient đang rỗng hoặc bạn chưa bật secret trong Notebook."
        )

    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

def map_summarize_chunk(chunk_text: str,
                        chunk_idx: int,
                        total: int) -> str:

    prompt = f"""You are a professional meeting assistant.
Summarize the following meeting transcript excerpt in {TARGET_LANGUAGE}.
INSTRUCTIONS:
1. Write a concise summary of the main points discussed.
2. For each speaker (SPEAKER_XX), list their key points in 1-3 bullet points.
3. List any decisions or action items mentioned.
4. Write in {TARGET_LANGUAGE}. Be concise.
TRANSCRIPT EXCERPT (part {chunk_idx} of {total}):
{chunk_text}
SUMMARY IN {TARGET_LANGUAGE}:"""
    return call_llm(prompt, timeout=600)

def reduce_summaries(chunk_summaries: List[str]) -> str:
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    combined = "\n\n---\n\n".join(
        [f"[PHẦN {i+1}]\n{s}" for i, s in enumerate(chunk_summaries)]
    )
    prompt = f"""You are a professional meeting minutes writer.
Below are summaries of different parts of a meeting. 
Create ONE comprehensive meeting summary in {TARGET_LANGUAGE}.
PARTIAL SUMMARIES:
{combined}
Create the final meeting summary with these EXACT sections in {TARGET_LANGUAGE}:
## I. TỔNG QUAN CUỘC HỌP
(2-4 câu tóm tắt toàn bộ cuộc họp)
## II. NỘI DUNG THEO TỪNG NGƯỜI NÓI
(Liệt kê ý chính của từng SPEAKER_XX theo định dạng bullet points)
## III. QUYẾT ĐỊNH VÀ HÀNH ĐỘNG CẦN THỰC HIỆN
(Action items – ai làm gì, deadline nếu có)
## IV. KẾT LUẬN
(1-2 câu kết luận)
FINAL SUMMARY IN {TARGET_LANGUAGE}:"""

    return call_llm(prompt, timeout=600)

def summarize_per_speaker(segments):
    speaker_text = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text_translated", seg.get("text", "")).strip()
        if text:
            if speaker not in speaker_text:
                speaker_text[speaker] = []
            speaker_text[speaker].append(text)
            
    summaries = {}
    speakers = sorted(speaker_text.keys())
    for sp in speakers:
        all_text = " ".join(speaker_text[sp])
        prompt = f"""Summarize what {sp} said during the meeting in {TARGET_LANGUAGE}.
Their statements:
{all_text[:3000]}  
Write 3-5 bullet points summarizing their main contributions in {TARGET_LANGUAGE}:"""
        summaries[sp] = call_llm(prompt, timeout=600)
        time.sleep(0.5)
    return summaries

    

def summarize_meeting(
    input_json:   str = INPUT_JSON,
    output_txt:   str = OUTPUT_TXT,
    output_json:  str = OUTPUT_JSON,
    target_lang:  str = TARGET_LANGUAGE,
) -> Dict:
    global TARGET_LANGUAGE
    TARGET_LANGUAGE = target_lang

    segments = load_formatted_script(input_json)
    chunks = split_into_chunks(segments)
    chunks_summaries = []
    for idx, chunk in enumerate(chunks, 1):
        text = chunk_to_text(chunk)
        summary = map_summarize_chunk(text, idx, len(chunks))
        chunks_summaries.append(summary)
        time.sleep(1)

    overall_summary = reduce_summaries(chunks_summaries)
    speaker_summary = summarize_per_speaker(segments)

    result = {
        "overall_summary":  overall_summary,
        "per_speaker":       speaker_summary,
        "chunk_summaries":   chunks_summaries,
        "total_segments":    len(segments),
        "total_chunks":      len(chunks),
        "speakers":          sorted(set(s["speaker"] for s in segments)),
    }

    # 1. Lưu file TXT (xuống dòng rõ ràng, dễ đọc cho con người)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  BIÊN BẢN TÓM TẮT CUỘC HỌP\n")
        f.write("=" * 60 + "\n\n")
        f.write(overall_summary.replace("\\n", "\n"))
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("  TÓM TẮT THEO TỪNG NGƯỜI NÓI\n")
        f.write("=" * 60 + "\n\n")
        for sp, s in speaker_summary.items():
            f.write(f"### {sp}\n{s.replace('\\n', '\n')}\n\n")

    # 2. Lưu file JSON (dữ liệu máy đọc cho các bước tiếp theo)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"Saved file json and text summary: '{output_json}'")

    return result
