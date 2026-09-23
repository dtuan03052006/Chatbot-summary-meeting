import json, os, time, re
from openai import OpenAI


GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
TARGET_LANGUAGE = "Tiếng Việt"
MODEL_NAME      = "openai/gpt-oss-120b"
BATCH_WORD_LIMIT = 2000
INPUT_JSON      = "final_transcriptions.json"
OUTPUT_TXT      = "formatted_transcript.txt"
OUTPUT_JSON     = "formatted_transcript.json"


def load_transcript(input_file: str) -> list[dict]:
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def format_time(seconds):
    """Chuyển đổi giây sang định dạng HH:MM:SS"""
    if isinstance(seconds, str):
        return seconds
    try:
        s = float(seconds)
        hours = int(s // 3600)
        minutes = int((s % 3600) // 60)
        secs = int(s % 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"
    except Exception:
        return str(seconds)

def segments_to_text(segments):
    """Chuyển đổi danh sách segments thành văn bản định dạng"""
    formatted_text = []
    for segment in segments:
        start_time = format_time(segment["start"])
        end_time = format_time(segment["end"])
        speaker=segment["speaker"]
        text = segment["text"].strip()
        if(text):
            formatted_text.append(f"[{start_time} - {end_time}] {speaker}: {text}\n")
    return "\n".join(formatted_text)

def split_into_batches(segments, word_limit):
    """Chia segments thành các batch dựa trên giới hạn từ"""
    batches = []
    current_batch = []
    current_word_count = 0
    for seg in segments:
        word_count=len(seg["text"].split())
        if(current_word_count + word_count > word_limit and current_batch):
            batches.append(current_batch)
            current_batch=[]
            current_word_count=0
        current_batch.append(seg)
        current_word_count+=word_count
    if current_batch:
        batches.append(current_batch)
    return batches

def build_translation_prompt(
    raw_text: str,
    target_language: str
) -> str:
    return f"""### TRANSLATION TASK – MANDATORY ###

You MUST translate every line below into {target_language}.
This is NOT optional. You are FORBIDDEN from returning the original language.

INPUT FORMAT (do not change this structure):
[HH:MM:SS - HH:MM:SS] SPEAKER_XX: <text>

STRICT RULES – VIOLATION IS NOT ALLOWED:
1. TRANSLATE every word after the colon into {target_language}. No exceptions.
2. KEEP timestamps exactly as-is (e.g. [00:01:23 - 00:01:30]).
3. KEEP speaker labels exactly as-is (e.g. SPEAKER_00).
4. OUTPUT the same number of lines as the input. Never merge or split lines.
5. Do NOT add any explanation, notes, or extra text.
6. Do NOT repeat the original language in the output.
7. If a word is a proper noun or technical term, keep it but translate the rest.

### INPUT TRANSCRIPT (translate this NOW):
{raw_text}

### OUTPUT IN {target_language} (translated lines only, no extra text):"""


def translate_batch_with_llm(   
        client: OpenAI,
        raw_text: str,
        target_language: str,
        model: str = MODEL_NAME,
        max_retries: int = 3,) -> str:
    import requests
    prompt = build_translation_prompt(raw_text, target_language)

    # Lấy key trực tiếp tại thời điểm gọi hàm
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY

    if not api_key:
        raise RuntimeError(
            " Chưa tìm thấy GROQ_API_KEY!\n"
            "   Nguyên nhân: Giá trị key lấy từ UserSecretsClient đang rỗng hoặc bạn chưa bật secret trong Notebook."
        )

    for i in range(1, max_retries + 1):
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                },
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if i < max_retries:
                wait_time = 2 ** i
                print(f"Lỗi: {e}. Đang chờ {wait_time} giây để thử lại...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"LLM call failed: {e}")
    return ""

def parse_translated_text_to_json(translated_text, original_segments):
    """
    Chuyển đổi văn bản đã dịch thành danh sách dict.
    Tự động xử lý mọi định dạng từ LLM: có timestamp, không timestamp, code block, đánh số...
    Luôn đảm bảo kết quả có đầy đủ dữ liệu từ original_segments.
    """
    if not original_segments:
        return []

    results = []
    cleaned_lines = []
    
    # 1. Lọc bỏ các dòng markdown thừa, code blocks, lời chào
    skip_prefixes = ("```", "###", "here is", "dưới đây", "translation", "bản dịch", "output:")
    for ln in translated_text.split("\n"):
        ln_clean = ln.strip()
        if not ln_clean:
            continue
        if any(ln_clean.lower().startswith(p) for p in skip_prefixes):
            continue
        cleaned_lines.append(ln_clean)

    # 2. Bóc tách nội dung câu nói từ từng dòng
    parsed_texts = []
    for line in cleaned_lines:
        text_content = ""
        if "]" in line:
            after_bracket = line.split("]", 1)[1].strip()
            if ":" in after_bracket:
                text_content = after_bracket.split(":", 1)[1].strip()
            else:
                text_content = after_bracket
        elif ":" in line:
            text_content = line.split(":", 1)[1].strip()
        else:
            text_content = re.sub(r"^\d+[\.\)]\s*", "", line).strip()

        if text_content:
            parsed_texts.append(text_content)

    # 3. Ghép nối với original_segments
    for idx, src in enumerate(original_segments):
        if idx < len(parsed_texts):
            trans_text = parsed_texts[idx]
        else:
            trans_text = src.get("text", "")

        results.append({
            "speaker": src.get("speaker", "UNKNOWN"),
            "start": format_time(src.get("start", 0)),
            "end": format_time(src.get("end", 0)),
            "text_original": src.get("text", ""),
            "text_translated": trans_text
        })

    return results
    

def translate_and_format_transcript(
    input_json: str = INPUT_JSON,
    output_txt: str = OUTPUT_TXT,
    output_json: str = OUTPUT_JSON,
    target_language: str = TARGET_LANGUAGE,
    model: str = MODEL_NAME,
    api_key: str = GROQ_API_KEY,
) -> tuple[str, list[dict]]:
    
    segments = load_transcript(input_json)
    print(f" Đọc được {len(segments)} segments từ '{input_json}'")
    if not segments:
        print(f"CẢNH BÁO: File '{input_json}' không có đoạn thoại nào! Hãy kiểm tra bước 3 (transcribe_audio).")
        return "", []

    batches = split_into_batches(segments, word_limit=BATCH_WORD_LIMIT)
    print(f" Đã chia thành {len(batches)} batch để dịch...")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key if api_key else "gsk-placeholder",
    )

    txt_parts, json_parts = [], []
    for i, batch in enumerate(batches, 1):
        print(f"  → Đang dịch batch {i}/{len(batches)} ({len(batch)} câu)...")
        raw = segments_to_text(batch)
        translated = translate_batch_with_llm(client, raw, target_language, model)
        txt_parts.append(translated)
        parsed = parse_translated_text_to_json(translated, batch)
        json_parts.extend(parsed)
        if i < len(batches):
            time.sleep(1)

    full_text = "\n\n".join(txt_parts)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_text)
   
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_parts, f, ensure_ascii=False, indent=4)
        
    print(f" Đã lưu thành công {len(json_parts)} segments vào '{output_json}'")
    return full_text, json_parts