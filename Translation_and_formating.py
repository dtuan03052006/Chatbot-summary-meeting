import json, os, time
from openai import OpenAI


OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "ollama")   # THAY ĐỔI 1/3
TARGET_LANGUAGE = "Tiếng Việt"
MODEL_NAME      = "gemma3:4b"                             # THAY ĐỔI 2/3
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
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"

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

    for i in range(1, max_retries + 1):
        try:
            # Dùng Ollama /api/generate thay vì chat.completions
            # → hoạt động với cả model "completion-only" (gemma3:4b)
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model":  model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 4096,
                    }
                },
                timeout=120
            )
            resp.raise_for_status()
            # Fix lỗi 2: dùng .get() → không bao giờ None → .strip() an toàn
            return resp.json().get("response", "").strip()
        except Exception as e:
            if i < max_retries:
                wait_time = 2 ** i
                print(f"Rate limit, waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"LLM call failed: {e}")
    return ""

def parse_translated_text_to_json(translated_text, original_segments):
    """Chuyển đổi văn bản đã dịch thành danh sách dict với định dạng chuẩn"""
    results=[]
    lines=[]
    idx=0
    for ln in translated_text.split("\n"):
        ln=ln.strip()
        if ln:
            lines.append(ln)
    for line in lines:
        if "]" in line and ":" in line:
            try:
                after=line.split("]",1)[1].strip()
                speaker, text=after.split(":",1)
                text=text.strip()
                if(idx < len(original_segments)):
                    src=original_segments[idx]
                    results.append({
                        "speaker" : src["speaker"],
                        "start" : format_time(src["start"]),
                        "end" : format_time(src["end"]),
                        "text_original" : src["text"],
                        "text_translated" : text
                    })
                    idx+=1
            except(ValueError, IndexError):
                continue
    return results
    

def translate_and_format_transcript(
    input_json: str = INPUT_JSON,
    output_txt: str = OUTPUT_TXT,
    output_json: str = OUTPUT_JSON,
    target_language: str = TARGET_LANGUAGE,
    model: str = MODEL_NAME,
    api_key: str = OPENAI_API_KEY,
) -> tuple[str, list[dict]]:
    
    segments = load_transcript(input_json)
    batches = split_into_batches(segments,word_limit=BATCH_WORD_LIMIT)

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key=api_key,
    )

    txt_parts, json_parts = [], []
    for i, batch in enumerate(batches, 1):
        
        raw = segments_to_text(batch)
        translated = translate_batch_with_llm(client, raw, target_language, model)
        txt_parts.append(translated)
        json_parts.extend(parse_translated_text_to_json(translated, batch))
        if i < len(batches):
            time.sleep(1)

    full_text = "\n\n".join(txt_parts)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_text)
   
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_parts, f, ensure_ascii=False, indent=4)
    return full_text, json_parts