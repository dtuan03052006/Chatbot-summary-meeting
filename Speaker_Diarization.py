import torch
import gc
from pyannote.audio import Pipeline

def speaker_diarization(audio_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" [Diarization] Đang tải mô hình lên {device}...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline.to(device)
    
    print(" [Diarization] Đang nhận diện người nói...")
    output = pipeline(audio_path, min_speakers=2, max_speakers=10)

    diarization = output.speaker_diarization
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2), 
        })

    # Giải phóng toàn bộ bộ nhớ GPU ngay lập tức để nhường chỗ cho Whisper
    del pipeline
    del output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" [Diarization] Đã giải phóng 100% VRAM thành công!")

    return segments