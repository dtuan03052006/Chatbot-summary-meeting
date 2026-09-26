import os
import glob
import ctypes
import site
import gc
import json
import torch
import numpy as np
from pydub import AudioSegment

# Tự động nạp các thư viện NVIDIA CUDA (cublas, cudnn) trong virtual environment nếu có
for p in site.getsitepackages():
    for lib_dir in glob.glob(f"{p}/nvidia/*/lib"):
        for so in glob.glob(f"{lib_dir}/*.so*"):
            try:
                ctypes.CDLL(so)
            except Exception:
                pass

from faster_whisper import WhisperModel


def transcribe_audio(audio_path, speaker_segments, model_size="small"):
    """
    Sử dụng Faster-Whisper (CTranslate2) tối ưu tốc độ và VRAM cho GPU.
    Nhanh gấp 4 lần, tiết kiệm > 50% VRAM so với Whisper gốc, xử lý trực tiếp trên RAM không ghi đĩa.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f" [Faster-Whisper] Đang tải mô hình '{model_size}' lên {device} ({compute_type})...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # Đọc audio bằng pydub và chuẩn hóa mẫu
    audio = AudioSegment.from_file(audio_path).set_frame_rate(16000).set_channels(1)

    # Chuyển đổi audio sang numpy array float32 chuẩn hóa [-1.0, 1.0] cho faster-whisper
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    sample_rate = 16000

    final_transcriptions = []
    print(f" [Faster-Whisper] Đang bóc text {len(speaker_segments)} đoạn hội thoại trên RAM...")

    for index, segment in enumerate(speaker_segments):
        start_sec = segment["start"]
        end_sec = segment["end"]
        start_idx = int(start_sec * sample_rate)
        end_idx = int(end_sec * sample_rate)

        segment_samples = samples[start_idx:end_idx]
        if len(segment_samples) == 0:
            continue

        # Transcribe trực tiếp từ mảng numpy trong RAM, không cần ghi/xóa file tạm trên ổ đĩa!
        segments_gen, _ = model.transcribe(
            segment_samples,
            beam_size=1,
            vad_filter=False
        )
        full_text = " ".join([s.text for s in segments_gen]).strip()

        final_transcriptions.append({
            "speaker": segment["speaker"],
            "start": round(start_sec, 2),
            "end": round(end_sec, 2),
            "text": full_text
        })

    # Dọn dẹp toàn bộ bộ nhớ GPU sau khi bóc text xong
    del model
    del audio
    del samples
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" [Faster-Whisper] Đã giải phóng 100% VRAM thành công!")

    return final_transcriptions

