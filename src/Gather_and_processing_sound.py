
import os
from pydub import AudioSegment
import sounddevice as sd
import wavio
import torch 
import torchaudio
import numpy as np
import subprocess, threading

#Transform the audio files to a specific format(WAV,16kHz,1 channel)
def standardi_audio(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_path, format="wav")
    return output_path

def record_meeting(output_path,device_index):
    fs=48000
    recording=[]
    print("Recording... Press Enter to stop.")
    def callback(indata, frames, time, status):
        if status:
            print(status)
        recording.append(indata.copy())

    with sd.InputStream(samplerate=fs, channels=1, device=device_index, callback=callback):
        input()
    print("Recording stopped.")
    audio_data=np.concatenate(recording, axis=0)
    wavio.write(output_path, audio_data, fs, sampwidth=2)
    return output_path

def record_system_audio(output_path: str = "system_audio.wav",
                        duration_hint: str = "Press Enter to stop") -> str:
    """
    Thu âm thanh TRỰC TIẾP từ hệ thống (Discord, YouTube, Zoom...)
    KHÔNG phụ thuộc vào loa hay tai nghe.
    Hoạt động bằng cách đọc PipeWire/PulseAudio Monitor.
    """
    # Tìm monitor device tự động
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True, text=True
    )

    monitor_name = None
    for line in result.stdout.splitlines():
        if ".monitor" in line:
            monitor_name = line.split()[1]
            break

    if not monitor_name:
        print("Don't find monitor device!")
        return None

    print(f"Monitor device : {monitor_name}")
    print(f"Recording... ({duration_hint})")

    # FFmpeg đọc trực tiếp từ PulseAudio monitor
    cmd = [
        "ffmpeg",
        "-f",  "pulse",           # input từ PulseAudio/PipeWire
        "-i",  monitor_name,      # monitor device
        "-ac", "1",               # mono
        "-ar", "16000",           # 16kHz (chuẩn Whisper)
        "-y",                     # overwrite
        output_path
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    input()   # Chờ người dùng nhấn Enter

    # Dừng ffmpeg
    try:
        proc.stdin.write(b"q\n")
        proc.stdin.flush()
    except:
        pass
    proc.wait()

    print(f"Saved at: '{output_path}'")
    return output_path



def record_both_sides(output_path: str = "output.wav",
                      mic_device_id: int = 18) -> str:
    """
    Ghi đồng thời 2 phía trong cuộc họp Discord/Zoom:
      - Giọng NGƯỜI KHÁC : PipeWire Monitor (âm thanh hệ thống)
      - Giọng BẠN        : Mic vật lý (device_id)
    Sau đó mix 2 luồng thành 1 file WAV duy nhất.

    Args:
        output_path   : file WAV output cuối cùng
        mic_device_id : index mic của bạn (mặc định 18 = pipewire)
    """
    mic_tmp    = "temp_mic_side.wav"
    system_tmp = "temp_system_side.wav"
    stop_event = threading.Event()

    # ── Tìm PipeWire Monitor device ──────────────────────────────
    result = subprocess.run(
        ["pactl", "list", "sources", "short"],
        capture_output=True, text=True
    )
    monitor = None
    for line in result.stdout.splitlines():
        if ".monitor" in line:
            monitor = line.split()[1]
            break

    if not monitor:
        print("Don't find monitor device!")
        monitor = None

    # ── Luồng 1: Thu giọng người khác (PipeWire Monitor) ─────────
    def record_system_thread():
        if not monitor:
            return
        proc = subprocess.Popen(
            ["ffmpeg", "-f", "pulse", "-i", monitor,
             "-ac", "1", "-ar", "16000", "-y", system_tmp],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        stop_event.wait()           # chờ tín hiệu dừng
        try:
            proc.stdin.write(b"q\n")
            proc.stdin.flush()
        except Exception:
            pass
        proc.wait()

    # ── Luồng 2: Thu giọng bạn (Mic) ─────────────────────────────
    def record_mic_thread():
        frames = []
        def callback(indata, f, t, status):
            frames.append(indata.copy())
        try:
            with sd.InputStream(samplerate=16000, channels=1,
                                device=mic_device_id, callback=callback):
                stop_event.wait()   # chờ tín hiệu dừng
            audio = np.concatenate(frames)
            wavio.write(mic_tmp, audio, 16000, sampwidth=2)
        except Exception as e:
            print(f"Mic error: {e}")

    # ── Chạy 2 luồng song song ────────────────────────────────────
    print("Press Enter to stop.")
    t1 = threading.Thread(target=record_system_thread, daemon=True)
    t2 = threading.Thread(target=record_mic_thread,    daemon=True)
    t1.start()
    t2.start()

    input()             # ← nhấn Enter để dừng
    stop_event.set()    # ← báo cả 2 luồng dừng lại

    t1.join(timeout=5)
    t2.join(timeout=5)
    print("Stop!")

    # ── Mix 2 file thành 1 ───────────────────────────────────────
    has_system = monitor and os.path.exists(system_tmp)
    has_mic    = os.path.exists(mic_tmp)

    if has_system and has_mic:
        system_audio = AudioSegment.from_wav(system_tmp)
        mic_audio    = AudioSegment.from_wav(mic_tmp)
        combined     = system_audio.overlay(mic_audio)
        combined.export(output_path, format="wav")
    elif has_system:
        os.rename(system_tmp, output_path)
    elif has_mic:
        os.rename(mic_tmp, output_path)
    else:
        return None

    # Dọn file tạm
    for f in [mic_tmp, system_tmp]:
        if os.path.exists(f):
            os.remove(f)

    print(f"Save at: '{output_path}'")
    return output_path


#remove_silence
def remove_silence(audio_path,output_path):
    model,utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                 model='silero_vad', 
                                 force_reload=False)
    get_speech_timestamps, save_audio, read_audio, _, _ = utils

    wav=read_audio(audio_path, sampling_rate=16000)
    speech_timestamps=get_speech_timestamps(wav,
                                            model,
                                            sampling_rate=16000)
    if len(speech_timestamps)==0:
        return None
    
    chunks=[]
    for ts in speech_timestamps:
        start=ts['start']
        end=ts['end']

        audio=wav[start:end]
        chunks.append(audio)
    speech_audio=torch.cat(chunks)
    save_audio(output_path, speech_audio, sampling_rate=16000)
    return output_path



def process_audio(input_source, output_final_path,
                  is_live=False, device_id=None,
                  use_system_audio=False,
                  both_sides=False):
    """
    Args:
        input_source      : đường dẫn file (nếu is_live=False)
        output_final_path : file output cuối cùng
        is_live           : True = ghi âm trực tiếp
        device_id         : index mic (khi is_live=True, both_sides=False, use_system_audio=False)
        use_system_audio  : True = chỉ thu âm hệ thống (giọng người khác)
        both_sides        : True = thu cả 2 (giọng bạn + giọng người khác) ← DISCORD
    """
    temp_standard_wav = "temp_standard.wav"
    temp_recorded_wav = "temp_recorded.wav"

    if is_live:
        if both_sides:
            record_both_sides(
                output_path=temp_standard_wav,
                mic_device_id=device_id if device_id is not None else 18
            )
        elif use_system_audio:
            # Thu chỉ âm hệ thống (giọng người khác, YouTube...)
            record_system_audio(temp_standard_wav)
        else:
            # Thu mic vật lý (họp offline trong phòng)
            record_meeting(temp_recorded_wav, device_id)
            standardi_audio(temp_recorded_wav, temp_standard_wav)
            os.remove(temp_recorded_wav)
    else:
        standardi_audio(input_source, temp_standard_wav)

    result_path = remove_silence(temp_standard_wav, output_final_path)

    if os.path.exists(temp_standard_wav):
        os.remove(temp_standard_wav)

    return result_path