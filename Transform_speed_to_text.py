import torch
import os
import whisper
import json
from pydub import AudioSegment

def transcribe_audio(audio_path,speaker_segments,model_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model(model_size)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    # Load and preprocess the audio
    final_transcriptions = []
    audio = AudioSegment.from_file(audio_path)
    max_chunk_duration = 30 * 1000  # 30 seconds in milliseconds
    for index, segment in enumerate(speaker_segments):

        start_ms = segment["start"] * 1000  # Convert to milliseconds
        end_ms = segment["end"] * 1000      # Convert to milliseconds
        check_duration = end_ms - start_ms
        full_text = ""
        if(check_duration > max_chunk_duration):
            curr_start = start_ms
            while(curr_start < end_ms):
                curr_end = min(curr_start + max_chunk_duration, end_ms)
                speaker_audio = audio[curr_start:curr_end]
                # Save the speaker's audio segment temporarily
                temp_audio_path = f"temp_speaker_{index}_{curr_start}.wav"
                speaker_audio.export(temp_audio_path, format="wav")
                result = model.transcribe(temp_audio_path)
                full_text += result["text"] + " "
                os.remove(temp_audio_path)
                curr_start = curr_end
        else:
            speaker_audio = audio[start_ms:end_ms]    
            temp_audio_path = f"temp_speaker_{index}.wav"
            speaker_audio.export(temp_audio_path, format="wav")
            result = model.transcribe(temp_audio_path)
            full_text = result["text"]
            os.remove(temp_audio_path)
            
        final_transcriptions.append({
            "speaker": segment["speaker"],
            "start": segment["start"],
            "end": segment["end"],
            "text": full_text.strip()
        })
    return final_transcriptions
