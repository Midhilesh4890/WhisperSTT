import sounddevice as sd
import whisper
import numpy as np


def record_audio_array(duration=5, samplerate=16000):
    print(f"🎙️  Recording for {duration} seconds... Speak now!")
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.\n")
    return np.squeeze(audio)  # Convert shape (N, 1) → (N,)


def transcribe_audio_array(audio_array, model_size="base"):
    print(f"📦 Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    # Pad or trim to match Whisper input expectations
    audio_array = whisper.pad_or_trim(audio_array)

    print("🧠 Transcribing...")
    result = model.transcribe(audio_array, fp16=False)
    print("✅ Transcription complete.\n")
    return result["text"]


if __name__ == "__main__":
    duration = 5  # You can customize this
    print("🎧 Press Enter to start recording...")
    input()  # Wait for user input

    audio = record_audio_array(duration=duration)
    text = transcribe_audio_array(audio, model_size="base")

    print("📝 Transcribed Text:\n", text)
