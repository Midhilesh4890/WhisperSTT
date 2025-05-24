import whisper


def transcribe_audio(file_path: str, model_size: str = "base") -> str:
    """
    Transcribes the given audio file using the specified Whisper model.

    Args:
        file_path (str): Path to the audio file.
        model_size (str): Size of the Whisper model. Options: "tiny", "base", "small", "medium", "large"

    Returns:
        str: Transcribed text from the audio.
    """
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)
    
    print(f"Transcribing audio file: {file_path}")
    result = model.transcribe(file_path)

    print("Transcription complete.\n")

    print("Detected language:", result["language"])

    return result['text']


if __name__ == "__main__":
    # Example usage
    audio_file = "spanish.wav"  # Replace with your audio file path
    transcription = transcribe_audio(audio_file, model_size="base")
    print("Transcribed Text:\n", transcription)
