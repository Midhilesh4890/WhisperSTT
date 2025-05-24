import soundfile as sf
import numpy as np
import streamlit as st
import os
import sys
import tempfile

# Fix PyTorch + Streamlit compatibility
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


# Import whisper with error handling
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError as e:
    st.error(f"Whisper not available: {e}")
    WHISPER_AVAILABLE = False

# Try to import webrtc with fallback
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("streamlit-webrtc not available. Using file upload mode only.")

# Streamlit UI setup
st.set_page_config(page_title="Whisper STT", layout="centered")
st.title("🎤 Whisper Speech-to-Text Demo")

if not WHISPER_AVAILABLE:
    st.error("Please install whisper: pip install openai-whisper")
    st.stop()

# Load Whisper model


@st.cache_resource
def load_whisper_model():
    try:
        return whisper.load_model("base")
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None


model = load_whisper_model()

if model is None:
    st.error("Failed to load Whisper model. Please check your installation.")
    st.stop()

# Initialize session state for audio data
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recording_complete' not in st.session_state:
    st.session_state.recording_complete = False

# Choose mode
mode = st.radio("Choose input method:", ["File Upload", "Microphone (WebRTC)"])

if mode == "File Upload":
    st.header("📁 Upload Audio File")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'mp4', 'm4a', 'flac', 'ogg']
    )

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button("📝 Transcribe Uploaded File"):
            with st.spinner("🔁 Processing audio..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    result = model.transcribe(tmp_file_path, fp16=False)
                    st.success("✅ Transcription complete!")
                    st.text_area("📝 Transcribed Text",
                                 result["text"], height=200)

                    with st.expander("📊 Additional Information"):
                        st.write(
                            f"**Language detected:** {result.get('language', 'Unknown')}")
                        if 'segments' in result:
                            st.write(
                                f"**Number of segments:** {len(result['segments'])}")

                except Exception as e:
                    st.error(f"Error during transcription: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_file_path)
                    except:
                        pass

elif mode == "Microphone (WebRTC)" and WEBRTC_AVAILABLE:
    st.header("🎤 Real-time Microphone")

    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.recorded_frames = []
            self.is_recording = False

        def recv_audio(self, frame):
            if self.is_recording:
                self.recorded_frames.append(frame.to_ndarray().flatten())
            return frame

        def start_recording(self):
            self.is_recording = True
            self.recorded_frames = []

        def stop_recording(self):
            self.is_recording = False
            return self.recorded_frames

    try:
        ctx = webrtc_streamer(
            key="whisper-stt",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=AudioProcessor,
            async_processing=True,
        )

        # Show connection status
        if ctx.state.playing:
            st.success("🟢 Microphone connected and recording!")
        elif ctx.state.signalling:
            st.info("🟡 Connecting to microphone...")
        else:
            st.info("🔴 Click START to connect microphone")

        # Control buttons and transcription
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎙️ Start Recording") and ctx.audio_processor:
                ctx.audio_processor.start_recording()
                st.session_state.recording_complete = False
                st.rerun()

        with col2:
            if st.button("⏹️ Stop Recording") and ctx.audio_processor:
                frames = ctx.audio_processor.stop_recording()
                if frames:
                    st.session_state.audio_data = np.concatenate(
                        frames, axis=0)
                    st.session_state.recording_complete = True
                st.rerun()

        # Show recording status
        if ctx.audio_processor and ctx.audio_processor.is_recording:
            st.warning(
                "🎙️ Recording in progress... Click 'Stop Recording' when done.")

        # Transcribe button appears after recording is complete
        if st.session_state.recording_complete and st.session_state.audio_data is not None:
            st.success("✅ Recording complete! Ready to transcribe.")

            if st.button("📝 Transcribe Recorded Audio", type="primary"):
                with st.spinner("🔁 Processing audio..."):
                    try:
                        # Save audio to temporary file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            sf.write(tmp_file.name,
                                     st.session_state.audio_data, 16000)

                            result = model.transcribe(
                                tmp_file.name, fp16=False)

                            st.success("✅ Transcription complete!")
                            st.text_area("📝 Transcribed Text",
                                         result["text"], height=200)

                            with st.expander("📊 Additional Information"):
                                st.write(
                                    f"**Language detected:** {result.get('language', 'Unknown')}")
                                if 'segments' in result:
                                    st.write(
                                        f"**Number of segments:** {len(result['segments'])}")

                            # Reset for next recording
                            st.session_state.audio_data = None
                            st.session_state.recording_complete = False

                    except Exception as e:
                        st.error(f"Error during transcription: {str(e)}")

                    finally:
                        try:
                            os.unlink(tmp_file.name)
                        except:
                            pass

    except Exception as e:
        st.error(f"WebRTC error: {e}")
        st.info("Please try refreshing the page or use the file upload method.")

else:
    st.error("WebRTC not available. Please use file upload method.")

# Instructions
with st.expander("📖 Instructions"):
    st.markdown("""
    ### Microphone Method:
    1. Click 'START' to connect your microphone
    2. Click '🎙️ Start Recording' to begin
    3. Speak clearly into your microphone
    4. Click '⏹️ Stop Recording' when done
    5. Click '📝 Transcribe' to convert speech to text
    
    ### File Upload Method:
    1. Record audio using your device's recorder
    2. Upload the audio file
    3. Click "Transcribe" to convert speech to text
    
    ### Tips:
    - Speak clearly and at normal volume
    - Minimize background noise
    - Use a quiet environment for best results
    """)

# System info
with st.expander("🔧 System Information"):
    st.write(f"**Whisper Available:** {'✅' if WHISPER_AVAILABLE else '❌'}")
    st.write(f"**WebRTC Available:** {'✅' if WEBRTC_AVAILABLE else '❌'}")
    st.write(f"**Model Loaded:** {'✅' if model else '❌'}")
    if 'ctx' in locals():
        st.write(f"**Microphone Status:** {ctx.state}")
