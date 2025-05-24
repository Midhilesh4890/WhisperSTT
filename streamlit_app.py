import soundfile as sf
import numpy as np
import tempfile
import sys
import os
import streamlit as st

# MUST be first Streamlit command
st.set_page_config(page_title="Whisper STT", layout="centered")


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
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode, RTCConfiguration, RTCIceServer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.warning("streamlit-webrtc not available. Using file upload mode only.")
# App title
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

# Initialize session state
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

# Choose mode
mode = st.radio("Choose input method:", [
                "File Upload", "Microphone (WebRTC)", "Simple Audio Recorder"])

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
    st.header("🎤 WebRTC Microphone")

    # Configure STUN servers for better connectivity
    rtc_configuration = RTCConfiguration({
        "iceServers": [
            RTCIceServer({"urls": ["stun:stun.l.google.com:19302"]}),
            RTCIceServer({"urls": ["stun:stun1.l.google.com:19302"]}),
            RTCIceServer({"urls": ["stun:stun2.l.google.com:19302"]}),
        ]
    })

    class AudioProcessor(AudioProcessorBase):
        def __init__(self) -> None:
            self.recorded_frames = []
            self.is_recording = False

        def recv_queued(self, frames):
            # Process multiple frames at once (more efficient)
            if self.is_recording:
                for frame in frames:
                    self.recorded_frames.append(frame.to_ndarray().flatten())
            return frames

        def start_recording(self):
            self.is_recording = True
            self.recorded_frames = []
            st.session_state.recording_active = True
            st.session_state.has_audio_data = False

        def stop_recording(self):
            self.is_recording = False
            st.session_state.recording_active = False
            if self.recorded_frames:
                st.session_state.audio_frames = self.recorded_frames.copy()
                st.session_state.has_audio_data = True
            return self.recorded_frames

    # Initialize session state
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    if 'has_audio_data' not in st.session_state:
        st.session_state.has_audio_data = False
    if 'audio_frames' not in st.session_state:
        st.session_state.audio_frames = []

    try:
        ctx = webrtc_streamer(
            key="whisper-stt-stun",
            mode=WebRtcMode.SENDONLY,
            media_stream_constraints={
                "audio": {
                    "sampleRate": 16000,  # Optimal for Whisper
                    "channelCount": 1,    # Mono audio
                    "echoCancellation": True,
                    "noiseSuppression": True,
                    "autoGainControl": True,
                },
                "video": False
            },
            audio_processor_factory=AudioProcessor,
            rtc_configuration=rtc_configuration,
            async_processing=True,
        )

        # Show detailed connection status
        if ctx.state.playing:
            st.success("🟢 Microphone connected and recording!")
        elif ctx.state.signalling:
            st.warning("🟡 Establishing connection... Please wait.")
        else:
            st.info("🔴 Click 'START' above to connect microphone")

        # Recording control buttons
        if ctx.audio_processor:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🎙️ Start Recording", disabled=st.session_state.recording_active):
                    ctx.audio_processor.start_recording()
                    st.rerun()

            with col2:
                if st.button("⏹️ Stop Recording", disabled=not st.session_state.recording_active):
                    ctx.audio_processor.stop_recording()
                    st.rerun()

            # Show current recording status
            if st.session_state.recording_active:
                st.warning(
                    "🎙️ **Recording in progress...** Speak now, then click 'Stop Recording' when done.")
                # Show a simple audio level indicator
                st.progress(50, text="🔴 Recording...")

            # Show transcribe button only when we have audio data
            if st.session_state.has_audio_data and not st.session_state.recording_active:
                st.success(
                    "✅ **Recording complete!** Click below to transcribe.")

                if st.button("📝 Transcribe Recorded Audio", type="primary"):
                    if st.session_state.audio_frames:
                        with st.spinner("🔁 Processing audio... Please wait."):
                            try:
                                # Combine all audio frames
                                audio_data = np.concatenate(
                                    st.session_state.audio_frames, axis=0)

                                # Save to temporary file
                                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                                    import soundfile as sf
                                    sf.write(tmp_file.name, audio_data, 16000)

                                    # Transcribe
                                    result = model.transcribe(
                                        tmp_file.name, fp16=False)

                                    # Display results
                                    st.success("✅ Transcription complete!")
                                    st.subheader("📝 Transcribed Text")
                                    st.text_area(
                                        "", result["text"], height=200, label_visibility="collapsed")

                                    # Additional info
                                    with st.expander("📊 Additional Information"):
                                        st.write(
                                            f"**Language detected:** {result.get('language', 'Unknown')}")
                                        if 'segments' in result:
                                            st.write(
                                                f"**Number of segments:** {len(result['segments'])}")
                                            duration = result['segments'][-1].get(
                                                'end', 0) if result['segments'] else 0
                                            st.write(
                                                f"**Duration:** {duration:.1f} seconds")

                                    # Reset for next recording
                                    st.session_state.has_audio_data = False
                                    st.session_state.audio_frames = []

                            except Exception as e:
                                st.error(
                                    f"❌ Error during transcription: {str(e)}")
                                st.info(
                                    "Try recording again or use the file upload method.")

                            finally:
                                try:
                                    os.unlink(tmp_file.name)
                                except:
                                    pass
                    else:
                        st.error("No audio data found. Please record again.")
                        st.session_state.has_audio_data = False

        else:
            st.info("Waiting for WebRTC connection... Click 'START' above first.")

    except Exception as e:
        st.error(f"WebRTC Configuration Error: {e}")
        st.info("Try refreshing the page or use the File Upload method instead.")

elif mode == "Simple Audio Recorder":
    st.header("🎙️ Browser Audio Recorder")
    st.info("This uses your browser's built-in audio recording capability.")

    # HTML5 Audio Recorder
    audio_recorder_html = """
    <div id="audio-recorder">
        <button id="startBtn" onclick="startRecording()">🎙️ Start Recording</button>
        <button id="stopBtn" onclick="stopRecording()" disabled>⏹️ Stop Recording</button>
        <audio id="audioPlayback" controls style="display:none; margin-top:10px;"></audio>
        <p id="status">Click 'Start Recording' to begin</p>
    </div>

    <script>
    let mediaRecorder;
    let recordedChunks = [];

    async function startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            
            mediaRecorder.ondataavailable = event => {
                if (event.data.size > 0) {
                    recordedChunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = () => {
                const blob = new Blob(recordedChunks, { type: 'audio/wav' });
                const audioURL = URL.createObjectURL(blob);
                document.getElementById('audioPlayback').src = audioURL;
                document.getElementById('audioPlayback').style.display = 'block';
                document.getElementById('status').innerText = 'Recording complete! Right-click the audio player and save the file.';
            };
            
            recordedChunks = [];
            mediaRecorder.start();
            
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('status').innerText = '🔴 Recording... Click Stop when done.';
            
        } catch (err) {
            document.getElementById('status').innerText = 'Error accessing microphone: ' + err.message;
        }
    }

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        document.getElementById('startBtn').disabled = false;
        document.getElementById('stopBtn').disabled = true;
    }
    </script>
    """

    st.components.v1.html(audio_recorder_html, height=200)
    st.markdown("""
    **Instructions:**
    1. Click 'Start Recording' and allow microphone access
    2. Speak clearly into your microphone
    3. Click 'Stop Recording' when done
    4. Right-click the audio player and select 'Save audio as...'
    5. Upload the saved file using the File Upload method above
    """)

else:
    st.error("WebRTC not available. Please use file upload method.")

# Instructions
with st.expander("📖 Troubleshooting WebRTC Issues"):
    st.markdown("""
    **If WebRTC connection fails:**
    
    1. **Try different browsers:** Chrome usually works best
    2. **Check firewall/antivirus:** May block WebRTC connections
    3. **Corporate networks:** Often block WebRTC traffic
    4. **Use File Upload instead:** Most reliable method
    5. **Try Simple Audio Recorder:** Uses browser's native recording
    
    **Network requirements for WebRTC:**
    - STUN servers must be accessible (stun.l.google.com:19302)
    - UDP traffic on various ports
    - Some corporate firewalls block this entirely
    """)

# System info
with st.expander("🔧 System Information"):
    st.write(f"**Whisper Available:** {'✅' if WHISPER_AVAILABLE else '❌'}")
    st.write(f"**WebRTC Available:** {'✅' if WEBRTC_AVAILABLE else '❌'}")
    st.write(f"**Model Loaded:** {'✅' if model else '❌'}")
