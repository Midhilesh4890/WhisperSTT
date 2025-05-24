import openai
import io
import wave
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
import logging
from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables
load_dotenv('.env')

# Get OpenAI API key from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up logging (only once)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app (only once)
app = FastAPI(title="OpenAI Whisper STT Server")

# CORS middleware (only once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Audio settings
SAMPLE_RATE = 16000

# Initialize OpenAI client
try:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    client = None


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket connected")

    if not client:
        await websocket.send_text("Error: OpenAI client not initialized. Check API key.")
        await websocket.close()
        return

    try:
        while True:
            try:
                # Receive binary audio data
                data = await websocket.receive_bytes()
                logger.info(f"📡 Received {len(data)} bytes of audio data")

                # Convert bytes to numpy array
                audio_data = np.frombuffer(data, dtype=np.float32)

                # Ensure we have enough audio data (at least 1 second)
                if len(audio_data) < SAMPLE_RATE:
                    logger.warning("⚠️ Audio chunk too short, skipping...")
                    continue

                # Convert to WAV format
                wav_buffer = create_wav_buffer(audio_data)

                # Transcribe with OpenAI
                transcription = await transcribe_with_openai(wav_buffer)

                if transcription and transcription.strip():
                    logger.info(f"✅ Transcription: {transcription}")
                    await websocket.send_text(transcription)
                else:
                    logger.info("🔇 No speech detected")

            except Exception as e:
                logger.error(f"❌ Error processing audio: {e}")
                await websocket.send_text(f"Error: {str(e)}")
                continue

    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")


def create_wav_buffer(audio_data):
    """Convert numpy array to WAV buffer"""
    wav_buffer = io.BytesIO()

    # Convert float32 to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer


async def transcribe_with_openai(audio_buffer):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        audio_buffer.name = "audio.wav"

        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_buffer,
            language="en",
            response_format="text"
        )

        return response

    except Exception as e:
        logger.error(f"❌ OpenAI API error: {e}")
        return ""


@app.get("/")
async def serve_index():
    """Serve the main HTML file"""
    if Path("index.html").exists():
        return FileResponse("index.html")
    else:
        return {
            "message": "OpenAI Whisper STT Server is running",
            "error": "index.html not found. Please create index.html file.",
            "websocket": "ws://localhost:8000/ws/transcribe"
        }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "whisper-1",
        "provider": "OpenAI",
        "websocket": "ws://localhost:8000/ws/transcribe",
        "api_key_configured": bool(OPENAI_API_KEY)
    }

# Run with: uvicorn openai_server:app --reload --host 0.0.0.0 --port 8000
