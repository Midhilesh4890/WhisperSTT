import base64
import io
import numpy as np
import soundfile as sf
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = whisper.load_model("base")
SAMPLE_RATE = 16000


@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("🔌 WebSocket connected")

    try:
        while True:
            # Receive binary data instead of text
            try:
                data = await websocket.receive_bytes()
                logger.info(f"Received {len(data)} bytes of audio data")

                # Convert bytes to numpy array
                audio_data = np.frombuffer(data, dtype=np.float32)

                # Ensure we have enough audio data (at least 1 second)
                if len(audio_data) < SAMPLE_RATE:
                    logger.warning("Audio chunk too short, skipping...")
                    continue

                # Pad or trim audio to 30 seconds (Whisper's expected input length)
                audio_padded = whisper.pad_or_trim(audio_data)

                # Transcribe
                result = model.transcribe(
                    audio_padded, fp16=False, language='en')
                transcription = result["text"].strip()

                if transcription:
                    logger.info(f"Transcription: {transcription}")
                    await websocket.send_text(transcription)
                else:
                    await websocket.send_text("[No speech detected]")

            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                await websocket.send_text(f"Error processing audio: {str(e)}")
                continue

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


@app.get("/")
async def get_client():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/api")
async def root():
    return {"message": "Whisper WebSocket STT server is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
