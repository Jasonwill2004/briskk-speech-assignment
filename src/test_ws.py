import asyncio
import websockets
import json
import pyaudio
import logging
import numpy as np
import sys
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio parameters optimized for Whisper
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
VOICE_THRESHOLD = 0.03
MAX_SILENCE_DURATION = 0.5
MIN_AUDIO_BUFFER = RATE * 1  

class WebSocketClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False

    async def ensure_connected(self) -> bool:
        """Ensure WebSocket connection is active"""
        if not self.connected:
            try:
                self.websocket = await websockets.connect(self.uri)
                self.connected = True
                logger.info("Connected to WebSocket server")
            except Exception as e:
                logger.error(f"Connection failed: {e}")
                return False
        return True

    async def send_audio(self, audio_data: bytes) -> bool:
        """Send audio data with connection check"""
        if await self.ensure_connected():
            try:
                await self.websocket.send(audio_data)
                return True
            except Exception as e:
                logger.error(f"Send failed: {e}")
                self.connected = False
        return False

    async def receive_message(self) -> Optional[dict]:
        """Receive and parse server message"""
        if await self.ensure_connected():
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=0.1)
                return json.loads(response)
            except asyncio.TimeoutError:
                return None
            except Exception as e:
                if "disconnect" not in str(e).lower():
                    logger.error(f"Receive failed: {e}")
                self.connected = False
        return None

    async def close(self):
        """Close WebSocket connection"""
        if self.websocket:
            await self.websocket.close()
            self.connected = False

def calculate_energy(audio_data: np.ndarray) -> float:
    """Calculate audio energy with noise filtering"""
    magnitude = np.abs(audio_data)
    mean_magnitude = np.mean(magnitude)
    peak_magnitude = np.max(magnitude)
    return mean_magnitude if peak_magnitude > VOICE_THRESHOLD * 2 else 0

async def stream_microphone():
    """Stream audio from microphone to WebSocket server"""
    uri = "ws://localhost:8000/ws/speech-to-search"
    ws_client = WebSocketClient(uri)
    
    audio = pyaudio.PyAudio()
    stream = None
    last_transcription = ""
    
    try:
        # Get default input device
        default_device = audio.get_default_input_device_info()
        logger.info(f"Using default input device: {default_device['name']}")
        
        # Open audio stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        print("\n🎤 Testing microphone levels...")
        for _ in range(5):
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            energy = calculate_energy(audio_float)
            if energy > VOICE_THRESHOLD:
                print("✅ Microphone working")
                break
            await asyncio.sleep(0.1)
        
        if not await ws_client.ensure_connected():
            raise ConnectionError("Failed to connect to WebSocket server")
        
        print("\n🎤 Start speaking...")
        
        audio_buffer = bytearray()
        last_voice_time = time.time()
        is_speaking = False
        voiced_frames = 0
        
        while True:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                energy = calculate_energy(audio_float)
                
                current_time = time.time()
                
                if energy > VOICE_THRESHOLD:
                    voiced_frames += 1
                    if not is_speaking and voiced_frames > 3:
                        print("\n🎤 Voice detected...")
                        is_speaking = True
                    last_voice_time = current_time
                    audio_buffer.extend(data)
                    print(".", end="", flush=True)
                    
                    if len(audio_buffer) >= MIN_AUDIO_BUFFER:
                        if await ws_client.send_audio(bytes(audio_buffer)):
                            audio_buffer.clear()
                            voiced_frames = 0
                
                elif is_speaking and (current_time - last_voice_time) > MAX_SILENCE_DURATION:
                    if len(audio_buffer) > MIN_AUDIO_BUFFER // 2:
                        if await ws_client.send_audio(bytes(audio_buffer)):
                            print("\n✅ Processing speech...")
                    audio_buffer.clear()
                    is_speaking = False
                    voiced_frames = 0
                
                # Handle server responses
                message = await ws_client.receive_message()
                if message:
                    if message.get("type") == "transcription":
                        text = message.get("text", "").strip()
                        if text and text != last_transcription:
                            print(f"\n📝 Transcribed: {text}")
                            last_transcription = text
                    elif message.get("type") == "suggestions":
                        suggestions = message.get("suggestions", [])
                        if suggestions:
                            print("\n🔍 Suggestions:")
                            for suggestion in suggestions:
                                print(f"  • {suggestion}")
                    elif message.get("type") == "error":
                        logger.error(f"Server error: {message.get('message')}")
                
                await asyncio.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                if not ws_client.connected:
                    await ws_client.ensure_connected()
    
    except Exception as e:
        logger.error(f"Setup error: {e}")
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        audio.terminate()
        await ws_client.close()
        print("\n👋 Microphone stream closed")

if __name__ == "__main__":
    print("""
🎤 Real-time Speech-to-Search Demo
--------------------------------
• Speak into your microphone
• Watch for voice detection indicators (...)
• Press Ctrl+C to stop
""")
    
    try:
        asyncio.run(stream_microphone())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")