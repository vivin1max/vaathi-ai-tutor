from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from backend.services.ai_adapter import text_to_speech, speech_to_text
from backend.utils.files import save_upload
from pydantic import BaseModel
import os
import uuid
from pathlib import Path
import subprocess
import shutil

router = APIRouter()

AUDIO_DIR = Path("./audio")
AUDIO_DIR.mkdir(exist_ok=True, parents=True)

def temp_audio_path():
    return str(AUDIO_DIR / f"{uuid.uuid4()}.mp3")

class TTSRequest(BaseModel):
    text: str

@router.post("/tts")
async def tts(request: TTSRequest):
    out_path = temp_audio_path()
    text_to_speech(request.text, out_path)
    return FileResponse(out_path, media_type="audio/mpeg", filename=os.path.basename(out_path))


import logging
import shutil
logger = logging.getLogger("backend.routers.media")

def find_ffmpeg():
    """Find ffmpeg executable in common locations."""
    # Check project-local ffmpeg first
    project_ffmpeg = Path(__file__).parent.parent.parent / "ffmpeg-8.0-essentials_build" / "bin" / "ffmpeg.exe"
    if project_ffmpeg.exists():
        return str(project_ffmpeg)
    
    # Check if ffmpeg is in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Check common Windows installation locations
    common_paths = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

@router.post("/stt")
async def stt(audio: UploadFile = File(...)):
    path = save_upload(audio)
    logger.info(f"[STT] Saved audio to: {path}")

    # Convert webm to wav if needed
    wav_path = None
    if path.lower().endswith('.webm'):
        wav_path = path[:-5] + '.wav'
        
        # Find ffmpeg
        ffmpeg_exe = find_ffmpeg()
        if not ffmpeg_exe:
            logger.error("[STT] ffmpeg not found in PATH or common locations")
            return {"text": ""}
        
        logger.info(f"[STT] Using ffmpeg: {ffmpeg_exe}")
        
        try:
            # Use subprocess to run ffmpeg directly
            cmd = [
                ffmpeg_exe,
                '-i', path,
                '-f', 'wav',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-y',  # Overwrite output
                wav_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"[STT] ffmpeg failed: {result.stderr}")
                return {"text": ""}
            
            # Log WAV file info
            wav_size = os.path.getsize(wav_path)
            logger.info(f"[STT] Converted to WAV: {wav_path} ({wav_size} bytes)")
            
            # Get audio duration using ffmpeg
            probe_cmd = [ffmpeg_exe, '-i', wav_path, '-f', 'null', '-']
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            # Duration is in stderr output
            import re
            duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', probe_result.stderr)
            if duration_match:
                h, m, s = duration_match.groups()
                duration_sec = int(h) * 3600 + int(m) * 60 + float(s)
                logger.info(f"[STT] Audio duration: {duration_sec:.2f} seconds")
            else:
                logger.warning(f"[STT] Could not determine audio duration")
        except Exception as e:
            logger.error(f"[STT] ffmpeg conversion failed: {e}")
            return {"text": ""}
    else:
        wav_path = path

    text = speech_to_text(wav_path)
    logger.info(f"[STT] Transcription result: '{text}'")
    return {"text": text}
