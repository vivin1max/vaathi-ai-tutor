"""Speech-to-Text backends.

Local STT: prefer Whisper (small) if available; fallback to Vosk if installed.
Cloud STT: stub only.
"""
from __future__ import annotations

import os
from typing import Optional


def transcribe_local(wav_path: str) -> str:
    """Transcribe using local models.

    Preference order: whisper (if installed) -> vosk (if installed) -> empty string.
    """
    import logging
    import sys
    from pathlib import Path
    logger = logging.getLogger("ai_core.stt")
    
    if not os.path.exists(wav_path):
        raise FileNotFoundError(wav_path)

    # Try Whisper first
    try:  # pragma: no cover - environment dependent
        import whisper  # type: ignore

        # Add ffmpeg to PATH if it exists in project directory
        project_root = Path(__file__).parent.parent
        ffmpeg_bin = project_root / "ffmpeg-8.0-essentials_build" / "bin"
        if ffmpeg_bin.exists():
            ffmpeg_bin_str = str(ffmpeg_bin)
            if ffmpeg_bin_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = ffmpeg_bin_str + os.pathsep + os.environ.get("PATH", "")
                logger.debug(f"Added ffmpeg to PATH: {ffmpeg_bin_str}")

        logger.debug(f"Loading Whisper model 'small'...")
        model = whisper.load_model("small")
        logger.debug(f"Transcribing {wav_path}...")
        res = model.transcribe(wav_path)
        logger.debug(f"Whisper raw result: {res}")
        text = (res.get("text") or "").strip()
        logger.debug(f"Extracted text: '{text}'")
        return text
    except Exception as e:
        logger.error(f"Whisper failed: {e}")
        logger.exception(e)
        pass

    # Try Vosk next
    try:  # pragma: no cover - environment dependent
        import wave
        import json
        from vosk import Model, KaldiRecognizer  # type: ignore

        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000, 32000, 44100]:
            # For simplicity, we don't resample here
            return ""
        model = Model(lang="en-us")
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)
        text_parts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                ans = rec.Result()
                text_parts.append(json.loads(ans).get("text", ""))
        final = rec.FinalResult()
        text_parts.append(json.loads(final).get("text", ""))
        return " ".join([t for t in text_parts if t]).strip()
    except Exception:
        pass

    return ""


def transcribe_cloud(wav_path: str) -> str:
    """Cloud STT stub (no implementation)."""
    # TODO: integrate cloud STT provider
    return ""
