import base64
import io
import logging
import re
import struct
import wave

try:
    from gtts import gTTS
except Exception:  # pragma: no cover - optional dependency for lightweight deploys
    gTTS = None


logger = logging.getLogger(__name__)


def _resolve_language(language: str) -> str:
    raw_language = (language or "").strip().lower()
    if raw_language.startswith("hi"):
        return "hi"
    if raw_language.startswith("en"):
        return "en"
    raise ValueError("Unsupported language for TTS. Allowed values: en, hi")


def split_tts_chunks(text: str, max_chars: int = 180) -> list[str]:
    content = (text or "").strip()
    if not content:
        return []

    parts = re.split(r"(?<=[.!?।])\s+", content)
    chunks: list[str] = []
    current = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        if len(current) + len(piece) + 1 <= max_chars:
            current = f"{current} {piece}".strip()
        else:
            if current:
                chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks or [content]


def generate_tts_bytes(text: str, language: str) -> bytes:
    clean_text = (text or "").strip()
    if not clean_text:
        return b""

    lang = _resolve_language(language)
    logger.info("[lang-debug] tts_language=%s", lang)
    if gTTS is None:
        # Deterministic fallback: return a short silent WAV to keep API contract stable.
        return _silent_wav_bytes(duration_ms=320)

    tts = gTTS(text=clean_text, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    return fp.getvalue()


def _silent_wav_bytes(duration_ms: int = 300, sample_rate: int = 16000) -> bytes:
    frame_count = max(1, int(sample_rate * max(50, duration_ms) / 1000))
    pcm = struct.pack("<h", 0) * frame_count
    buff = io.BytesIO()
    with wave.open(buff, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buff.getvalue()


def generate_tts(text: str, language: str) -> str:
    payload = generate_tts_bytes(text, language)
    return base64.b64encode(payload).decode("utf-8") if payload else ""
