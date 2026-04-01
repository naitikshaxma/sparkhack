import os
import shutil
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Optional

from backend.core.config import get_settings
from backend.core.logger import log_event
from backend.shared.language.language import normalize_language_code

try:
    import imageio_ffmpeg  # type: ignore
except Exception:
    imageio_ffmpeg = None

if shutil.which("ffmpeg") is None and imageio_ffmpeg is not None:
    bundled_ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    ffmpeg_alias = bundled_ffmpeg_path.with_name("ffmpeg.exe")
    if not ffmpeg_alias.exists():
        try:
            shutil.copy2(bundled_ffmpeg_path, ffmpeg_alias)
        except Exception:
            ffmpeg_alias = bundled_ffmpeg_path

    bundled_dir = str(ffmpeg_alias.parent)
    os.environ["PATH"] = f"{bundled_dir}{os.pathsep}{os.environ.get('PATH', '')}"

WHISPER_MODEL_NAME = get_settings().whisper_model_size
DEFAULT_TRANSCRIBE_LANGUAGE = "en"
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
_model = None
_effective_model_name = ""
_whisper_module = None
_whisper_import_error: Optional[Exception] = None
logger = logging.getLogger(__name__)


def _normalize_suffix(source_suffix: Optional[str]) -> str:
    suffix = (source_suffix or ".webm").strip()
    if not suffix:
        return ".webm"
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return suffix


def warmup_whisper() -> None:
    _get_model()


def _get_model():
    global _model, _effective_model_name, _whisper_module, _whisper_import_error
    if _model is None:
        if _whisper_module is None and _whisper_import_error is None:
            try:
                import whisper as whisper_module  # type: ignore

                _whisper_module = whisper_module
            except Exception as exc:
                _whisper_import_error = exc
                raise RuntimeError("Whisper dependency unavailable") from exc

        if _whisper_module is None:
            raise RuntimeError("Whisper dependency unavailable") from _whisper_import_error

        configured_name = (WHISPER_MODEL_NAME or "").strip()
        effective_name = configured_name[:-3] if configured_name.endswith(".en") else configured_name
        if effective_name != configured_name:
            logger.warning(
                "Configured Whisper model '%s' is English-only; using multilingual '%s' to preserve source-language transcription.",
                configured_name,
                effective_name,
            )
        logger.info("Loading Whisper STT model", extra={"configured_model": configured_name, "effective_model": effective_name})
        _model = _whisper_module.load_model(effective_name)
        _effective_model_name = effective_name
    return _model


def get_whisper_status() -> dict:
    return {
        "model_name": WHISPER_MODEL_NAME,
        "effective_model_name": _effective_model_name or WHISPER_MODEL_NAME,
        "model_loaded": _model is not None,
        "ffmpeg_available": FFMPEG_AVAILABLE,
        "default_language": DEFAULT_TRANSCRIBE_LANGUAGE,
    }


def transcribe_audio(audio_bytes: bytes, language: Optional[str] = None, source_suffix: str = ".webm") -> str:
    """
    Transcribe audio bytes using Whisper.
    """
    if not audio_bytes:
        return ""
    if not FFMPEG_AVAILABLE:
        raise RuntimeError("ffmpeg is not available in PATH; Whisper transcription requires ffmpeg.")

    suffix = _normalize_suffix(source_suffix)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    normalized_path = ""
    provided_language = (language or "").strip().lower()

    try:
        requested_language = normalize_language_code(language, default=DEFAULT_TRANSCRIBE_LANGUAGE)
        normalized_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        normalized_tmp.close()
        normalized_path = normalized_tmp.name

        # Normalize to 16k mono wav for stable STT behavior across browsers/devices.
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-ac",
                "1",
                "-ar",
                "16000",
                normalized_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        options = {
            "fp16": False,
            "task": "transcribe",
            "temperature": 0,
        }
        # Avoid forcing English when caller language is implicit/default "en".
        # Let Whisper auto-detect source language to preserve original Hindi text.
        if requested_language == "hi" or provided_language == "hi":
            options["language"] = "hi"
        result = _get_model().transcribe(normalized_path, **options)
        transcript = result.get("text", "").strip()
        detected_or_used_language = result.get("language") or options.get("language") or requested_language
        log_event(
            "stt_transcription_complete",
            endpoint="whisper_service",
            status="success",
            language=str(detected_or_used_language or ""),
            transcript_length=len(transcript or ""),
        )
        return transcript
    except Exception:
        requested_language = normalize_language_code(language, default=DEFAULT_TRANSCRIBE_LANGUAGE)
        options = {
            "fp16": False,
            "task": "transcribe",
            "temperature": 0,
        }
        if requested_language == "hi" or provided_language == "hi":
            options["language"] = "hi"
        result = _get_model().transcribe(tmp_path, **options)
        transcript = result.get("text", "").strip()
        detected_or_used_language = result.get("language") or options.get("language") or requested_language
        log_event(
            "stt_transcription_complete",
            endpoint="whisper_service",
            status="success",
            language=str(detected_or_used_language or ""),
            transcript_length=len(transcript or ""),
        )
        return transcript
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if normalized_path and os.path.exists(normalized_path):
            os.remove(normalized_path)
