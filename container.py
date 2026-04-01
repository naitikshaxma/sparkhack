from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends

from backend.core.config import Settings, get_settings
from .services.conversation_service import ConversationService
from .services.intent_service import IntentService
from .services.ocr_service import OcrService
from .services.stt_service import STTService
from .services.system_service import SystemService
from .services.tts_service import TTSService
from backend.shared.validators.input_validator import InputValidator


@dataclass(frozen=True)
class ServiceContainer:
    settings: Settings
    intent_service: IntentService
    conversation_service: ConversationService
    ocr_service: OcrService
    tts_service: TTSService
    stt_service: STTService
    system_service: SystemService
    input_validator: InputValidator


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    settings = get_settings()
    return ServiceContainer(
        settings=settings,
        intent_service=IntentService(),
        conversation_service=ConversationService(),
        ocr_service=OcrService(),
        tts_service=TTSService(),
        stt_service=STTService(),
        system_service=SystemService(),
        input_validator=InputValidator(max_chars=settings.max_text_input_chars),
    )


def get_service_container() -> ServiceContainer:
    return get_container()


def inject_container(container: ServiceContainer = Depends(get_service_container)) -> ServiceContainer:
    return container
