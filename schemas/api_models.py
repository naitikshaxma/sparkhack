from pydantic import BaseModel
from typing import Optional


class IntentRequest(BaseModel):
    text: str
    session_id: str = ""
    language: str = ""


class TTSRequest(BaseModel):
    text: str
    language: str = ""
    session_id: str = ""
    tone: str = ""


class AutofillRequest(BaseModel):
    session_id: str


class ResetSessionRequest(BaseModel):
    session_id: str


class LoginRequest(BaseModel):
    email: str
    name: str = ""
    password: str = ""


class SignupRequest(BaseModel):
    email: str
    password: str


class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    aadhaar: Optional[str] = None
