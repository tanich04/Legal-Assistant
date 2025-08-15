import os
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    # API Keys
    API_KEY: str = "your_default_api_key"
    OPENAI_API_KEY: str = ""
    GROQ_API_TOKEN: str = os.getenv("GROQ_API_KEY", "")

    
    # Server Configuration
    API_PREFIX: str = "/api"
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # Model Configuration
    ENGLISH_MODEL: str = "google/flan-t5-base"
    HINDI_MODEL: str = "google/mt5-base"
    HINGLISH_MODEL: str = "google/mt5-base"
    MAX_NEW_TOKENS: int = 256
    
    # Twilio Configuration
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_NUMBER: str = os.getenv("TWILIO_NUMBER", "")
    
    # Storage Configuration
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/specter.log")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    PDF_STORAGE_PATH: str = os.getenv("PDF_STORAGE_PATH", "storage/pdfs")
    MAX_PDF_AGE_DAYS: int = int(os.getenv("MAX_PDF_AGE_DAYS", "7"))
    LEGAL_DOCS_PATH: str = os.getenv("LEGAL_DOCS_PATH", "data/legal_docs")
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
    
    # Public URL for audio files
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8001")

    model_config = {
        "env_file": ".env",
        "extra": "ignore"
    }

settings = Settings()
print(f"Loaded API_KEY: {settings.API_KEY}")
