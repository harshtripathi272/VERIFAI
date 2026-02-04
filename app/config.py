"""
VERIFAI Configuration

Environment variables, model paths, and thresholds.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""
    
    # === API Keys ===
    NCBI_API_KEY: str | None = os.getenv("NCBI_API_KEY")  # For higher E-utilities rate limits
    NCBI_EMAIL: str = os.getenv("NCBI_EMAIL")  # Required for Entrez
    SEMANTIC_SCHOLAR_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    HUGGINGFACE_TOKEN: str | None = os.getenv("HUGGINGFACE_TOKEN")  # For gated models
    
    # === Model Paths (HAI-DEF Models) ===
    MEDSIGLIP_MODEL: str = "google/medsiglip-448"
    MEDGEMMA_4B_MODEL: str = "google/medgemma-1.5-4b-it"
    MEDGEMMA_27B_MODEL: str = "google/medgemma-27b-it"
    
    # === FHIR Configuration ===
    FHIR_BASE_URL: str = "https://fhir.mimic-iv-demo.physionet.org/fhir"  # Public test server
    FHIR_AUTH_TOKEN: str | None = None
    
    # === Uncertainty Thresholds ===
    THRESHOLD_HISTORIAN: float = 0.30  # U >= 0.30 -> invoke Historian
    THRESHOLD_LITERATURE: float = 0.40  # U >= 0.40 -> invoke Literature
    THRESHOLD_CHIEF: float = 0.50  # U >= 0.50 -> escalate to Chief
    
    # === Execution Limits ===
    MAX_ROUTING_STEPS: int = 5  # Prevent infinite loops
    
    # === Mock Mode ===
    # Enable to run without downloading large models (~50GB+)
    MOCK_MODELS: bool = True
    
    # === Environment ===
    ENV: str = "development"
    DEBUG: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
