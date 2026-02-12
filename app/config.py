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
    NCBI_EMAIL: str | None = os.getenv("NCBI_EMAIL")  # Required for Entrez (optional for testing)
    SEMANTIC_SCHOLAR_API_KEY: str | None = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    HUGGINGFACE_TOKEN: str | None = os.getenv("HUGGINGFACE_TOKEN")  # For gated models
    
    # === Multiple API Keys Support (Advanced) ===
    # Format: [{"key": "xxx", "requests_per_second": 10}, {"key": "yyy", "requests_per_second": 1}]
    NCBI_API_KEYS: list | None = None
    SEMANTIC_SCHOLAR_API_KEYS: list | None = None
    
    # === Model Paths (HAI-DEF Models) ===
    MEDSIGLIP_MODEL: str = "google/medsiglip-448"
    MEDGEMMA_4B_MODEL: str = "google/medgemma-1.5-4b-it"
    MEDGEMMA_27B_MODEL: str = "google/medgemma-27b-it"
    
    # === MedGemma 4B Fine-Tuned Paths ===
    MEDGEMMA_LORA_ADAPTERS: str = os.getenv("MEDGEMMA_LORA_ADAPTERS", "path/to/lora/adapters")
    MEDGEMMA_PROJECTOR_WEIGHTS: str = os.getenv("MEDGEMMA_PROJECTOR_WEIGHTS", "path/to/projector.pt")
    
    # === Text Embedding Model (for KLE Uncertainty) ===
    # Switch to any sentence-transformers compatible model
    TEXT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # === KLE Uncertainty Settings ===
    KLE_UNCERTAINTY_THRESHOLD: float = 0.30  # Consensus requires uncertainty < threshold
    KLE_NUM_SAMPLES: int = 5  # Number of samples to generate for KLE
    
    # === FHIR Configuration ===
    FHIR_BASE_URL: str = "https://fhir.mimic-iv-demo.physionet.org/fhir"  # Public test server
    FHIR_AUTH_TOKEN: str | None = None
    
    # === Uncertainty Thresholds ===
    THRESHOLD_HISTORIAN: float = 0.30  # U >= 0.30 -> invoke Historian
    THRESHOLD_LITERATURE: float = 0.40  # U >= 0.40 -> invoke Literature
    THRESHOLD_CHIEF: float = 0.50  # U >= 0.50 -> escalate to Chief
    
    # === Execution Limits ===
    MAX_ROUTING_STEPS: int = 5  # Prevent infinite loops
    
    # === DEBATE SETTINGS ===
    # Maximum rounds of debate between Critic and Evidence Team
    DEBATE_MAX_ROUNDS: int = 3
    
    # Maximum confidence disagreement for consensus (0.15 = 15%)
    DEBATE_CONSENSUS_THRESHOLD: float = 0.15
    
    # Enable debate workflow (set False to use legacy routing)
    USE_DEBATE_WORKFLOW: bool = True
    
    # === OPTIMIZATION FLAGS ===
    # Enable fast literature mode (parallel search without ReAct)
    # Set to False to use MedGemma for literature reasoning (slower but potentially more accurate)
    USE_FAST_LITERATURE_MODE: bool = False  # CHANGED: Now Literature uses GPU too
    
    # Enable literature query caching
    USE_LITERATURE_CACHE: bool = True
    
    # Enable parallel agent execution where possible
    USE_PARALLEL_AGENTS: bool = True
    
    # Preload models at startup (uses more memory but faster inference)
    PRELOAD_MODELS: bool = False
    
    # === LLM CRITIC FLAGS ===
    # Enable second-stage MedGemma semantic critic in Critic agent
    ENABLE_LLM_CRITIC: bool = False
    
    # === Mock Mode ===
    # Enable to run without downloading large models (~50GB+)
    MOCK_MODELS: bool = False  # CHANGED: Use real models, not mocks
    
    # === Environment ===
    ENV: str = "development"
    DEBUG: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
