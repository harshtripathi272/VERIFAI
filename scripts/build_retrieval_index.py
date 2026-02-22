"""
Build MIMIC-CXR Retrieval Index

This script pre-builds a FAISS index of MIMIC-CXR training set embeddings
for the CXR-RePaiR retrieval tool.

Run this once offline to create:
- data/mimic_corpus.faiss (FAISS index)
- data/mimic_corpus_metadata.json (sentence metadata)

Usage:
    python scripts/build_retrieval_index.py --mimic_root /path/to/mimic-cxr \\
        --output_dir data/ --num_studies 1000

Requirements:
- MIMIC-CXR dataset access
- MedSigLIP vision encoder
- NLTK punkt tokenizer: nltk.download('punkt')
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# FAISS for indexing
import faiss

# NLTK for sentence tokenization
from nltk.tokenize import sent_tokenize

# Vision model
from transformers import AutoImageProcessor
from transformers import SiglipVisionModel

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
    MEDSIGLIP_BASE_MODEL: str = "google/medsiglip-448" 
    MEDSIGLIP_WEIGHTS_PATH : str = "../output/medsiglip_full_model.pt" 
    MEDGEMMA_4B_MODEL: str = "google/medgemma-1.5-4b-it"
    #MEDGEMMA_27B_MODEL: str = "google/medgemma-27b-it"
    
    # === MedGemma 4B Fine-Tuned Paths ===
    MEDGEMMA_LORA_ROOT: str = os.getenv("MEDGEMMA_LORA_ROOT", "../dataset/med/fine_tuned_model/v1/checkpoint-700/")
    MEDGEMMA_LORA_ADAPTERS: str = os.getenv("MEDGEMMA_LORA_ADAPTERS", "../dataset/med/fine_tuned_model/v1/checkpoint-700/")
    
    # === Text Embedding Model (for KLE Uncertainty) ===
    # Switch to any sentence-transformers compatible model
    TEXT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # === KLE Uncertainty Settings ===
    KLE_UNCERTAINTY_THRESHOLD: float = 0.30  # Consensus requires uncertainty < threshold
    KLE_NUM_SAMPLES: int = 3  # Number of samples to generate for KLE
    
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
    
    # === PAST MISTAKES MEMORY ===
    # Enable historical mistake retrieval in critic
    ENABLE_PAST_MISTAKES_MEMORY: bool = bool(os.getenv("ENABLE_PAST_MISTAKES_MEMORY", "True"))
    
    # Past mistakes database path (DuckDB with VSS extension)
    PAST_MISTAKES_DB_PATH: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "verifai_past_mistakes.duckdb"
    )
    
    # Retrieval settings
    PAST_MISTAKES_TOP_K: int = 5  # Maximum similar cases to retrieve
    PAST_MISTAKES_SIMILARITY_THRESHOLD: float = 0.75  # Minimum cosine similarity
    PAST_MISTAKES_KLE_TOLERANCE: float = 0.2  # +/- range for KLE filtering
    ENABLE_PAST_MISTAKES_RERANKING: bool = bool(os.getenv("ENABLE_PAST_MISTAKES_RERANKING", "True"))  # Neural re-ranking
    
    # === SUPABASE (Cloud Database) ===
    # Supabase connection for cloud-based structured logging
    SUPABASE_URL: str | None = os.getenv("SUPABASE_URL")
    SUPABASE_KEY: str | None = os.getenv("SUPABASE_KEY")
    SUPABASE_SERVICE_KEY: str | None = os.getenv("SUPABASE_SERVICE_KEY")  # Optional: for admin operations
    
    # Database mode selection
    DATABASE_MODE: str = os.getenv("DATABASE_MODE", "supabase")  # 'supabase' or 'sqlite'
    
    # === DOCTOR FEEDBACK LOOP ===
    # Enable doctor feedback-driven reprocessing
    ENABLE_DOCTOR_FEEDBACK: bool = bool(os.getenv("ENABLE_DOCTOR_FEEDBACK", "True"))
    
    # Automatically restart from critic when feedback is provided
    FEEDBACK_RESTART_FROM_CRITIC: bool = bool(os.getenv("FEEDBACK_RESTART_FROM_CRITIC", "True"))
    
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



def load_vision_encoder():
    """Load custom saved MedSigLIP encoder."""

    print(f"Loading custom MedSigLIP encoder from: {settings.MEDSIGLIP_WEIGHTS_PATH}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load base architecture
    vision_encoder = SiglipVisionModel.from_pretrained(
        settings.MEDSIGLIP_BASE_MODEL
    )

    # Load your trained weights
    state_dict = torch.load(
        settings.MEDSIGLIP_WEIGHTS_PATH,
        map_location=device
    )

    # If checkpoint contains full wrapper, extract backbone
    if "vision_model." in list(state_dict.keys())[0]:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vision_model."):
                new_state_dict[k.replace("vision_model.", "")] = v
        state_dict = new_state_dict

    vision_encoder.load_state_dict(state_dict, strict=False)

    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()

    image_processor = AutoImageProcessor.from_pretrained(
        settings.MEDSIGLIP_BASE_MODEL
    )

    return vision_encoder, image_processor



def select_study_images(all_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select up to 2 representative images from a study.
    
    Strategy:
    - If PA exists: take first PA image
    - If AP exists: take first AP image  
    - Use both if both exist
    - If neither PA nor AP: take first LAT
    - If no standard views: take first image
    
    Args:
        all_images: List of dicts with keys 'path' and 'view_position'
    
    Returns:
        List of 1-2 selected images
    """
    pa_images = [i for i in all_images if i["view_position"] == "PA"]
    ap_images = [i for i in all_images if i["view_position"] == "AP"]
    lat_images = [i for i in all_images if i["view_position"] in ["LAT", "LATERAL"]]
    
    selected = []
    
    # Add one PA if available
    if pa_images:
        selected.append(sorted(pa_images, key=lambda x: x["path"])[0])
    
    # Add one AP if available
    if ap_images:
        selected.append(sorted(ap_images, key=lambda x: x["path"])[0])
    
    # If we have at least one frontal view, return
    if selected:
        return selected
    
    # Fallback to LAT if no frontal views
    if lat_images:
        return [sorted(lat_images, key=lambda x: x["path"])[0]]
    
    # Last resort: first image
    if all_images:
        return [sorted(all_images, key=lambda x: x["path"])[0]]
    
    return []


def embed_image(image_path: str, vision_encoder, image_processor) -> np.ndarray:
    """
    Embed a single image using MedSigLIP.
    
    Returns:
        embedding: numpy array of shape (hidden_size,)
    """
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    
    pixel_values = inputs.pixel_values.to(
        vision_encoder.device,
        dtype=vision_encoder.dtype
    )
    
    with torch.no_grad():
        vision_outputs = vision_encoder(pixel_values=pixel_values)
        
        # Use pooler_output for global representation
        if hasattr(vision_outputs, 'pooler_output'):
            embedding = vision_outputs.pooler_output.squeeze()
        else:
            # Fallback: mean pool over patches
            embedding = vision_outputs.last_hidden_state.mean(dim=1).squeeze()
        
        embedding = embedding.float().cpu().numpy().astype("float32")
    
    return embedding


def load_mimic_metadata(mimic_root: Path) -> pd.DataFrame:
    """
    Load MIMIC-CXR metadata.
    
    Expected structure:
    - mimic_root/mimic-cxr-2.0.0-metadata.csv
    - mimic_root/mimic-cxr-2.0.0-split.csv
    
    Returns:
        DataFrame with columns: study_id, subject_id, dicom_id, ViewPosition, split
    """
    metadata_file = mimic_root / "mimic-cxr-2.0.0-metadata.csv"
    split_file = mimic_root / "mimic-cxr-2.0.0-split.csv"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    
    metadata = pd.read_csv(metadata_file)
    splits = pd.read_csv(split_file)
    
    # Merge metadata with splits
    df = metadata.merge(splits, on=['dicom_id', 'subject_id', 'study_id'], how='inner')
    
    return df


def load_mimic_reports(mimic_root: Path) -> pd.DataFrame:
    """
    Load MIMIC-CXR radiology reports.
    
    Expected structure:
    - mimic_root/files/p10/p10000032/s50414267.txt
    
    Returns:
        DataFrame with columns: study_id, findings, impression
    """
    # This is a simplified version - you'll need to adapt to your MIMIC-CXR structure
    # Typically you'd parse the structured text files
    
    reports = []
    files_dir = mimic_root / "files"
    
    if not files_dir.exists():
        raise FileNotFoundError(f"Reports directory not found: {files_dir}")
    
    # Search for all .txt report files
    for report_file in files_dir.rglob("*.txt"):
        study_id = report_file.stem  # e.g., s50414267
        
        # Parse report (simplified - adapt to your format)
        with open(report_file, 'r') as f:
            content = f.read()
        
        # Extract FINDINGS and IMPRESSION sections
        findings = ""
        impression = ""
        
        if "FINDINGS:" in content:
            findings_start = content.index("FINDINGS:") + len("FINDINGS:")
            findings_end = content.index("IMPRESSION:") if "IMPRESSION:" in content else len(content)
            findings = content[findings_start:findings_end].strip()
        
        if "IMPRESSION:" in content:
            impression_start = content.index("IMPRESSION:") + len("IMPRESSION:")
            impression = content[impression_start:].strip()
        
        reports.append({
            "study_id": study_id,
            "findings": findings,
            "impression": impression
        })
    
    return pd.DataFrame(reports)


def build_index_from_jsonl(
    jsonl_path: Path,
    image_root: Path,
    output_dir: Path,
    num_studies: int = None
):
    print("="*80)
    print("Building Retrieval Index From JSONL Dataset")
    print("="*80)

    vision_encoder, image_processor = load_vision_encoder()

    studies = []

    # Load JSONL
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                studies.append(json.loads(line))

    if num_studies:
        studies = studies[:num_studies]

    print(f"Loaded {len(studies)} studies")

    embeddings = []
    metadata = []

    for study in tqdm(studies, desc="Embedding studies"):

        findings = study.get("findings", "")
        impression = study.get("impression", "")
        images_info = study.get("images", [])

        if not findings and not impression:
            continue

        # Select representative images (reuse your logic)
        selected_images = select_study_images([
            {
                "path": str(image_root / img["path"]),
                "view_position": img.get("view", "UNKNOWN")
            }
            for img in images_info
        ])

        study_embeddings = []

        for img_info in selected_images:
            if not Path(img_info["path"]).exists():
                continue

            try:
                emb = embed_image(img_info["path"], vision_encoder, image_processor)
                study_embeddings.append(emb)
            except:
                continue

        if not study_embeddings:
            continue

        # Normalize + average
        if len(study_embeddings) == 2:
            emb = np.vstack(study_embeddings)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            avg_embedding = np.mean(emb, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        else:
            avg_embedding = study_embeddings[0]
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        report_text = findings + " " + impression
        sentences = sent_tokenize(report_text)

        for sentence in sentences:
            if len(sentence.strip()) < 10:
                continue

            embeddings.append(avg_embedding.astype("float32"))
            metadata.append({
                "sentence": sentence.strip(),
                "views_used": [img["view_position"] for img in selected_images]
            })

    if len(embeddings) == 0:
        raise ValueError("No embeddings generated.")

    print(f"Built {len(embeddings)} sentence embeddings")

    # Build FAISS
    embedding_matrix = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embedding_matrix)

    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embedding_matrix)

    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "retrieval_index.faiss"
    metadata_path = output_dir / "retrieval_metadata.json"

    faiss.write_index(index, str(index_path))
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Index saved to {index_path}")
    print(f"Metadata saved to {metadata_path}")



def main():
    parser = argparse.ArgumentParser(description="Build Retrieval Index from JSONL")
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--num_studies", type=int, default=None)

    args = parser.parse_args()

    build_index_from_jsonl(
        jsonl_path=Path(args.jsonl_path),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        num_studies=args.num_studies
    )



if __name__ == "__main__":
    main()
