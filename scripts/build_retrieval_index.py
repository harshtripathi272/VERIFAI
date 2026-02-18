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

from app.config import settings


def load_vision_encoder():
    """Load MedSigLIP vision encoder."""
    print(f"Loading MedSigLIP: {settings.MEDSIGLIP_MODEL}")
    
    image_processor = AutoImageProcessor.from_pretrained(
        settings.MEDSIGLIP_MODEL,
        size={"height": 384, "width": 384}
    )
    
    vision_encoder = SiglipVisionModel.from_pretrained(
        settings.MEDSIGLIP_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    ).eval()
    
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
        
        embedding = embedding.cpu().numpy()
    
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


def build_index(
    mimic_root: Path,
    output_dir: Path,
    num_studies: int = None,
    split: str = "train"
):
    """
    Build FAISS index from MIMIC-CXR training set.
    
    Args:
        mimic_root: Root directory of MIMIC-CXR dataset
        output_dir: Where to save index and metadata
        num_studies: Limit to first N studies (for testing)
        split: Which split to use ('train', 'validate', 'test')
    """
    print("="*80)
    print("MIMIC-CXR Retrieval Index Builder")
    print("="*80)
    
    # Ensure NLTK punkt is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        import nltk
        nltk.download('punkt')
    
    # Load vision encoder
    vision_encoder, image_processor = load_vision_encoder()
    print(f"Vision encoder loaded on: {vision_encoder.device}")
    
    # Load MIMIC metadata
    print(f"\nLoading MIMIC-CXR metadata from {mimic_root}...")
    metadata_df = load_mimic_metadata(mimic_root)
    print(f"Loaded {len(metadata_df)} records")
    
    # Filter to training set
    train_df = metadata_df[metadata_df['split'] == split].copy()
    print(f"Filtered to {len(train_df)} {split} records")
    
    # Load reports
    print(f"\nLoading radiology reports...")
    reports_df = load_mimic_reports(mimic_root)
    print(f"Loaded {len(reports_df)} reports")
    
    # Merge images with reports
    train_df = train_df.merge(reports_df, on='study_id', how='inner')
    print(f"Merged to {len(train_df)} records with reports")
    
    # Group by study_id
    studies = train_df.groupby('study_id')
    print(f"\nProcessing {len(studies)} unique studies...")
    
    if num_studies:
        study_ids = list(studies.groups.keys())[:num_studies]
        print(f"Limited to first {num_studies} studies")
    else:
        study_ids = list(studies.groups.keys())
    
    # Build embeddings and metadata
    embeddings = []
    metadata = []
    
    for study_id in tqdm(study_ids, desc="Embedding studies"):
        study_group = studies.get_group(study_id)
        
        # Get all images for this study
        study_images = []
        for _, row in study_group.iterrows():
            img_path = mimic_root / "files" / f"p{str(row['subject_id'])[:2]}" / f"p{row['subject_id']}" / f"s{study_id}" / f"{row['dicom_id']}.jpg"
            
            if img_path.exists():
                study_images.append({
                    "path": str(img_path),
                    "view_position": row['ViewPosition']
                })
        
        if not study_images:
            continue
        
        # Select 1-2 representative images
        selected_images = select_study_images(study_images)
        
        # Embed each selected image
        study_embeddings = []
        for img_info in selected_images:
            try:
                embedding = embed_image(img_info["path"], vision_encoder, image_processor)
                study_embeddings.append(embedding)
            except Exception as e:
                print(f"\nWarning: Failed to embed {img_info['path']}: {e}")
                continue
        
        if not study_embeddings:
            continue
        
        # Average embeddings if we have 2 views
        if len(study_embeddings) == 2:
            avg_embedding = np.mean(study_embeddings, axis=0)
        else:
            avg_embedding = study_embeddings[0]
        
        # Get report text
        report_row = study_group.iloc[0]
        report_text = report_row['findings'] + " " + report_row['impression']
        
        # Split into sentences
        try:
            sentences = sent_tokenize(report_text)
        except Exception as e:
            print(f"\nWarning: Failed to tokenize report for {study_id}: {e}")
            continue
        
        # Infer primary label (simplified - you may want more sophisticated logic)
        primary_label = "No Finding"  # Default
        if "pneumonia" in report_text.lower():
            primary_label = "Pneumonia"
        elif "cardiomegaly" in report_text.lower():
            primary_label = "Cardiomegaly"
        elif "effusion" in report_text.lower():
            primary_label = "Effusion"
        # Add more label extraction logic as needed
        
        # Store one entry per sentence (all share same averaged embedding)
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            embeddings.append(avg_embedding)
            metadata.append({
                "sentence": sentence.strip(),
                "study_id": study_id,
                "primary_label": primary_label,
                "views_used": [img["view_position"] for img in selected_images]
            })
    
    print(f"\n\nBuilt {len(embeddings)} sentence embeddings from {len(study_ids)} studies")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    embedding_matrix = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embedding_matrix)  # L2 normalize for cosine similarity
    
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])  # Inner product = cosine after normalize
    index.add(embedding_matrix)
    
    print(f"FAISS index built with {index.ntotal} vectors of dimension {embedding_matrix.shape[1]}")
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = output_dir / "mimic_corpus.faiss"
    metadata_path = output_dir / "mimic_corpus_metadata.json"
    
    faiss.write_index(index, str(index_path))
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Index saved to: {index_path}")
    print(f"✓ Metadata saved to: {metadata_path}")
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Build MIMIC-CXR retrieval index")
    parser.add_argument(
        "--mimic_root",
        type=str,
        required=True,
        help="Root directory of MIMIC-CXR dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for index and metadata"
    )
    parser.add_argument(
        "--num_studies",
        type=int,
        default=None,
        help="Limit to first N studies (for testing)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validate", "test"],
        help="Which split to use"
    )
    
    args = parser.parse_args()
    
    mimic_root = Path(args.mimic_root)
    output_dir = Path(args.output_dir)
    
    if not mimic_root.exists():
        print(f"Error: MIMIC-CXR root directory not found: {mimic_root}")
        return
    
    build_index(
        mimic_root=mimic_root,
        output_dir=output_dir,
        num_studies=args.num_studies,
        split=args.split
    )


if __name__ == "__main__":
    main()
