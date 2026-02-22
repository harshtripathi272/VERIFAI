"""
CXR-RePaiR Retrieval Tool

Given chest X-ray images from a study, search a pre-built FAISS index of MIMIC-CXR 
training embeddings to find the most visually similar historical cases.

Image Selection Strategy:
- Use up to 2 views: one PA image + one AP image
- If study has both PA and AP: use both (2 embeddings)
- If study has only PA or only AP: use that one (1 embedding)
- If study has neither PA nor AP: fall back to one LAT image
- If no standard views exist: use first image in study
"""

import faiss
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter
from typing import TYPE_CHECKING, List, Dict, Any

if TYPE_CHECKING:
    from graph.state import VerifaiState


class CXRRetrieverTool:
    """
    Retrieval tool for finding similar historical cases using FAISS.
    
    Uses MedSigLIP vision encoder for embedding images.
    """
    
    def __init__(
        self, 
        index_path: str, 
        metadata_path: str, 
        vision_encoder, 
        image_processor
    ):
        """
        Args:
            index_path: Path to FAISS index file (*.faiss)
            metadata_path: Path to metadata JSON
            vision_encoder: MedSigLIP vision encoder model
            image_processor: MedSigLIP image processor
        """
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.vision_encoder = vision_encoder
        self.image_processor = image_processor
        self.vision_encoder.eval()
    
    def _embed_image(self, image_path: str) -> np.ndarray:
        """
        Embed a single image using MedSigLIP encoder.
        
        Returns:
            embedding: numpy array of shape (hidden_size,)
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        pixel_values = inputs.pixel_values.to(
            self.vision_encoder.device,
            dtype=self.vision_encoder.dtype
        )
        
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=pixel_values)
            # Use pooler_output for global representation
            if hasattr(vision_outputs, 'pooler_output'):
                embedding = vision_outputs.pooler_output.squeeze()
            else:
                # Fallback: mean pool over patches
                embedding = vision_outputs.last_hidden_state.mean(dim=1).squeeze()
            
            embedding = embedding.cpu().numpy()
        
        return embedding
    
    def _select_study_images(self, all_images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        pa_images = [i for i in all_images if i.get("view_position", "").upper() == "PA"]
        ap_images = [i for i in all_images if i.get("view_position", "").upper() == "AP"]
        lat_images = [i for i in all_images if i.get("view_position", "").upper() in ["LAT", "LATERAL"]]
        
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
    
    def _load_study_images(self, image_path_or_dir: Any) -> List[Dict[str, Any]]:
        """
        Load all images for the study.
        
        Supports:
        - Single string path to image file
        - List of string paths to image files
        - Path to directory
        
        Args:
            image_path_or_dir: Path(s) to image file(s) or directory
        
        Returns:
            List of dicts: [{"path": str, "view_position": str}, ...]
        """
        # Handle list of paths
        if isinstance(image_path_or_dir, list):
            results = []
            for p in image_path_or_dir:
                results.extend(self._load_study_images(p))
            return results
            
        path = Path(image_path_or_dir)
        
        if path.is_file():
            # Single image - try to infer view from filename
            view = self._infer_view_from_filename(path.name)
            return [{"path": str(path), "view_position": view}]
        
        elif path.is_dir():
            # Directory of images
            images = []
            for img_path in path.glob("*.jpg") or path.glob("*.png") or path.glob("*.dcm"):
                view = self._infer_view_from_filename(img_path.name)
                images.append({"path": str(img_path), "view_position": view})
            return images
        
        else:
            # Fallback
            return [{"path": str(image_path_or_dir), "view_position": "PA"}]
    
    def _infer_view_from_filename(self, filename: str) -> str:
        """
        Infer view position from filename.
        
        Common patterns: PA, AP, LAT, LATERAL, etc.
        """
        filename_upper = filename.upper()
        
        if "PA" in filename_upper or "FRONTAL" in filename_upper:
            return "PA"
        elif "AP" in filename_upper:
            return "AP"
        elif "LAT" in filename_upper or "LATERAL" in filename_upper:
            return "LAT"
        else:
            # Default to PA for frontal views
            return "PA"
    
    def _infer_label_from_sentence(self, sentence: str) -> str:
        """
        Infer a CheXbert-style label from the sentence text when metadata
        does not contain an explicit primary_label field.
        
        Uses simple keyword matching — good enough for consensus heuristics.
        """
        s = sentence.lower()
        if any(k in s for k in ["pleural effusion", "effusion"]):
            return "Pleural Effusion"
        if any(k in s for k in ["pneumothorax"]):
            return "Pneumothorax"
        if any(k in s for k in ["pneumonia", "consolidat", "airspace opacity", "opacification"]):
            return "Pneumonia"
        if any(k in s for k in ["edema", "pulmonary vascular congestion", "interstitial"]):
            return "Edema"
        if any(k in s for k in ["atelectasis", "discoid", "linear opacity", "subsegmental"]):
            return "Atelectasis"
        if any(k in s for k in ["cardiomegaly", "cardiac silhouette is enlarged", "enlarged heart"]):
            return "Cardiomegaly"
        if any(k in s for k in ["fracture"]):
            return "Fracture"
        if any(k in s for k in ["mass", "nodule", "lesion"]):
            return "Lung Lesion"
        if any(k in s for k in ["support device", "line", "catheter", "tube", "pacemaker", "icd"]):
            return "Support Devices"
        if any(k in s for k in ["no acute", "unremarkable", "clear lung", "no evidence", "normal", "no abnormal"]):
            return "No Finding"
        return "Unknown"

    def execute(self, state: "VerifaiState") -> Dict[str, Any]:
        """
        Search for similar historical cases and return their report sentences.
        
        Args:
            state: Current VERIFAI state containing image_path
        
        Returns:
            Dict with retrieval results including:
                - retrieved_sentences: List of top sentences
                - consensus_diagnosis: Most common diagnosis in top results
                - support_count: How many top results agree
                - agrees_with_chexbert: Whether retrieval agrees with CheXbert
                - top_similarity: Highest similarity score
                - query_views_used: Which views were used for query
        """
        # Get all images for current study
        study_images = self._load_study_images(state["image_paths"])
        
        if not study_images:
            return {
                "retrieved_sentences": [],
                "consensus_diagnosis": "Unknown",
                "support_count": "0 out of 5",
                "agrees_with_chexbert": False,
                "top_similarity": 0.0,
                "query_views_used": [],
                "error": "No images found"
            }
        
        # Select 1-2 representative images (same logic as index building)
        query_images = self._select_study_images(study_images)
        
        # Embed each selected image
        query_embeddings = []
        for img_info in query_images:
            try:
                embedding = self._embed_image(img_info["path"])
                query_embeddings.append(embedding)
            except Exception as e:
                print(f"[Retrieval] Failed to embed {img_info['path']}: {e}")
                continue
        
        if not query_embeddings:
            return {
                "retrieved_sentences": [],
                "consensus_diagnosis": "Unknown",
                "support_count": "0 out of 5",
                "agrees_with_chexbert": False,
                "top_similarity": 0.0,
                "query_views_used": [],
                "error": "Failed to embed images"
            }
        
        # Average embeddings if we have 2 views
        if len(query_embeddings) == 2:
            query_embedding = np.mean(query_embeddings, axis=0)
        else:
            query_embedding = query_embeddings[0]
        
        # Prepare for FAISS search
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        
        # Search FAISS for top 10 similar sentences
        k = min(10, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k=k)
        
        # Collect results — handle either schema (with or without primary_label)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            entry = self.metadata[idx]
            sentence = entry["sentence"]
            # Use stored label if present; otherwise infer from sentence text
            label = entry.get("primary_label") or entry.get("label") or self._infer_label_from_sentence(sentence)
            results.append({
                "sentence": sentence,
                "label": label,
                "similarity": float(dist),
                "study_id": entry.get("study_id", "unknown")
            })
        
        # Find consensus diagnosis from top 5
        top_labels = [r["label"] for r in results[:5] if r["label"] != "Unknown"]
        
        if top_labels:
            consensus = Counter(top_labels).most_common(1)[0]
            consensus_label = consensus[0]
            consensus_count = consensus[1]
        else:
            consensus_label = "Unknown"
            consensus_count = 0
        
        # Check if retrieval agrees with CheXbert
        agrees = False
        chexbert_output = state.get("chexbert_output")
        if chexbert_output and hasattr(chexbert_output, 'labels'):
            chexbert_positives = [
                label for label, val in chexbert_output.labels.items()
                if val == "Positive"
            ]
            agrees = consensus_label in chexbert_positives
        
        return {
            "retrieved_sentences": [r["sentence"] for r in results[:5]],
            "consensus_diagnosis": consensus_label,
            "support_count": f"{consensus_count} out of 5",
            "agrees_with_chexbert": agrees,
            "top_similarity": results[0]["similarity"] if results else 0.0,
            "query_views_used": [img["view_position"] for img in query_images]
        }

