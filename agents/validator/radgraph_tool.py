"""
RadGraph Entity Matching Tool

Extract structured clinical entities and relations from radiology text using
the official RadGraph Python package.

Prerequisites:
- pip install radgraph
- Run: python scripts/install_radgraph_model.py  (first-time setup)
"""

import sys
from typing import TYPE_CHECKING, Dict, Any, Tuple, Set, List
from transformers import BertTokenizer
try:
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
except ImportError:
    TokenizersBackend = None

# Monkey-patch for transformers 5.x compatibility
for cls in [BertTokenizer, TokenizersBackend]:
    if cls and not hasattr(cls, "encode_plus"):
        def encode_plus_patch(self, *args, **kwargs):
            if args and isinstance(args[0], list):
                return {"input_ids": self.convert_tokens_to_ids(args[0])}
            return self(*args, **kwargs)
        cls.encode_plus = encode_plus_patch
        print(f"[RadGraph] Patched {cls.__name__}.encode_plus for transformers 5.x compatibility")

if TYPE_CHECKING:
    from graph.state import VerifaiState

# torch 2.x removed _cast_Long from torch.onnx.symbolic_opset9 but
# radgraph's bundled allennlp still imports it at model-load time.
# We restore it here before any radgraph import so it is available.
try:
    import torch as _torch
    import torch.onnx.symbolic_opset9  # ensure it is in sys.modules
    _opset9 = sys.modules["torch.onnx.symbolic_opset9"]
    if not hasattr(_opset9, "_cast_Long"):
        def _cast_Long(g, input, non_blocking):
            return g.op("Cast", input, to_i=_torch.onnx.TensorProtoDataType.INT64)
        _opset9._cast_Long = _cast_Long
except Exception:
    pass

try:
    from radgraph import RadGraph
    RADGRAPH_AVAILABLE = True
except ImportError:
    RADGRAPH_AVAILABLE = False
    RadGraph = None  # type: ignore
    # print("[RadGraph] Warning: radgraph not installed. RadGraph tool will be disabled.")
    # print("[RadGraph] Install with: pip install radgraph")


class RadGraphEntityTool:
    """
    Extract and compare clinical entities using RadGraph model.
    
    RadGraph extracts:
    - Entities: anatomical structures, observations, etc.
    - Relations: spatial, severity, temporal relationships
    """
    
    def __init__(self, model_type: str = "modern-radgraph-xl"):
        """
        Args:
            model_type: RadGraph model variant to use (default: modern-radgraph-xl)
        """
        self.radgraph = None
        self.model_type = model_type
        
        if not RADGRAPH_AVAILABLE:
            print(f"[RadGraph] Skipping model load - radgraph package not available")
            return
        
        try:
            from app.config import settings
            import torch
            cache_dir = getattr(settings, "RADGRAPH_CACHE_DIR", None)

            # Verify the model directory actually exists at that cache path
            if cache_dir:
                from pathlib import Path as _Path
                model_dir = _Path(cache_dir) / model_type
                if not model_dir.exists():
                    print(f"[RadGraph] WARNING: Expected model dir not found at {model_dir}")
                    print(f"[RadGraph]   Run: python scripts/install_radgraph_model.py --tar <path>")
                    cache_dir = None   # let radgraph use its own CACHE_DIR / download

            cuda = 0 if torch.cuda.is_available() else -1

            kwargs = dict(model_type=model_type, cuda=cuda)
            if cache_dir:
                kwargs["model_cache_dir"] = cache_dir
                print(f"[RadGraph] Loading {model_type} from local cache: {cache_dir}")
            else:
                print(f"[RadGraph] Loading {model_type} (auto-downloading from HuggingFace if needed)...")

            self.radgraph = RadGraph(**kwargs)  # type: ignore
            print(f"[RadGraph] Model loaded successfully")
        except Exception as e:
            print(f"[RadGraph] Failed to load model: {e}")
            self.radgraph = None

    
    def extract_entities_and_relations(
        self, 
        text: str
    ) -> Tuple[Set[str], Set[str]]:
        """
        Extract structured entities and relations from radiology text.
        
        Args:
            text: Radiology report text
        
        Returns:
            entities: Set of strings like "consolidation[ANAT-DP]"
            relations: Set of strings like "consolidation[located_at]right_lower_lobe"
        """
        if not self.radgraph:
            # Return empty sets if model not available
            return set(), set()
        
        if not text.strip():
            return set(), set()
        
        try:
            # Run RadGraph annotation
            # Returns a dict like {"0": {"entities": {...}, "text": ..., ...}}
            raw = self.radgraph([text])
        except Exception as e:
            print(f"[RadGraph] Prediction failed: {e}")
            return set(), set()
        
        entities = set()
        relations = set()
        
        if not raw:
            return entities, relations
        
        # Get the first annotation (keyed as "0")
        annotation = next(iter(raw.values()))
        if not isinstance(annotation, dict):
            return entities, relations
        
        # Extract entities — label format: "Observation::definitely present"
        ent_map = annotation.get("entities", {})
        for entity_id, entity_data in ent_map.items():
            entity_text = entity_data.get("tokens", "")
            entity_label = entity_data.get("label", "")
            if entity_text and entity_label:
                entities.add(f"{entity_text}[{entity_label}]")
        
        # Extract relations — stored inside each entity's "relations" list
        # Format: [["relation_type", "target_entity_id"], ...]
        for entity_id, entity_data in ent_map.items():
            source_text = entity_data.get("tokens", "")
            for rel in entity_data.get("relations", []):
                if len(rel) == 2:
                    rel_type, target_id = rel
                    target_text = ent_map.get(str(target_id), {}).get("tokens", "")
                    if source_text and target_text and rel_type:
                        relations.add(f"{source_text}[{rel_type}]{target_text}")
        
        return entities, relations
    
    def execute(
        self, 
        state: "VerifaiState", 
        retrieved_sentences: List[str]
    ) -> Dict[str, Any]:
        """
        Compare entities between generated report and retrieved sentences.
        
        Args:
            state: Current VERIFAI state
            retrieved_sentences: List of sentences from retrieval tool
        
        Returns:
            Dict with:
                - entity_f1: F1 score for entity matching
                - relation_f1: F1 score for relation matching
                - matched_entities: List of common entities
                - unmatched_in_report: Entities in report but not retrieved
                - verdict: "strong", "moderate", or "weak" match
        """
        if not self.radgraph:
            # Return default values if model not available
            return {
                "entity_f1": 0.0,
                "relation_f1": 0.0,
                "matched_entities": [],
                "unmatched_in_report": [],
                "verdict": "unavailable",
                "error": "RadGraph model not loaded"
            }
        
        # Extract from generated report
        radiologist_output = state.get("radiologist_output")
        if not radiologist_output:
            return {
                "entity_f1": 0.0,
                "relation_f1": 0.0,
                "matched_entities": [],
                "unmatched_in_report": [],
                "verdict": "no_report",
                "error": "No radiologist output available"
            }
        
        report_text = (
            radiologist_output.findings + " " +
            radiologist_output.impression
        )
        report_entities, report_relations = self.extract_entities_and_relations(report_text)
        
        # Extract from retrieved historical sentences
        retrieved_text = " ".join(retrieved_sentences) if retrieved_sentences else ""
        retrieved_entities, retrieved_relations = self.extract_entities_and_relations(retrieved_text)
        
        # Compute entity F1
        common_entities = report_entities & retrieved_entities
        
        if len(report_entities) > 0:
            ent_precision = len(common_entities) / len(report_entities)
        else:
            ent_precision = 0.0
        
        if len(retrieved_entities) > 0:
            ent_recall = len(common_entities) / len(retrieved_entities)
        else:
            ent_recall = 0.0
        
        if (ent_precision + ent_recall) > 0:
            ent_f1 = 2 * ent_precision * ent_recall / (ent_precision + ent_recall)
        else:
            ent_f1 = 0.0
        
        # Compute relation F1
        common_relations = report_relations & retrieved_relations
        
        if (len(report_relations) + len(retrieved_relations)) > 0:
            rel_f1 = (
                2 * len(common_relations) / 
                (len(report_relations) + len(retrieved_relations))
            )
        else:
            rel_f1 = 0.0
        
        # Find unmatched entities (in report but not in retrieved)
        unmatched = report_entities - retrieved_entities
        
        # Determine verdict
        if ent_f1 > 0.8:
            verdict = "strong"
        elif ent_f1 < 0.5:
            verdict = "weak"
        else:
            verdict = "moderate"
        
        return {
            "entity_f1": round(ent_f1, 3),
            "relation_f1": round(rel_f1, 3),
            "matched_entities": list(common_entities),
            "unmatched_in_report": list(unmatched),
            "verdict": verdict,
            "report_entity_count": len(report_entities),
            "retrieved_entity_count": len(retrieved_entities)
        }
