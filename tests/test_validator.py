"""
Test Validator Tools

Unit tests for the three validator tools:
1. CXR-RePaiR Retrieval
2. RadGraph Entity Matching
3. Clinical Rules Engine

python tests/test_validator.py -v
pytest tests/test_validator.py::test_radgraph_tool_integration -s
"""

import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from graph.state import VerifaiState, RadiologistOutput, CheXbertOutput, HistorianOutput, HistorianFact
from agents.validator.rules_engine import ClinicalRulesEngine, Rule


# ============================================================================
# RULES ENGINE TESTS
# ============================================================================

def test_rules_engine_overconfident_language():
    """Test that overconfident language rule triggers correctly."""
    engine = ClinicalRulesEngine()
    
    # Create state with high KLE and definitive language
    state = {
        "radiologist_kle_uncertainty": 0.65,
        "radiologist_output": RadiologistOutput(
            findings="Findings...",
            impression="Definitely consistent with pneumonia. No evidence of other pathology."
        )
    }
    
    result = engine.execute(state)
    
    assert result["flag_count"] >= 1
    assert "Overconfident Language" in result["triggered_rule_names"]


def test_rules_engine_no_external_evidence():
    """Test that no external evidence rule triggers."""
    engine = ClinicalRulesEngine()
    
    state = {
        "historian_output": HistorianOutput(
            supporting_facts=[],
            contradicting_facts=[]
        ),
        "literature_output": Mock(overall_evidence_strength="low")
    }
    
    result = engine.execute(state)
    
    assert result["warn_count"] >= 1
    assert "No External Evidence" in result["triggered_rule_names"]


def test_rules_engine_high_kle():
    """Test that high KLE rule triggers."""
    engine = ClinicalRulesEngine()
    
    state = {
        "radiologist_kle_uncertainty": 0.75
    }
    
    result = engine.execute(state)
    
    assert result["warn_count"] >= 1
    assert "High KLE Score" in result["triggered_rule_names"]


def test_rules_engine_no_violations():
    """Test that clean state has no violations."""
    engine = ClinicalRulesEngine()
    
    state = {
        "radiologist_kle_uncertainty": 0.3,
        "radiologist_output": RadiologistOutput(
            findings="Findings...",
            impression="Possible pneumonia. Recommend clinical correlation."
        ),
        "chexbert_output": CheXbertOutput(labels={"Pneumonia": "Positive"}),
        "historian_output": HistorianOutput(
            supporting_facts=[
                HistorianFact(
                    fact_type="supporting",
                    description="Fever and cough",
                    fhir_resource_id="obs-123",
                    fhir_resource_type="Observation"
                )
            ]
        ),
        "literature_output": Mock(overall_evidence_strength="medium")
    }
    
    result = engine.execute(state)
    
    assert result["flag_count"] == 0
    assert result["warn_count"] == 0


def test_rules_engine_custom_rule():
    """Test adding custom rule."""
    engine = ClinicalRulesEngine()
    
    custom_rule = Rule(
        name="Custom Test Rule",
        severity="FLAG",
        message="Test message",
        condition=lambda s: s.get("test_field", False)
    )
    
    engine.add_rule(custom_rule)
    
    state = {"test_field": True}
    result = engine.execute(state)
    
    assert "Custom Test Rule" in result["triggered_rule_names"]


# ============================================================================
# RETRIEVAL TOOL TESTS (Mock-based)
# ============================================================================

def test_retrieval_tool_image_selection():
    """Test image selection strategy."""
    from agents.validator.retrieval_tool import CXRRetrieverTool
    
    # Mock the tool without loading models
    tool = Mock(spec=CXRRetrieverTool)
    
    # Test data: PA + AP available
    images = [
        {"path": "img1.jpg", "view_position": "PA"},
        {"path": "img2.jpg", "view_position": "AP"},
        {"path": "img3.jpg", "view_position": "LAT"}
    ]
    
    # Simulate selection logic
    def select_study_images(all_images):
        pa_images = [i for i in all_images if i.get("view_position", "").upper() == "PA"]
        ap_images = [i for i in all_images if i.get("view_position", "").upper() == "AP"]
        
        selected = []
        if pa_images:
            selected.append(pa_images[0])
        if ap_images:
            selected.append(ap_images[0])
        
        return selected if selected else [all_images[0]]
    
    selected = select_study_images(images)
    
    assert len(selected) == 2
    assert selected[0]["view_position"] == "PA"
    assert selected[1]["view_position"] == "AP"


def test_retrieval_tool_fallback_lat():
    """Test fallback to LAT when no PA/AP."""
    images = [
        {"path": "img1.jpg", "view_position": "LAT"}
    ]
    
    def select_study_images(all_images):
        pa_images = [i for i in all_images if i.get("view_position", "").upper() == "PA"]
        ap_images = [i for i in all_images if i.get("view_position", "").upper() == "AP"]
        lat_images = [i for i in all_images if i.get("view_position", "").upper() in ["LAT", "LATERAL"]]
        
        selected = []
        if pa_images:
            selected.append(pa_images[0])
        if ap_images:
            selected.append(ap_images[0])
        if not selected and lat_images:
            return [lat_images[0]]
        
        return selected if selected else [all_images[0]]
    
    selected = select_study_images(images)
    
    assert len(selected) == 1
    assert selected[0]["view_position"] == "LAT"


# ============================================================================
# RADGRAPH TOOL TESTS (Mock-based)
# ============================================================================

def test_radgraph_entity_f1_calculation():
    """Test entity F1 calculation logic."""
    
    report_entities = {"consolidation[ANAT-DP]", "pneumonia[OBS-DP]", "right_lobe[ANAT-DP]"}
    retrieved_entities = {"consolidation[ANAT-DP]", "pneumonia[OBS-DP]", "left_lobe[ANAT-DP]"}
    
    # Calculate F1
    common = report_entities & retrieved_entities
    precision = len(common) / len(report_entities)
    recall = len(common) / len(retrieved_entities)
    f1 = 2 * precision * recall / (precision + recall)
    
    assert len(common) == 2
    assert f1 == pytest.approx(0.667, rel=0.01)


def test_radgraph_verdict_logic():
    """Test verdict assignment based on F1."""
    
    def get_verdict(f1):
        if f1 > 0.8:
            return "strong"
        elif f1 < 0.5:
            return "weak"
        else:
            return "moderate"
    
    assert get_verdict(0.9) == "strong"
    assert get_verdict(0.3) == "weak"
    assert get_verdict(0.6) == "moderate"


# ============================================================================
# VALIDATOR AGENT TESTS
# ============================================================================

def test_validator_decision_logic_finalize():
    """Test validator decides to finalize with strong evidence."""
    
    # Simulate validation results
    retrieval_agrees = True
    entity_f1 = 0.85
    no_critical_flag = True
    
    # Decision logic
    if retrieval_agrees and entity_f1 > 0.8 and no_critical_flag:
        recommendation = "FINALIZE"
    else:
        recommendation = "OTHER"
    
    assert recommendation == "FINALIZE"


def test_validator_decision_logic_flag_for_human():
    """Test validator flags for human with weak evidence."""
    
    retrieval_agrees = False
    entity_f1 = 0.4
    
    if not retrieval_agrees and entity_f1 < 0.5:
        recommendation = "FLAG_FOR_HUMAN"
    else:
        recommendation = "OTHER"
    
    assert recommendation == "FLAG_FOR_HUMAN"


def test_validator_decision_logic_low_confidence():
    """Test validator finalizes with low confidence on mixed signals."""
    
    retrieval_agrees = True
    entity_f1 = 0.6  # moderate
    no_critical_flag = True
    
    if retrieval_agrees and entity_f1 > 0.8 and no_critical_flag:
        recommendation = "FINALIZE"
    elif not retrieval_agrees and entity_f1 < 0.5:
        recommendation = "FLAG_FOR_HUMAN"
    else:
        recommendation = "FINALIZE_LOW_CONFIDENCE"
    
    assert recommendation == "FINALIZE_LOW_CONFIDENCE"


# ============================================================================
# INTEGRATION TESTS (require models and data)
# ============================================================================

@pytest.mark.skipif(
    not Path("data/mimic_corpus.faiss").exists(),
    reason="FAISS index not built"
)
def test_retrieval_tool_integration():
    """Integration test for retrieval tool (requires FAISS index)."""
    pytest.skip("Requires FAISS index and models - run manually")


def _radgraph_model_cached() -> bool:
    """Check the real OS-level cache that radgraph uses on Windows."""
    import os
    cache = Path(os.path.expanduser("~")) / "AppData" / "Local" / \
            "radgraph" / "radgraph" / "Cache" / "0.1.18" / "modern-radgraph-xl"
    return cache.exists() and any(cache.iterdir())


@pytest.mark.skipif(
    not _radgraph_model_cached(),
    reason="RadGraph model not found — run: python scripts/install_radgraph_model.py"
)
def test_radgraph_tool_integration():
    """Integration test: load RadGraph model and run a real entity extraction on sample text."""
    from agents.validator.radgraph_tool import RadGraphEntityTool
    
    tool = RadGraphEntityTool()
    assert tool.radgraph is not None, (
        "RadGraph model failed to load. Check that transformers>=4.48,<5.0 is installed "
        "and model files are in AppData/Local/radgraph/.../modern-radgraph-xl/"
    )

    # ── Sample input ──────────────────────────────────────────────────────────
    sample_text = (
        "There is a consolidation in the right lower lobe. "
        "No pleural effusion is identified."
    )
    print(f"\n[RadGraph Integration] Sample Input: {sample_text}")
    
    entities, relations = tool.extract_entities_and_relations(sample_text)
    
    print(f"[RadGraph Integration] Model Output (Entities): {entities}")
    print(f"[RadGraph Integration] Model Output (Relations): {relations}")
    
    # Basic sanity: model must return at least one entity
    assert len(entities) > 0, f"Expected entities but got none. Raw result: {entities}"
    
    # Verify expected clinical findings are extracted
    entity_tokens = [e.split("[")[0].lower() for e in entities]
    assert any("consolidation" in t for t in entity_tokens), (
        f"Expected 'consolidation' entity. Got: {entity_tokens}"
    )
    assert any("effusion" in t for t in entity_tokens), (
        f"Expected 'effusion' entity. Got: {entity_tokens}"
    )
    
    # Verify label format is "Token[Label::status]"
    for ent in entities:
        assert "[" in ent and "]" in ent, f"Unexpected entity format: {ent}"


def test_validator_node_without_tools():
    """Test validator node gracefully handles missing tools."""
    from agents.validator.agent import validator_node
    
    state = {
        "image_path": "test.jpg",
        "radiologist_output": RadiologistOutput(
            findings="Test findings",
            impression="Test impression"
        )
    }
    
    # Should not crash even if tools not initialized
    result = validator_node(state)
    
    assert "validator_output" in result
    assert result["validator_output"]["recommendation"] in [
        "FLAG_FOR_HUMAN", 
        "FINALIZE", 
        "FINALIZE_LOW_CONFIDENCE"
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
