"""
Tests for Enhancement Features

Tests cover:
- REST API endpoints (CRUD operations)
- Automatic mistake detection (severity scoring, error classification)
- Neural re-ranking (temporal recency, clinical relevance)
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import tempfile
import os

# Import FastAPI app
from app.main import app

# Import modules to test
from db.auto_detect_mistakes import (
    calculate_severity_score,
    classify_error_type,
    detect_mistake,
    generate_mistake_summary
)
from db.rerank_mistakes import (
    calculate_recency_weight,
    calculate_clinical_relevance,
    rerank_mistakes
)
from graph.state import VerifaiState, RadiologistOutput, CriticOutput


client = TestClient(app)


class TestPastMistakesAPI:
    """Test REST API endpoints."""
    
    def test_create_mistake(self):
        """Test POST /api/past-mistakes."""
        payload = {
            "session_id": "api-test-001",
            "image_path": "test.jpg",
            "original_diagnosis": "Normal chest X-ray",
            "corrected_diagnosis": "Pneumonia",
            "disease_type": "pneumonia",
            "error_type": "misdiagnosis",
            "severity_level": 4,
            "kle_uncertainty": 0.35,
            "safety_score": 0.45
        }
        
        response = client.post("/api/past-mistakes", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert "mistake_id" in data
        assert data["message"] == "Mistake successfully recorded"
    
    def test_get_stats(self):
        """Test GET /api/past-mistakes/stats/summary."""
        response = client.get("/api/past-mistakes/stats/summary")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_mistakes" in data
        assert "by_disease_type" in data
        assert "by_error_type" in data


class TestAutomaticDetection:
    """Test automatic mistake detection."""
    
    def test_severity_scoring(self):
        """Test severity score calculation."""
        # High severity: high uncertainty + low safety + critical pathology
        severity = calculate_severity_score(
            initial_diagnosis="Normal",
            final_diagnosis="Pneumothorax",
            kle_uncertainty=0.6,
            critic_safety_score=0.3,
            clinical_outcome="Emergent intervention required"
        )
        assert severity >= 4
        
        # Low severity: low uncertainty + high safety
        severity = calculate_severity_score(
            initial_diagnosis="Minimal atelectasis",
            final_diagnosis="Mild atelectasis",
            kle_uncertainty=0.2,
            critic_safety_score=0.8,
            clinical_outcome="Resolved without treatment"
        )
        assert severity <= 2
    
    def test_error_classification(self):
        """Test error type classification."""
        # Overconfidence
        error_type = classify_error_type(
            initial_diagnosis="Clearly shows pneumonia",
            final_diagnosis="Pulmonary edema",
            kle_uncertainty=0.5,
            critic_was_overconfident=True
        )
        assert error_type == "overconfidence"
        
        # Misdiagnosis
        error_type = classify_error_type(
            initial_diagnosis="Normal chest X-ray",
            final_diagnosis="Pneumonia with consolidation",
            kle_uncertainty=0.3,
            critic_was_overconfident=False
        )
        assert error_type == "misdiagnosis"
    
    def test_detect_mistake_with_discrepancy(self):
        """Test mistake detection with clear discrepancy."""
        state = {
            "radiologist_output": RadiologistOutput(
                findings="Clear lungs, no consolidation",
                impression="Normal chest X-ray",
                chexbert_labels={}
            ),
            "critic_output": CriticOutput(
                is_overconfident=True,
                concern_flags=["[HIGH] Certainty phrases"],
                recommended_hedging=None,
                safety_score=0.4
            ),
            "radiologist_kle_uncertainty": 0.45
        }
        
        mistake = detect_mistake(state, final_validated_diagnosis="Community-Acquired Pneumonia")
        
        assert mistake is not None
        assert mistake['original_diagnosis'] == "Normal chest X-ray"
        assert mistake['corrected_diagnosis'] == "Community-Acquired Pneumonia"
        assert mistake['error_type'] in ["misdiagnosis", "overconfidence"]
        assert 1 <= mistake['severity_level'] <= 5
    
    def test_detect_mistake_no_discrepancy(self):
        """Test mistake detection with no discrepancy."""
        state = {
            "radiologist_output": RadiologistOutput(
                findings="Consolidation in RLL",
                impression="Community-Acquired Pneumonia",
                chexbert_labels={}
            ),
            "radiologist_kle_uncertainty": 0.25
        }
        
        mistake = detect_mistake(state, final_validated_diagnosis="Community-Acquired Pneumonia")
        
        assert mistake is None


class TestNeuralReranking:
    """Test neural re-ranking algorithms."""
    
    def test_recency_weight(self):
        """Test temporal recency weighting."""
        now = datetime.now()
        
        # Recent mistake (1 week ago)
        recent = now - timedelta(days=7)
        weight_recent = calculate_recency_weight(recent, decay_days=180)
        
        # Old mistake (1 year ago)
        old = now - timedelta(days=365)
        weight_old = calculate_recency_weight(old, decay_days=180)
        
        # Recent should have higher weight
        assert weight_recent > weight_old
        assert 0.8 < weight_recent <= 1.0
        assert 0.1 <= weight_old < 0.5
    
    def test_clinical_relevance(self):
        """Test clinical relevance scoring."""
        # High relevance: similar uncertainty + high severity + matching terms
        relevance = calculate_clinical_relevance(
            current_impression="Possible pneumonia, uncertain consolidation",
            current_kle=0.40,
            current_chexbert={"Consolidation": "uncertain"},
            mistake_original_diagnosis="Unclear opacity",
            mistake_corrected_diagnosis="Pneumonia with consolidation",
            mistake_kle=0.42,
            mistake_error_type="misdiagnosis",
            mistake_severity=4
        )
        assert relevance > 0.5
        
        # Low relevance: different disease, different uncertainty
        relevance = calculate_clinical_relevance(
            current_impression="Pleural effusion",
            current_kle=0.20,
            current_chexbert={"Effusion": "present"},
            mistake_original_diagnosis="Pneumothorax",
            mistake_corrected_diagnosis="Normal",
            mistake_kle=0.70,
            mistake_error_type="overconfidence",
            mistake_severity=1
        )
        assert relevance < 0.5
    
    def test_rerank_mistakes(self):
        """Test full re-ranking algorithm."""
        now = datetime.now()
        
        # Mock retrieved mistakes
        mistakes = [
            {
                'mistake_id': '1',
                'original_diagnosis': 'Normal',
                'corrected_diagnosis': 'Pneumonia',
                'error_type': 'misdiagnosis',
                'severity_level': 5,
                'kle_uncertainty': 0.38,
                'created_at': now - timedelta(days=10),  # Recent
                'similarity': 0.85
            },
            {
                'mistake_id': '2',
                'original_diagnosis': 'Atelectasis',
                'corrected_diagnosis': 'Effusion',
                'error_type': 'missed_differential',
                'severity_level': 2,
                'kle_uncertainty': 0.60,
                'created_at': now - timedelta(days=500),  # Old
                'similarity': 0.90  # Higher vector similarity
            }
        ]
        
        reranked = rerank_mistakes(
            current_impression="Possible pneumonia",
            current_kle=0.40,
            current_chexbert={"Consolidation": "uncertain"},
            retrieved_mistakes=mistakes,
            recency_weight_factor=0.4,
            clinical_relevance_factor=0.4,
            feedback_factor=0.2
        )
        
        # Check re-ranking occurred
        assert len(reranked) == 2
        assert all('rerank_score' in m for m in reranked)
        assert all('recency_score' in m for m in reranked)
        assert all('clinical_relevance_score' in m for m in reranked)
        
        # First mistake should rank higher despite lower vector similarity
        # (high severity + recent + clinically relevant)
        assert reranked[0]['mistake_id'] == '1'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
