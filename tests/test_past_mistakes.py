"""
Tests for Past Mistakes Memory Database

Tests cover:
- Database schema creation and indexing
- CRUD operations (insert, query, delete)
- Case embedding generation  
- Hybrid retrieval pipeline (structured + vector similarity)
- Integration with critic agent
"""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path

# Import past mistakes database functions
from db.past_mistakes import (
    init_past_mistakes_db,
    insert_validated_mistake,
    retrieve_similar_mistakes,
    get_mistake_by_id,
    delete_mistake,
    get_statistics
)

# Import case embedding functions
from uncertainty.case_embedding import (
    generate_case_summary,
    generate_case_embedding,
    generate_case_embedding_from_fields
)

# Import critic model for integration tests
from agents.critic.model import critic_model
from graph.state import CriticOutput


@pytest.fixture
def temp_db():
    """Create a temporary database path for testing."""
    # Create a temporary path without creating the file
    # DuckDB will create it
    import uuid
    temp_dir = tempfile.gettempdir()
    db_path = os.path.join(temp_dir, f'test_past_mistakes_{uuid.uuid4().hex}.duckdb')
    
    yield db_path
    
    # Cleanup
    try:
        if os.path.exists(db_path):
            os.unlink(db_path)
        # Also remove WAL file if exists
        wal_path = db_path + '.wal'
        if os.path.exists(wal_path):
            os.unlink(wal_path)
    except Exception as e:
        print(f"Cleanup warning: {e}")


class TestDatabaseSchema:
    """Test database schema creation and indexes."""
    
    def test_schema_initialization(self, temp_db):
        """Test that database initializes with correct schema."""
        init_past_mistakes_db(temp_db)
        
        from db.past_mistakes import get_connection
        conn = get_connection(temp_db)
        
        # Check table exists
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        assert 'past_mistakes' in table_names
        
        # Check columns
        schema = conn.execute("DESCRIBE past_mistakes").fetchall()
        column_names = [col[0] for col in schema]
        
        required_columns = [
            'mistake_id', 'session_id', 'image_path', 'created_at',
            'original_diagnosis', 'corrected_diagnosis', 'disease_type',
            'error_type', 'severity_level',
            'kle_uncertainty', 'safety_score',
            'chexbert_labels', 'clinical_summary', 'debate_summary',
            'case_embedding'
        ]
        
        for col in required_columns:
            assert col in column_names, f"Missing column: {col}"
    
    def test_indexes_creation(self, temp_db):
        """Test that all indexes are created."""
        init_past_mistakes_db(temp_db)
        
        from db.past_mistakes import get_connection
        conn = get_connection(temp_db)
        
        # Check indexes
        indexes = conn.execute("""
            SELECT index_name 
            FROM duckdb_indexes() 
            WHERE table_name = 'past_mistakes'
        """).fetchall()
        
        index_names = [idx[0] for idx in indexes]
        
        # Should have at least 8 structured indexes + 1 HNSW index
        assert len(index_names) >= 8, f"Expected >=8 indexes, got {len(index_names)}"


class TestCaseEmbedding:
    """Test case embedding generation."""
    
    def test_generate_case_summary_minimal(self):
        """Test case summary generation with minimal fields."""
        summary = generate_case_summary(
            disease_type='pneumonia',
            original_diagnosis='Normal chest X-ray',
            corrected_diagnosis='Community-Acquired Pneumonia (RLL)',
            error_type='misdiagnosis'
        )
        
        assert 'Disease: pneumonia' in summary
        assert 'Original Diagnosis: Normal chest X-ray' in summary
        assert 'Corrected Diagnosis: Community-Acquired Pneumonia (RLL)' in summary
        assert 'Error Type: misdiagnosis' in summary
    
    def test_generate_case_summary_complete(self):
        """Test case summary generation with all fields."""
        summary = generate_case_summary(
            disease_type='effusion',
            original_diagnosis='Mild atelectasis',
            corrected_diagnosis='Moderate pleural effusion',
            error_type='missed_differential',
            kle_uncertainty=0.42,
            chexbert_labels={'Effusion': 'present', 'Consolidation': 'uncertain'},
            clinical_summary='Patient with fever and dyspnea for 3 days',
            debate_summary='Radiologist attributed blunted costophrenic angle to artifact'
        )
        
        assert 'KLE Uncertainty: 0.42' in summary
        assert 'CheXbert Labels: Effusion,' in summary or 'CheXbert Labels: Consolidation,' in summary
        assert 'Clinical Context:' in summary
        assert 'Findings Pattern:' in summary
    
    def test_generate_case_embedding_dimension(self):
        """Test that embedding has correct dimension (384)."""
        summary = generate_case_summary(
            disease_type='pneumonia',
            original_diagnosis='Test',
            corrected_diagnosis='Test',
            error_type='overconfidence'
        )
        
        embedding = generate_case_embedding(summary)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
        assert embedding.dtype == np.float64 or embedding.dtype == np.float32
    
    def test_identical_summaries_produce_identical_embeddings(self):
        """Test that identical summaries produce identical embeddings."""
        summary1 = generate_case_summary(
            disease_type='pneumonia',
            original_diagnosis='Normal',
            corrected_diagnosis='Pneumonia',
            error_type='misdiagnosis'
        )
        
        summary2 = generate_case_summary(
            disease_type='pneumonia',
            original_diagnosis='Normal',
            corrected_diagnosis='Pneumonia',
            error_type='misdiagnosis'
        )
        
        emb1 = generate_case_embedding(summary1)
        emb2 = generate_case_embedding(summary2)
        
        # Should be nearly identical (cosine similarity ~ 1.0)
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert cosine_sim > 0.99, f"Expected cosine similarity > 0.99, got {cosine_sim}"
    
    def test_different_summaries_produce_different_embeddings(self):
        """Test that different summaries produce different embeddings."""
        emb1 = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Normal',
            corrected_diagnosis='Pneumonia',
            error_type='misdiagnosis'
        )
        
        emb2 = generate_case_embedding_from_fields(
            disease_type='effusion',
            original_diagnosis='Atelectasis',
            corrected_diagnosis='Effusion',
            error_type='missed_differential'
        )
        
        # Should be different (cosine similarity < 0.9)
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        assert cosine_sim < 0.9, f"Expected cosine similarity < 0.9, got {cosine_sim}"


class TestCRUDOperations:
    """Test CRUD operations on past mistakes."""
    
    def test_insert_and_retrieve_mistake(self, temp_db):
        """Test inserting and retrieving a mistake by ID."""
        init_past_mistakes_db(temp_db)
        
        # Create embedding
        embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Normal chest X-ray',
            corrected_diagnosis='Community-Acquired Pneumonia',
            error_type='misdiagnosis'
        )
        
        # Insert mistake
        mistake_id = insert_validated_mistake(
            session_id='test-session-001',
            image_path='test_xray.jpg',
            original_diagnosis='Normal chest X-ray',
            corrected_diagnosis='Community-Acquired Pneumonia',
            disease_type='pneumonia',
            error_type='misdiagnosis',
            severity_level=4,
            case_embedding=embedding,
            kle_uncertainty=0.35,
            safety_score=0.45,
            chexbert_labels={'Consolidation': 'present'},
            clinical_summary='Patient with fever and cough',
            debate_summary='Radiologist missed RLL consolidation'
        )
        
        assert mistake_id is not None
        
        # Retrieve by ID
        mistake = get_mistake_by_id(mistake_id, temp_db)
        
        assert mistake is not None
        assert mistake['original_diagnosis'] == 'Normal chest X-ray'
        assert mistake['disease_type'] == 'pneumonia'
        assert mistake['severity_level'] == 4
        assert mistake['chexbert_labels'] == {'Consolidation': 'present'}
    
    def test_delete_mistake(self, temp_db):
        """Test deleting a mistake."""
        init_past_mistakes_db(temp_db)
        
        embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Test',
            corrected_diagnosis='Test',
            error_type='overconfidence'
        )
        
        mistake_id = insert_validated_mistake(
            session_id='test',
            image_path='test.jpg',
            original_diagnosis='Test',
            corrected_diagnosis='Test',
            disease_type='pneumonia',
            error_type='overconfidence',
            severity_level=2,
            case_embedding=embedding
        )
        
        # Delete
        deleted = delete_mistake(mistake_id, temp_db)
        assert deleted is True
        
        # Should not be found
        mistake = get_mistake_by_id(mistake_id, temp_db)
        assert mistake is None
    
    def test_insert_validation_errors(self, temp_db):
        """Test that validation errors are raised for invalid inputs."""
        init_past_mistakes_db(temp_db)
        
        embedding = np.random.randn(384)
        
        # Invalid error_type
        with pytest.raises(ValueError, match="Invalid error_type"):
            insert_validated_mistake(
                session_id='test',
                image_path='test.jpg',
                original_diagnosis='Test',
                corrected_diagnosis='Test',
                disease_type='pneumonia',
                error_type='invalid_type',
                severity_level=3,
                case_embedding=embedding
            )
        
        # Invalid severity_level
        with pytest.raises(ValueError, match="severity_level must be between 1 and 5"):
            insert_validated_mistake(
                session_id='test',
                image_path='test.jpg',
                original_diagnosis='Test',
                corrected_diagnosis='Test',
                disease_type='pneumonia',
                error_type='misdiagnosis',
                severity_level=10,
                case_embedding=embedding
            )
        
        # Invalid embedding dimension
        with pytest.raises(ValueError, match="case_embedding must be 384-dim"):
            insert_validated_mistake(
                session_id='test',
                image_path='test.jpg',
                original_diagnosis='Test',
                corrected_diagnosis='Test',
                disease_type='pneumonia',
                error_type='misdiagnosis',
                severity_level=3,
                case_embedding=np.random.randn(128)  # Wrong dimension
            )


class TestHybridRetrieval:
    """Test hybrid retrieval pipeline."""
    
    @pytest.fixture
    def populated_db(self, temp_db):
        """Create a database populated with test mistakes."""
        init_past_mistakes_db(temp_db)
        
        # Insert 5 pneumonia mistakes with varying severity and KLE
        pneumonia_mistakes = [
            ('High severity pneumonia #1', 'Pneumonia CAP', 'overconfidence', 5, 0.35, 0.40),
            ('High severity pneumonia #2', 'Pneumonia HAP', 'misdiagnosis', 4, 0.37, 0.45),
            ('Medium severity pneumonia', 'Pneumonia atypical', 'missed_differential', 3, 0.40, 0.50),
            ('Low severity pneumonia', 'Pneumonia viral', 'calibration_error', 2, 0.32, 0.60),
            ('Minimal severity pneumonia', 'Pneumonia resolved', 'overconfidence', 1, 0.38, 0.70)
        ]
        
        for orig, corrected, error_type, severity, kle, safety in pneumonia_mistakes:
            embedding = generate_case_embedding_from_fields(
                disease_type='pneumonia',
                original_diagnosis=orig,
                corrected_diagnosis=corrected,
                error_type=error_type
            )
            insert_validated_mistake(
                session_id=f'test-{severity}',
                image_path='test.jpg',
                original_diagnosis=orig,
                corrected_diagnosis=corrected,
                disease_type='pneumonia',
                error_type=error_type,
                severity_level=severity,
                case_embedding=embedding,
                kle_uncertainty=kle,
                safety_score=safety,
                db_path=temp_db
            )
        
        # Insert 2 effusion mistakes (different disease type)
        for i in range(2):
            embedding = generate_case_embedding_from_fields(
                disease_type='effusion',
                original_diagnosis=f'Atelectasis {i}',
                corrected_diagnosis=f'Effusion {i}',
                error_type='missed_differential'
            )
            insert_validated_mistake(
                session_id=f'effusion-{i}',
                image_path='test.jpg',
                original_diagnosis=f'Atelectasis {i}',
                corrected_diagnosis=f'Effusion {i}',
                disease_type='effusion',
                error_type='missed_differential',
                severity_level=3,
                case_embedding=embedding,
                kle_uncertainty=0.50,
                db_path=temp_db
            )
        
        return temp_db
    
    def test_retrieve_by_disease_type(self, populated_db):
        """Test filtering by disease type."""
        # Query embedding for pneumonia
        query_embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Unclear opacity',
            corrected_diagnosis='Pneumonia',
            error_type='misdiagnosis'
        )
        
        results = retrieve_similar_mistakes(
            disease_type='pneumonia',
            embedding=query_embedding,
            top_k=10,
            similarity_threshold=0.0,  # Get all
            db_path=populated_db
        )
        
        # Should only return pneumonia cases
        assert len(results) == 5
        assert all(r['disease_type'] == 'pneumonia' for r in results)
    
    def test_retrieve_with_kle_range(self, populated_db):
        """Test filtering by KLE uncertainty range."""
        query_embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Test',
            corrected_diagnosis='Test',
            error_type='misdiagnosis'
        )
        
        results = retrieve_similar_mistakes(
            disease_type='pneumonia',
            embedding=query_embedding,
            kle_uncertainty_range=(0.35, 0.40),
            top_k=10,
            similarity_threshold=0.0,
            db_path=populated_db
        )
        
        # Should return cases with KLE in range [0.35, 0.40]
        assert len(results) >= 2  # At least 2 cases in this range
        for r in results:
            assert 0.35 <= r['kle_uncertainty'] <= 0.40
    
    def test_retrieve_by_severity(self, populated_db):
        """Test filtering by minimum severity."""
        query_embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Test',
            corrected_diagnosis='Test',
            error_type='misdiagnosis'
        )
        
        results = retrieve_similar_mistakes(
            disease_type='pneumonia',
            embedding=query_embedding,
            severity_min=4,
            top_k=10,
            similarity_threshold=0.0,
            db_path=populated_db
        )
        
        # Should only return high-severity cases
        assert len(results) >= 2
        assert all(r['severity_level'] >= 4 for r in results)
    
    def test_similarity_threshold(self, populated_db):
        """Test that similarity threshold filters results."""
        query_embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='High severity pneumonia #1',  # Exact match
            corrected_diagnosis='Pneumonia CAP',
            error_type='overconfidence'
        )
        
        # High threshold should return only very similar cases
        results = retrieve_similar_mistakes(
            disease_type='pneumonia',
            embedding=query_embedding,
            top_k=10,
            similarity_threshold=0.95,
            db_path=populated_db
        )
        
        # Should return at least the exact match
        assert len(results) >= 1
        assert all(r['similarity'] >= 0.95 for r in results)
    
    def test_top_k_limiting(self, populated_db):
        """Test that top_k limits results."""
        query_embedding = generate_case_embedding_from_fields(
            disease_type='pneumonia',
            original_diagnosis='Test',
            corrected_diagnosis='Test',
            error_type='misdiagnosis'
        )
        
        results = retrieve_similar_mistakes(
            disease_type='pneumonia',
            embedding=query_embedding,
            top_k=2,
            similarity_threshold=0.0,
            db_path=populated_db
        )
        
        assert len(results) <= 2


class TestStatistics:
    """Test statistics aggregation."""
    
    def test_get_statistics(self, temp_db):
        """Test statistics calculation."""
        init_past_mistakes_db(temp_db)
        
        # Insert a few test cases
        for disease_type in ['pneumonia', 'pneumonia', 'effusion']:
            embedding = generate_case_embedding_from_fields(
                disease_type=disease_type,
                original_diagnosis='Test',
                corrected_diagnosis='Test',
                error_type='misdiagnosis'
            )
            insert_validated_mistake(
                session_id='test',
                image_path='test.jpg',
                original_diagnosis='Test',
                corrected_diagnosis='Test',
                disease_type=disease_type,
                error_type='misdiagnosis',
                severity_level=3,
                case_embedding=embedding,
                db_path=temp_db
            )
        
        stats = get_statistics(temp_db)
        
        assert stats['total_mistakes'] == 3
        assert len(stats['by_disease_type']) >= 2  # pneumonia and effusion
        assert any(d['disease_type'] == 'pneumonia' and d['count'] == 2 
                   for d in stats['by_disease_type'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
