# Validator Agent - Implementation Guide

## Overview

The Validator Agent aggregates ALL agent outputs and runs three validation tools to prepare a comprehensive final answer for human review.

**When Validator Runs (2 scenarios):**
1. ✅ **Consensus Reached**: Debate achieves consensus → Validator validates the consensus
2. ⚠️ **Max Rounds Exceeded**: Debate runs for max rounds without consensus → Validator escalates with evidence

**Three Validation Tools:**

1. **CXR-RePaiR Retrieval Tool**: Finds similar historical cases from MIMIC-CXR
2. **RadGraph Entity Matching Tool**: Verifies clinical facts against retrieved cases
3. **Clinical Rules Engine**: Checks for contradictions and inconsistencies

**Output**: Comprehensive aggregation of all agent outputs + validation tool results

## Files Created

```
agents/validator/
├── __init__.py              # Module exports
├── agent.py                 # Main validator node
├── retrieval_tool.py        # CXR-RePaiR retrieval
├── radgraph_tool.py         # RadGraph entity matching
└── rules_engine.py          # Clinical rules engine

scripts/
├── build_retrieval_index.py # Script to build FAISS index
└── install_radgraph_model.py # Manually install RadGraph model

graph/
└── state.py                 # Updated with validator_output field
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install faiss-cpu==1.7.4 radgraph nltk==3.8.1
```

For GPU support, use `faiss-gpu` instead of `faiss-cpu`.

Download NLTK punkt tokenizer:
```python
import nltk
nltk.download('punkt')
```

### 2. Manual RadGraph Installation (Required)

Automatic downloading of the model is often unreliable. Manual setup is **highly recommended**:

1.  **Download**: Get `modern-radgraph-xl.tar.gz` (approx 1.2GB) from [HuggingFace](https://huggingface.co/StanfordAIMI/RRG_scorers/blob/main/modern-radgraph-xl.tar.gz).
2.  **Placement**: Place the downloaded file directly in the project root directory (`VERIFAI/`).
3.  **Install**: Run the installation script to extract the model to the correct system cache:
    ```bash
    python scripts/install_radgraph_model.py
    ```

### 3. Build Retrieval Index
This step requires access to the MIMIC-CXR dataset.

```bash
python scripts/build_retrieval_index.py \
    2-mimic_root /path/to/mimic-cxr \
    --output_dir data/ \
    --num_studies 1000  # Optional: limit for testing
```

**Expected outputs:**
- `data/mimic_corpus.faiss` (FAISS index)
- `data/mimic_corpus_metadata.json` (sentence metadata)

**Time estimate:** ~1-2 hours for full MIMIC-CXR training set

## Usage

### Initializing the Validator

```python
from agents.validator import initialize_validator_tools
from agents.radiologist.model import _load_models

# Load radiologist models (includes vision encoder)
_load_models()

# Initialize validator tools
initialize_validator_tools(
    vision_encoder=_vision_encoder,
    image_processor=_image_processor
)
```

### Running the Validator Node

The validator automatically runs when debate fails to reach consensus:

```python
from agents.validator import validator_node

# Validator receives state after debate
result = validator_node(state)

# Check recommendation
print(result["validator_output"]["recommendation"])
# Possible values: "FINALIZE", "FLAG_FOR_HUMAN", "FINALIZE_LOW_CONFIDENCE"
```

### Validator Output Structure

The validator aggregates ALL agent outputs plus its own tool results:

```python
{
    "recommendation": "FINALIZE",  # or "FLAG_FOR_HUMAN", "FINALIZE_LOW_CONFIDENCE"
    "confidence_level": "high",    # "high", "medium", "low"
    "explanation": "All validation tools show strong agreement",
    
    # === VALIDATION TOOLS ===
    "retrieval": {
        "retrieved_sentences": [...],
        "consensus_diagnosis": "Pneumonia",
        "support_count": "4 out of 5",
        "agrees_with_chexbert": True,
        "top_similarity": 0.85,
        "query_views_used": ["PA", "AP"]
    },
    
    "entity_matching": {
        "entity_f1": 0.82,
        "relation_f1": 0.75,
        "matched_entities": ["consolidation[ANAT-DP]", ...],
        "unmatched_in_report": [...],
        "verdict": "strong"
    },
    
    "rules": {
        "rules_triggered": [...],
        "flag_count": 0,
        "warn_count": 1,
        "has_critical_flag": False,
        "summary": "0 flags, 1 warnings"
    },
    
    # === ALL AGENT OUTPUTS SUMMARY ===
    "agent_summary": {
        "radiologist": {
            "findings": "...",
            "impression": "...",
            "kle_uncertainty": 0.45
        },
        "chexbert": {
            "positive_labels": ["Pneumonia"],
            "uncertain_labels": []
        },
        "critic": {
            "is_overconfident": False,
            "concern_count": 1,
            "safety_score": 0.85
        },
        "historian": {
            "supporting_facts_count": 3,
            "contradicting_facts_count": 0
        },
        "literature": {
            "evidence_strength": "medium",
            "citation_count": 5
        },
        "debate": {
            "consensus_reached": True,
            "rounds": 2,
            "escalated": False
        }
    },
    
    # === FINAL VERDICT FOR HUMAN ===
    "final_verdict": {
        "recommendation": "FINALIZE",
        "confidence": "high",
        "reasoning": "All agents agree with strong external evidence",
        "all_agents_agree": True,
        "external_evidence_strength": "strong",
        "key_concerns": []
    },
    
    "summary": {
        "historical_support": "4 out of 5",
        "entity_match_f1": 0.82,
        "flags": 0,
        "warnings": 1
    }
}
```

## Validation Decision Logic

The validator uses the following decision tree:

```
IF retrieval agrees with CheXbert 
   AND entity F1 > 0.8 
   AND no critical flags
   → FINALIZE (high confidence)

ELIF retrieval disagrees 
     AND entity F1 < 0.5
   → FLAG_FOR_HUMAN (weak evidence)

ELIF has critical rule violation
   → FLAG_FOR_HUMAN

ELSE
   → FINALIZE_LOW_CONFIDENCE (mixed signals)
```

## Clinical Rules

The rules engine checks for:

1. **Overconfident Language** (FLAG): High KLE uncertainty (>0.6) but definitive language
2. **Spatial Contradiction** (FLAG): Diagnosis region mismatch with Grad-CAM
3. **No External Evidence** (WARN): Neither FHIR nor literature support
4. **Pneumonia Normal Labs** (WARN): Pneumonia with normal WBC
5. **Diffuse Heatmap Lobar Disease** (WARN): Lobar pneumonia with diffuse activation
6. **High KLE Score** (WARN): Very high uncertainty (>0.7)
7. **Debate No Consensus** (FLAG): Debate failed after max rounds
8. **Critic High Concern** (WARN): Multiple critic flags
9. **Historical Mistakes Similar** (FLAG): Similar to past mistakes

## Testing

### Run Integration Tests
To verify RadGraph and the Validator are working correctly:

```bash
# Run all validator tests
pytest tests/test_validator.py -v

# Run with output capture (to see extracted entities)
pytest tests/test_validator.py::test_radgraph_tool_integration -s
```

### Test Individual Tools

```python
# Test retrieval tool
from agents.validator.retrieval_tool import CXRRetrieverTool

retriever = CXRRetrieverTool(
    index_path="data/mimic_corpus.faiss",
    metadata_path="data/mimic_corpus_metadata.json",
    vision_encoder=vision_encoder,
    image_processor=image_processor
)

result = retriever.execute(state)
print(f"Found {len(result['retrieved_sentences'])} similar sentences")
```

```python
# Test RadGraph tool
from agents.validator.radgraph_tool import RadGraphEntityTool

# Requires manual installation via scripts/install_radgraph_model.py
radgraph = RadGraphEntityTool(model_type="modern-radgraph-xl") 
entities, relations = radgraph.extract_entities_and_relations(text)
```

### Validator Integration

The validator is integrated into the workflow to run in TWO cases:
1. When debate reaches consensus (validation)
2. When debate hits max rounds without consensus (escalation)

```python
# In graph/workflow.py
from agents.validator import validator_node, initialize_validator_tools

# Initialize during graph build
initialize_validator_tools(vision_encoder, image_processor)

# Add node
graph.add_node("validator", validator_node)

# Route after debate (BOTH scenarios)
def route_after_debate(state):
    # ALWAYS go to validator for comprehensive validation
    return "validator"

# After validator decision
def route_after_validator(state):
    recommendation = state["validator_output"]["recommendation"]
    if recommendation == "FLAG_FOR_HUMAN":
        return "chief"  # Escalate to human
    else:
        return "finalize"
```

## Setup & Maintenance

1. **Install RadGraph**: `pip install radgraph`
2. **Model Version**: The system uses `modern-radgraph-xl` (ModernBERT-based).
3. **Manual Installation**: Run `python scripts/install_radgraph_model.py` after placing the tarball in the root directory.
4. **Dependencies**: Requires `transformers>=4.48.0,<5.0.0` and `torchvision>=0.18.0`.


## Troubleshooting

### FAISS index not found
```
FileNotFoundError: data/mimic_corpus.faiss
```
**Solution:** Run `scripts/build_retrieval_index.py` first

### RadGraph model not loaded
```
[RadGraph] Failed to load model: ...
```
**Solution:** 
1. Download model from PhysioNet
2. Place in `models/radgraph/model.tar.gz`
3. Ensure allennlp is installed: `pip install allennlp==2.9.3`

### Grad-CAM fields missing
```
AttributeError: 'RadiologistOutput' object has no attribute 'gradcam_anatomical_region'
```
**Solution:** These fields are optional. Rules checking Grad-CAM will be skipped if not present.

### Memory issues with FAISS
If index is too large for RAM, consider:
1. Use fewer studies when building index
2. Use FAISS IVF index instead of Flat
3. Increase system RAM or use disk-based index

## Performance Notes

- **Retrieval**: ~50-100ms per query (on GPU)
- **RadGraph**: ~200-500ms per report
- **Rules Engine**: <10ms
- **Total validator time**: ~500-1000ms

## Future Enhancements

- [ ] Add Grad-CAM generation in radiologist agent
- [ ] Support multi-study retrieval for longitudinal cases
- [ ] Add more clinical rules for specific conditions
- [ ] Implement confidence calibration based on validator signals
- [ ] Add validator output to database logging
