"""
Simple test script for Historian agent only.
Tests the evidence gathering and reasoning with MedGemma using shared model loader.
"""

import sys
import os
# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.historian.reasoner import reason_over_fhir
import json


def test_historian():
    print("=" * 70)
    print("HISTORIAN AGENT TEST - EVIDENCE GATHERING & REASONING")
    print("=" * 70)
    
    # Test inputs
    patient_id = "test-patient-001"
    hypothesis = "pneumonia"
    
    print(f"\nInput:")
    print(f"  Patient ID: {patient_id}")
    print(f"  Hypothesis: {hypothesis}")
    
    # Create sample FHIR evidence data for testing
    # This simulates what would be fetched from the FHIR database
    evidence = {
        "conditions": [
            {
                "id": "cond-001",
                "resourceType": "Condition",
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "233604007",
                        "display": "Pneumonia"
                    }]
                }
            },
            {
                "id": "cond-002",
                "resourceType": "Condition",
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "38341003",
                        "display": "Hypertension"
                    }]
                }
            }
        ],
        "observations": [
            {
                "id": "obs-001",
                "resourceType": "Observation",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "33756-8",
                        "display": "White blood cell count"
                    }]
                },
                "valueQuantity": {
                    "value": 15000,
                    "unit": "/uL"
                }
            },
            {
                "id": "obs-002",
                "resourceType": "Observation",
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": "8310-5",
                        "display": "Body temperature"
                    }]
                },
                "valueQuantity": {
                    "value": 38.5,
                    "unit": "C"
                }
            }
        ],
        "medications": [
            {
                "id": "med-001",
                "resourceType": "MedicationRequest",
                "medicationCodeableConcept": {
                    "coding": [{
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": "1659149",
                        "display": "Azithromycin 500 MG"
                    }]
                }
            }
        ],
        "procedures": [
            {
                "id": "proc-001",
                "resourceType": "Procedure",
                "code": {
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "168731009",
                        "display": "Chest X-ray"
                    }]
                }
            }
        ],
        "allergies": [],
        "encounters": [
            {
                "id": "enc-001",
                "resourceType": "Encounter",
                "reasonCode": [{
                    "coding": [{
                        "system": "http://snomed.info/sct",
                        "code": "49727002",
                        "display": "Cough"
                    }]
                }]
            }
        ],
        "documents": [
            {
                "id": "doc-001",
                "resourceType": "DiagnosticReport",
                "category": "Radiology",
                "text": """Chest X-ray Report:
FINDINGS: Patchy airspace opacities noted in the right lower lobe. 
Heart size is within normal limits. No pleural effusion. 
Bony thorax is intact.
IMPRESSION: Findings consistent with right lower lobe pneumonia."""
            }
        ],
        "source": "test"
    }
    
    print("\n" + "-" * 70)
    print("Step 1: Sample FHIR Evidence Created")
    print("-" * 70 + "\n")
    
    print("\n" + "=" * 70)
    print("FHIR EVIDENCE (SAMPLE DATA)")
    print("=" * 70)
    print(f"\nConditions found: {len(evidence.get('conditions', []))}")
    print(f"Observations found: {len(evidence.get('observations', []))}")
    print(f"Medications found: {len(evidence.get('medications', []))}")
    print(f"Procedures found: {len(evidence.get('procedures', []))}")
    print(f"Allergies found: {len(evidence.get('allergies', []))}")
    print(f"Encounters found: {len(evidence.get('encounters', []))}")
    print(f"Documents found: {len(evidence.get('documents', []))}")
    
    # Pretty print evidence summary
    print("\n" + "=" * 70)
    print("EVIDENCE SUMMARY")
    print("=" * 70)
    
    if evidence.get('conditions'):
        print("\nConditions:")
        for cond in evidence['conditions']:
            coding = cond.get('code', {}).get('coding', [{}])[0]
            print(f"  - {coding.get('display', 'N/A')} (Condition/{cond.get('id')})")
    
    if evidence.get('observations'):
        print("\nObservations (Labs):")
        for obs in evidence['observations']:
            coding = obs.get('code', {}).get('coding', [{}])[0]
            value = obs.get('valueQuantity', {}).get('value', 'N/A')
            unit = obs.get('valueQuantity', {}).get('unit', '')
            print(f"  - {coding.get('display', 'N/A')}: {value} {unit}")
    
    if evidence.get('medications'):
        print("\nMedications:")
        for med in evidence['medications']:
            med_code = med.get('medicationCodeableConcept', {}).get('coding', [{}])[0]
            print(f"  - {med_code.get('display', 'N/A')}")
    
    if evidence.get('documents'):
        print("\nDocuments:")
        for doc in evidence['documents']:
            print(f"  - {doc.get('resourceType')}/{doc.get('id')}")
            print(f"    Category: {doc.get('category', 'N/A')}")
            print(f"    Text preview:\n{doc.get('text', '')[:150]}...")
    
    print("\n" + "-" * 70)
    print("Step 2: Running MedGemma Reasoning (Shared Model)...")
    print("         This will load the shared model if not already loaded")
    print("-" * 70 + "\n")
    
    try:
        # Step 2: Reason over the evidence with MedGemma
        reasoning_result = reason_over_fhir(
            hypothesis=hypothesis,
            evidence=evidence
        )
        
        print("\n" + "=" * 70)
        print("REASONING OUTPUT")
        print("=" * 70)
        
        supporting = reasoning_result.get('supporting_facts', [])
        contradicting = reasoning_result.get('contradicting_facts', [])
        confidence_adj = reasoning_result.get('confidence_adjustment', 0.0)
        
        print(f"\nSupporting Facts: {len(supporting)}")
        for i, fact in enumerate(supporting, 1):
            print(f"  {i}. {fact.get('description', 'N/A')}")
            print(f"     Resource: {fact.get('resource_type', 'N/A')}/{fact.get('resource_id', 'N/A')}")
        
        print(f"\nContradicting Facts: {len(contradicting)}")
        for i, fact in enumerate(contradicting, 1):
            print(f"  {i}. {fact.get('description', 'N/A')}")
            print(f"     Resource: {fact.get('resource_type', 'N/A')}/{fact.get('resource_id', 'N/A')}")
        
        print(f"\nConfidence Adjustment: {confidence_adj:+.3f}")
        
        # Pretty print full reasoning result
        print("\n" + "=" * 70)
        print("FULL REASONING RESULT (JSON)")
        print("=" * 70)
        print(json.dumps(reasoning_result, indent=2))
        
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR OCCURRED")
        print("=" * 70)
        print(f"\nException type: {type(e).__name__}")
        print(f"Exception message: {e}")
        
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        
        print("\n✗ Test failed!")
        return False
    
    return True


if __name__ == "__main__":
    success = test_historian()
    exit(0 if success else 1)
