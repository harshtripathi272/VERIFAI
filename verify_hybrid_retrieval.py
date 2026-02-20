
import sys
import os
import json

# Adjust path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.historian.fhir_client import fhir_client

def verify_hybrid():
    print("Verifying Hybrid FHIR Retrieval...")
    
    # 1. Identify a patient
    # We'll just grab one from the DB to be sure
    con = fhir_client.con
    try:
        pat_row = con.execute("SELECT patient_ref FROM fhir_resources LIMIT 1").fetchone()
        if not pat_row:
            print("No patients found in DB! ETL failed?")
            return
        patient_id = pat_row[0]
        print(f"Testing with Patient ID: {patient_id}")
    except Exception as e:
        print(f"DB Error: {e}")
        return

    # 2. Test Hypothesis: "Pneumonia"
    # We expect respiratory conditions or finding matching "pneumonia" to be ranked high
    print("\n--- Query: 'pneumonia' ---")
    evidence = fhir_client.fetch_evidence_hybrid(patient_id, "pneumonia")
    
    # Check source

    print(f"Source: {evidence.get('source')}")
    
    # Check some results
    for key in ["conditions", "observations", "documents"]:
        items = evidence.get(key, [])
        print(f"\n{key.upper()} ({len(items)}):")
        for i, item in enumerate(items[:3]): # Show top 3
            score = item.get("_relevance_score", "N/A")
            summary = item.get("_summary", "N/A")
            print(f"  {i+1}. [{score}] {summary}")
            
    # 3. Test Hypothesis: "Fracture" (Should rank different things higher)
    print("\n--- Query: 'Fracture' ---")
    evidence_frac = fhir_client.fetch_evidence_hybrid(patient_id, "fracture")
    
    # Simple check: Top document/condition should differ if data is diverse enough
    # If not, scores should at least differ.
    
    # 4. output check
    if evidence["source"] == "hybrid_faiss":
        print("\n✅ Verification Passed: Hybrid source confirmed.")
    else:
        print("\n❌ Verification Failed: Source mismatch.")

if __name__ == "__main__":
    verify_hybrid()
