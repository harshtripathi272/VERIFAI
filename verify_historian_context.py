import sys
import os

# Add current dir to path
sys.path.append(os.getcwd())

from agents.historian.fhir_client import fhir_client
from agents.historian.reasoner import summarize_fhir_evidence
import json

# Pick a patient ID from the DB
import duckdb
con = duckdb.connect("verifai_fhir.duckdb")
patient_id = con.execute("SELECT patient_id FROM fhir WHERE patient_id IS NOT NULL LIMIT 1").fetchone()[0]

print(f"Testing for Patient ID: {patient_id}")

evidence = fhir_client.fetch_evidence_for_hypothesis(patient_id, "pneumonia")
summary = summarize_fhir_evidence(evidence)

print("\n--- SUMMARIZED EVIDENCE ---")
print(summary)
print("\n--- END SUMMARY ---")
