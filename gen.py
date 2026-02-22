import json
import os
import random
import uuid
from datetime import datetime, timedelta

# ---------------------------------------
# CONFIG
# ---------------------------------------

OUTPUT_DIR = "output/fhir"
N_PER_DISEASE = 5

DISEASE_CONFIG = {
    "pneumonia": {
        "display": "Pneumonia",
        "code": "233604007",
        "templates": [
            "Consolidation in the {location} consistent with pneumonia.",
            "Airspace opacity in the {location}, suspicious for pneumonia.",
            "Patchy infiltrates involving the {location}, likely infectious."
        ],
        "locations": ["right lower lobe", "left upper lobe", "bilateral lungs"]
    },
    "tuberculosis": {
        "display": "Pulmonary tuberculosis",
        "code": "154283005",
        "templates": [
            "Cavitary lesion in the {location} concerning for tuberculosis.",
            "Upper lobe consolidation compatible with tuberculosis.",
            "Findings suggest chronic tuberculosis involvement."
        ],
        "locations": ["right upper lobe", "left upper lobe"]
    },
    "covid-19": {
        "display": "COVID-19",
        "code": "840539006",
        "templates": [
            "Bilateral ground-glass opacities consistent with COVID-19 pneumonia.",
            "Peripheral opacities suggest viral pneumonia.",
            "Diffuse infiltrates suspicious for COVID-19 infection."
        ],
        "locations": ["bilateral lungs"]
    },
    "copd": {
        "display": "Chronic obstructive pulmonary disease",
        "code": "13645005",
        "templates": [
            "Hyperinflated lungs consistent with COPD.",
            "Flattened diaphragms and increased AP diameter.",
            "Emphysematous changes noted bilaterally."
        ],
        "locations": [""]
    },
    "pulmonary_edema": {
        "display": "Pulmonary edema",
        "code": "40541001",
        "templates": [
            "Bilateral perihilar opacities consistent with pulmonary edema.",
            "Diffuse interstitial markings suggest fluid overload.",
            "Cardiogenic pulmonary edema suspected."
        ],
        "locations": [""]
    },
    "cardiomegaly": {
        "display": "Cardiomegaly",
        "code": "8186001",
        "templates": [
            "Enlarged cardiac silhouette consistent with cardiomegaly.",
            "Cardiothoracic ratio increased.",
            "Marked cardiomegaly observed."
        ],
        "locations": [""]
    },
    "interstitial_lung_disease": {
        "display": "Pulmonary fibrosis",
        "code": "51672001",
        "templates": [
            "Reticular opacities consistent with interstitial lung disease.",
            "Fibrotic changes in lower lobes.",
            "Diffuse interstitial thickening observed."
        ],
        "locations": [""]
    },
    "pleural_effusion": {
        "display": "Pleural effusion",
        "code": "60046008",
        "templates": [
            "Moderate pleural effusion on the {location}.",
            "Blunting of costophrenic angle due to effusion.",
            "Fluid accumulation in pleural space."
        ],
        "locations": ["right side", "left side"]
    },
    "atelectasis": {
        "display": "Atelectasis",
        "code": "46635009",
        "templates": [
            "Subsegmental atelectasis in the {location}.",
            "Volume loss consistent with atelectasis.",
            "Linear opacity suggestive of atelectasis."
        ],
        "locations": ["right lower lobe", "left lower lobe"]
    },
    "pneumothorax": {
        "display": "Pneumothorax",
        "code": "233604008",
        "templates": [
            "Small pneumothorax on the {location}.",
            "Collapsed lung segment consistent with pneumothorax.",
            "Air in pleural space observed."
        ],
        "locations": ["right side", "left side"]
    },
    "lung_nodule": {
        "display": "Lung nodule",
        "code": "255108000",
        "templates": [
            "Solitary pulmonary nodule in the {location}.",
            "Small round opacity suspicious for lung nodule.",
            "Well-circumscribed pulmonary nodule noted."
        ],
        "locations": ["right upper lobe", "left lower lobe"]
    }
}

# ---------------------------------------
# HELPERS
# ---------------------------------------

def random_patient():
    return {
        "age": random.randint(18, 85),
        "gender": random.choice(["male", "female"])
    }

def random_date():
    return (datetime.now() - timedelta(days=random.randint(0, 1000))).isoformat()

# ---------------------------------------
# MAIN
# ---------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

for disease, config in DISEASE_CONFIG.items():
    for i in range(N_PER_DISEASE):

        patient_id = str(uuid.uuid4())
        condition_id = str(uuid.uuid4())
        report_id = str(uuid.uuid4())
        encounter_id = str(uuid.uuid4())

        patient_info = random_patient()
        location = random.choice(config["locations"])
        template = random.choice(config["templates"])
        narrative = template.format(location=location)

        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": patient_id,
                        "gender": patient_info["gender"],
                        "birthDate": str(datetime.now().year - patient_info["age"])
                    }
                },
                {
                    "resource": {
                        "resourceType": "Encounter",
                        "id": encounter_id,
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "period": {"start": random_date()}
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": condition_id,
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "code": {
                            "coding": [{
                                "system": "http://snomed.info/sct",
                                "code": config["code"],
                                "display": config["display"]
                            }]
                        },
                        "clinicalStatus": {
                            "coding": [{"code": "active"}]
                        }
                    }
                },
                {
                    "resource": {
                        "resourceType": "DiagnosticReport",
                        "id": report_id,
                        "subject": {"reference": f"Patient/{patient_id}"},
                        "encounter": {"reference": f"Encounter/{encounter_id}"},
                        "conclusion": narrative,
                        "effectiveDateTime": random_date()
                    }
                }
            ]
        }

        filename = f"{disease}_{i+1}.json"
        file_path = os.path.join(OUTPUT_DIR, filename)

        with open(file_path, "w") as f:
            json.dump(bundle, f, indent=2)

print("Deterministic radiology FHIR generation complete (flat structure).")
