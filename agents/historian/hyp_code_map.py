CHEST_HYPOTHESIS_CODE_MAP = {
    # Infections
    "pneumonia": {
        "conditions": [],  # often not coded yet
        "labs": ["6690-2", "1988-5"],  # WBC, CRP
        "medications": []
    },
    "tuberculosis": {
        "conditions": ["56717001"],  # SNOMED: Tuberculosis
        "labs": [],
        "medications": []
    },
    "covid-19": {
        "conditions": [],
        "labs": ["1988-5"],  # CRP
        "medications": []
    },

    # Opportunistic
    "pcp pneumonia": {
        "conditions": [],
        "labs": ["1988-5"],
        "medications": []
    },

    # Cardiac-related
    "pulmonary edema": {
        "conditions": ["84114007"],  # Heart failure
        "labs": ["3094-0"],          # BNP
        "medications": []
    },
    "cardiomegaly": {
        "conditions": ["84114007"],
        "labs": [],
        "medications": []
    },

    # Pulmonary
    "pleural effusion": {
        "conditions": [],
        "labs": [],
        "medications": []
    },
    "atelectasis": {
        "conditions": [],
        "labs": [],
        "medications": []
    },

    # Chronic disease
    "copd": {
        "conditions": ["13645005"],  # SNOMED: COPD
        "labs": [],
        "medications": []
    },
    "interstitial lung disease": {
        "conditions": ["64667001"],  # SNOMED: ILD
        "labs": [],
        "medications": []
    },

    
    # Acute findings
    
    "pneumothorax": {
        "conditions": [],
        "labs": [],
        "medications": []
    },
    "lung collapse": {
        "conditions": [],
        "labs": [],
        "medications": []
    },

    # Mass-like findings
    "lung nodule": {
        "conditions": [],
        "labs": [],
        "medications": []
    }
}

def normalize_hypothesis(name: str) -> str:
    name = name.lower().strip()
    for key in CHEST_HYPOTHESIS_CODE_MAP:
        if key in name:
            return key
    return name
