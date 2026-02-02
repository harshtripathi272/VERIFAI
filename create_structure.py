import os

dirs = [
    "verifai/app",
    "verifai/graph",
    "verifai/agents/radiologist",
    "verifai/agents/critic",
    "verifai/agents/historian",
    "verifai/agents/literature",
    "verifai/agents/chief",
    "verifai/tools",
    "verifai/proof_layer",
    "verifai/ui",
    "verifai/data/sample_dicom",
    "verifai/data/sample_fhir",
    "verifai/data/embeddings",
    "verifai/cache",
    "verifai/tests",
    "verifai/docker"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)
    # Create __init__.py in python packages
    if "data" not in d and "docker" not in d and "cache" not in d:
        with open(os.path.join(d, "__init__.py"), "w") as f:
            pass
