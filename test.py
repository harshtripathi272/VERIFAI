import requests

FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

r = requests.get(
    f"{FHIR_BASE_URL}/Patient",
    headers={"Accept": "application/fhir+json"},
    params={
        "_summary": "true",
        "_count": 1
    },
    timeout=10
)

print("Status:", r.status_code)
print(r.json())
