import duckdb, json

con = duckdb.connect("verifai_fhir.duckdb")

row = con.execute("""
    SELECT resourceType, id, json
    FROM fhir
    WHERE resourceType IN ('DiagnosticReport', 'DocumentReference')
    LIMIT 1
""").fetchone()

print("ResourceType:", row[0])
print("ID:", row[1])
print("\nRAW JSON:\n")
print(json.dumps(json.loads(row[2]), indent=2))
