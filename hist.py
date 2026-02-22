from graph.state import VerifaiState, CheXbertOutput, RadiologistOutput
from agents.historian.agent import historian_node
from agents.historian.fhir_client import fhir_client

def verify_historian_pipeline():

    print("\n==============================")
    print("VERIFYING HISTORIAN PIPELINE")
    print("==============================\n")

    # 1️⃣ Get REAL patient from DB
    con = fhir_client.con
    row = con.execute("SELECT DISTINCT patient_ref FROM fhir_resources LIMIT 1").fetchone()

    if not row:
        print("❌ No patients found in DB.")
        return

    patient_id = row[0]
    print(f"Using Patient ID: {patient_id}\n")

    # 2️⃣ Radiologist output
    rad_output = RadiologistOutput(
        findings="Right lower lobe consolidation.",
        impression="Findings consistent with pneumonia."
    )

    # 3️⃣ CheXbert output
    chex_output = CheXbertOutput(
        labels={
            "Lung Cancer": "present",
            "Atelectasis": "present"
        }
    )

    # 4️⃣ Build state
    state: VerifaiState = {
        "image_path": "",
        "patient_id": patient_id,
        "dicom_metadata": None,
        "view": None,

        "radiologist_output": rad_output,
        "chexbert_output": chex_output,
        "critic_output": None,
        "historian_output": None,
        "literature_output": None,
        "debate_output": None,
        "validator_output": None,

        "current_uncertainty": 0.2,
        "routing_decision": "",
        "steps_taken": 0,

        "radiologist_kle_uncertainty": None,
        "final_diagnosis": None,

        "doctor_feedback": None,
        "is_feedback_iteration": False,
        "trace": []
    }

    # 5️⃣ Run historian
    print("Running Historian Node...\n")
    result = historian_node(state)

    historian_output = result.get("historian_output")

    print("\n========== HISTORIAN OUTPUT ==========\n")

    if historian_output:
        print(historian_output.model_dump_json(indent=2))
    else:
        print("❌ Historian returned None")

    print("\n========== TRACE ==========\n")
    for step in result.get("trace", []):
        print(step)


if __name__ == "__main__":
    verify_historian_pipeline()
