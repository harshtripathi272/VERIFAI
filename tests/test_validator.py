import torch
from transformers import AutoImageProcessor
from transformers import SiglipVisionModel

from agents.validator import (
    initialize_validator_tools,
    validator_node
)

from graph.state import VerifaiState

# -------------------------
# 1. Load vision encoder
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

vision_encoder = SiglipVisionModel.from_pretrained(
    "google/medsiglip-448"
).to(device).eval()

image_processor = AutoImageProcessor.from_pretrained(
    "google/medsiglip-448"
)

# -------------------------
# 2. Initialize validator tools
# -------------------------
initialize_validator_tools(
    vision_encoder=vision_encoder,
    image_processor=image_processor
)

# -------------------------
# 3. Build Fake State
# -------------------------
state = VerifaiState()

# Simulate radiologist output
state["radiologist_output"] = type("R", (), {
    "findings": "Left lower lobe opacity suggestive of pneumonia.",
    "impression": "Findings consistent with pneumonia."
})()

state["current_uncertainty"] = 0.12

# Simulate CheXbert
state["chexbert_output"] = type("C", (), {
    "labels": {
        "Pneumonia": "Positive",
        "Cardiomegaly": "Negative"
    }
})()

# Simulate Critic
state["critic_output"] = type("CR", (), {
    "is_overconfident": False,
    "concern_flags": [],
    "safety_score": 0.9
})()

# Simulate Historian
state["historian_output"] = type("H", (), {
    "supporting_facts": ["Similar pneumonia case in training data."],
    "contradicting_facts": []
})()

# Simulate Literature
state["literature_output"] = type("L", (), {
    "overall_evidence_strength": "high",
    "citations": ["PMID:123456"]
})()

# Simulate Debate
state["debate_output"] = type("D", (), {
    "final_consensus": True,
    "rounds": [1,2],
    "escalate_to_chief": False
})()

# Add image path for retrieval tool
state["image_path"] = "img1.jpg"

# -------------------------
# 4. Run Validator
# -------------------------
result = validator_node(state)

print("\n========== VALIDATOR RESULT ==========")
print("Routing Decision:", result["routing_decision"])
print("\nTrace:")
for t in result["trace"]:
    print("-", t)

print("\nValidator Output Keys:")
print(result["validator_output"].keys())
