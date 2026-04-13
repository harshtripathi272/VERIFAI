import sys
sys.path.append(".")
from graph.state import VerifaiState, RadiologistOutput
from agents.chexbert.agent import chexbert_node

state = VerifaiState({
    "image_paths": ["dummy"],
    "views": ["AP"],
    "patient_id": "test",
    "current_uncertainty": 0.5,
    "is_feedback_iteration": False,
    "human_feedback": None,
    "radiologist_output": RadiologistOutput(findings="Cardiomegaly with pulmonary edema", impression="")
})

print(chexbert_node(state))
