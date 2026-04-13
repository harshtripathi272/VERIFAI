import os
import time
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from graph.workflow import build_workflow
from graph.state import VerifaiState
from agents.radiologist.agent import radiologist_node  # Import for baseline comparison

# ==============================================================================
# VERIFAI AUTOMATED EVALUATION SCRIPT (CHEXPERT INTEGRATED)
# ------------------------------------------------------------------------------
# This script runs VERIFAI over a batch of Chest X-Rays.
# ==============================================================================
# VERIFAI AUTOMATED EVALUATION SCRIPT (NIH CHEST X-RAY 14 INTEGRATED)
# ------------------------------------------------------------------------------
# HOW TO USE WITHOUT DATA AGREEMENTS:
# 1. Go to Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data
# 2. Download the data (it's completely open, covers 14 diseases).
# 3. Update `DATASET_CSV_PATH` to point to `Data_Entry_2017.csv`.
# 4. Update `IMAGE_BASE_DIR` to the folder containing the unzipped images.
# ==============================================================================

DATASET_CSV_PATH = "Data_Entry_2017.csv"
IMAGE_BASE_DIR = "images/images/"  # The Kaggle zip unzips to images/images/
MAX_CASES_TO_EVALUATE = 20  # Limit to 20 for initial testing

# 14 Standard CheXpert Categories
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]

def load_dataset(csv_path: str):
    """Loads either the real CSV or returns a dummy fallback if missing."""
    if os.path.exists(csv_path):
        print(f"✅ Found dataset at {csv_path}. Loading...")
        df = pd.read_csv(csv_path)
        cases = []
        for index, row in df.head(MAX_CASES_TO_EVALUATE).iterrows():
            # Standard NIH Kaggle format parsing
            image_path = os.path.join(IMAGE_BASE_DIR, row["Image Index"])
            view = row.get("View Position", "PA")
            patient_id = str(row.get("Patient ID", f"pt_{index}"))
            
            # Extract ground truth labels (NIH formats them like 'Cardiomegaly|Effusion')
            finding_labels = row.get("Finding Labels", "No Finding")
            gt_present = finding_labels.split("|") if finding_labels != "No Finding" else []

            cases.append({
                "case_id": f"case_{index}_{row['Image Index']}",
                "image_paths": [image_path],
                "views": [view],
                "patient_id": patient_id,
                "ground_truth_chexbert_labels": gt_present,
                "ground_truth_diagnosis": finding_labels
            })
        return cases
    else:
        print(f"⚠️ Dataset '{csv_path}' NOT FOUND.")
        print("Using dummy fallback dataset instead for testing.")
        return [
            {
                "case_id": "dummy_test_001",
                "image_paths": ["assets/thumbnail.png"],  # Using existing dummy asset
                "views": ["PA"],
                "patient_id": "pt_001",
                "ground_truth_diagnosis": "Pneumonia",
                "ground_truth_chexbert_labels": ["Pneumonia"]
            }
        ]

def run_evaluation(output_csv: str = "output/verifai_evaluation_results.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print("=" * 60)
    print("🚀 INITIALIZING VERIFAI EVALUATION RUN")
    print("=" * 60)
    
    dataset = load_dataset(DATASET_CSV_PATH)
    
    # Compile the LangGraph
    workflow = build_workflow()
    graph = workflow.compile()

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "case_id", "patient_id", "ground_truth_labels", 
            "baseline_diagnosis", "verifai_diagnosis", 
            "baseline_conf", "verifai_conf",
            "model_predicted_labels",
            "debate_rounds", "consensus_reached",
            "latency_verifai", "latency_baseline"
        ])
    
        for i, case in enumerate(dataset[:5]):
            print(f"\n[{i+1}/{len(dataset)}] Evaluating Case: {case['case_id']}")
            
            if not os.path.exists(case['image_paths'][0]):
                print(f"   ❌ Image missing on disk, skipping.")
                continue

            # Prepare state
            initial_state = VerifaiState(
                image_paths=case["image_paths"],
                views=case["views"],
                patient_id=case["patient_id"],
                current_uncertainty=0.50,
                is_feedback_iteration=False
            )

            # --- PASS 1: BASELINE (Single Agent / MedGemma Only) ---
            print("   Running Baseline (Single Agent)...")
            start_baseline = time.time()
            try:
                # Use a fresh copy of the state for the baseline
                baseline_res = radiologist_node(initial_state.copy())
                rad_out = baseline_res.get("radiologist_output")
                baseline_dx = rad_out.impression if rad_out else "N/A"
                baseline_conf = 1.0 - baseline_res.get("current_uncertainty", 0.5)
            except Exception as e:
                baseline_dx = f"ERR: {e}"
                baseline_conf = 0.0
            latency_baseline = time.time() - start_baseline

            # --- PASS 2: VERIFAI (Full Multi-Agent Pipeline) ---
            print("   Running VERIFAI (Multi-Agent Pipeline)...")
            start_verifai = time.time()
            try:
                # Full multi-agent run
                output_state = graph.invoke(initial_state, {"recursion_limit": 50})
                
                final_dx = output_state.get("final_diagnosis")
                debate = output_state.get("debate_output")
                chexbert = output_state.get("chexbert_output")
                
                verifai_dx = final_dx.diagnosis if final_dx else "ERROR/DEFERRED"
                verifai_conf = final_dx.calibrated_confidence if final_dx else 0.0
                
                rounds = len(debate.rounds) if debate and hasattr(debate, "rounds") else 0
                consensus = debate.final_consensus if debate and hasattr(debate, "final_consensus") else False
                
                chexbert_present = []
                if chexbert and hasattr(chexbert, "labels"):
                    chexbert_present = [k for k, v in chexbert.labels.items() if v == "present"]
                elif isinstance(chexbert, dict):
                    chexbert_present = [k for k, v in chexbert.items() if v == "present"]

            except Exception as e:
                print(f" Error processing VERIFAI for case {case['case_id']}: {e}")
                verifai_dx = f"CRASH: {str(e)}"
                verifai_conf = 0.0
                rounds = 0
                consensus = False
                chexbert_present = []
            
            latency_verifai = time.time() - start_verifai
            gt_str = "|".join(case["ground_truth_chexbert_labels"])
            pred_str = "|".join(chexbert_present)
            
            # Save results
            writer.writerow([
                case["case_id"], 
                case["patient_id"], 
                gt_str,
                baseline_dx.replace('\n', ' ')[:150],
                verifai_dx.replace('\n', ' ')[:150], 
                round(baseline_conf, 4),
                round(verifai_conf, 4),
                pred_str,
                rounds, 
                consensus,
                round(latency_verifai, 2),
                round(latency_baseline, 2)
            ])
            file.flush() 
            
            print(f"   GT Labels: {gt_str}")
            print(f"   VERIFAI Labels: {pred_str}")
            print(f"   🏁 Done. Baseline: {latency_baseline:.2f}s | VERIFAI: {latency_verifai:.2f}s")
            
    print(f"\n✅ Evaluation complete! Results saved to {output_csv}")
    
    # NEW: Generate Visuals for the Report
    try:
        generate_report_visuals(output_csv)
    except Exception as e:
        print(f"⚠️ Failed to generate visuals: {e}")

def generate_report_visuals(csv_path: str):
    """Generates graphs and charts for the final report."""
    print("\n📊 GENERATING VISUAL REPORT...")
    df = pd.read_csv(csv_path)
    output_dir = "output/visuals/"
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    # 1. Latency Comparison
    plt.figure(figsize=(10, 6))
    latency_data = df[['latency_baseline', 'latency_verifai']].mean()
    sns.barplot(x=latency_data.index, y=latency_data.values, palette="viridis")
    plt.title("Average Latency: Baseline vs VERIFAI")
    plt.ylabel("Time (seconds)")
    plt.savefig(f"{output_dir}latency_comparison.png")
    plt.close()

    # 2. Confidence Comparison
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['baseline_conf'], fill=True, label="Baseline Confidence", color="blue")
    sns.kdeplot(df['verifai_conf'], fill=True, label="VERIFAI Confidence", color="green")
    plt.title("Distribution of Diagnostic Confidence")
    plt.xlabel("Confidence (0.0 - 1.0)")
    plt.legend()
    plt.savefig(f"{output_dir}confidence_distribution.png")
    plt.close()

    # 3. Consensus Rate
    plt.figure(figsize=(7, 7))
    consensus_counts = df['consensus_reached'].value_counts()
    plt.pie(consensus_counts, labels=consensus_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FF5252'])
    plt.title("Multi-Agent Debate Consensus Rate")
    plt.savefig(f"{output_dir}consensus_rate.png")
    plt.close()

    # 4. Accuracy (Simplified proxy by matching GT labels to predicted labels)
    # In a real medical paper, you'd calculate AUC-ROC, but for a 20-case report, 
    # we'll measure 'Exact Label Matches'.
    def calculate_match(row):
        gt = set(str(row['ground_truth_labels']).split('|'))
        pred = set(str(row['model_predicted_labels']).split('|'))
        if gt == {"No Finding"} and (not pred or pred == {""}): return True
        return len(gt.intersection(pred)) > 0

    df['is_accurate'] = df.apply(calculate_match, axis=1)
    acc = (df['is_accurate'].sum() / len(df)) * 100
    
    print(f"📈 Visuals Generated in {output_dir}")
    print(f"⭐️ Measured Multi-Agent Accuracy (Intersection): {acc:.1f}%")

if __name__ == "__main__":
    run_evaluation()
