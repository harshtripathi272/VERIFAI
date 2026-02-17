"""
Quick diagnostic to check report lengths in your dataset.
This will help us understand if 7% valid tokens is correct or too low.
"""

import json
from pathlib import Path

# Adjust this path to your actual location
TRAIN_JSONL = Path("/data3/Pranshu/elephant_detection/med/dataset/med/train_capped_clean.jsonl")

def analyze_report_lengths(jsonl_path, num_samples=10):
    """Analyze report lengths in the dataset."""
    
    print("="*70)
    print("REPORT LENGTH ANALYSIS")
    print("="*70)
    
    findings_lengths = []
    impression_lengths = []
    total_lengths = []
    
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            
            study = json.loads(line)
            findings = study.get("findings", "").strip()
            impression = study.get("impression", "").strip()
            
            findings_words = len(findings.split())
            impression_words = len(impression.split())
            total_words = findings_words + impression_words
            
            findings_lengths.append(findings_words)
            impression_lengths.append(impression_words)
            total_lengths.append(total_words)
            
            if i < 3:  # Show first 3 samples
                print(f"\nSample {i}:")
                print(f"  Findings: {findings_words} words")
                print(f"  Impression: {impression_words} words")
                print(f"  Total: {total_words} words (~{total_words * 1.3:.0f} tokens)")
                print(f"  First 100 chars of findings: {findings[:100]}...")
                print(f"  First 100 chars of impression: {impression[:100]}...")
    
    # Statistics
    avg_findings = sum(findings_lengths) / len(findings_lengths)
    avg_impression = sum(impression_lengths) / len(impression_lengths)
    avg_total = sum(total_lengths) / len(total_lengths)
    
    print("\n" + "="*70)
    print("STATISTICS (from first 10 samples):")
    print("="*70)
    print(f"Average findings length: {avg_findings:.0f} words (~{avg_findings * 1.3:.0f} tokens)")
    print(f"Average impression length: {avg_impression:.0f} words (~{avg_impression * 1.3:.0f} tokens)")
    print(f"Average total report: {avg_total:.0f} words (~{avg_total * 1.3:.0f} tokens)")
    
    # Calculate expected valid ratio
    # User input: ~3 images * 256 image tokens + ~50 tokens for instruction
    # = ~818 tokens for user input
    # Assistant output: ~avg_total * 1.3 tokens
    user_tokens = 818
    assistant_tokens = avg_total * 1.3
    total = user_tokens + assistant_tokens
    expected_ratio = assistant_tokens / total * 100
    
    print("\n" + "="*70)
    print("EXPECTED TOKEN DISTRIBUTION:")
    print("="*70)
    print(f"User input (images + instruction): ~{user_tokens} tokens")
    print(f"Assistant output (report): ~{assistant_tokens:.0f} tokens")
    print(f"Total sequence: ~{total:.0f} tokens")
    print(f"Expected valid token ratio: {expected_ratio:.1f}%")
    
    if expected_ratio < 15:
        print("\n⚠️  YOUR REPORTS ARE VERY SHORT!")
        print("   This is why you're seeing 7% valid tokens.")
        print("   This is NORMAL for your dataset if reports are truly this short.")
    elif expected_ratio > 40:
        print("\n✓ Reports are long enough, should see higher valid ratio")
    
    return avg_total * 1.3

if __name__ == "__main__":
    try:
        expected_tokens = analyze_report_lengths(TRAIN_JSONL)
    except FileNotFoundError:
        print(f"Error: Could not find {TRAIN_JSONL}")
        print("Please update the TRAIN_JSONL path in this script")