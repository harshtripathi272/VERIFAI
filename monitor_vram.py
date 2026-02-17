#!/usr/bin/env python3
"""
VRAM Monitoring Script

Monitors GPU memory usage during workflow test to verify optimization.
Run this in a separate terminal while running tests/test_workflow.py
"""

import subprocess
import time
import sys

def get_gpu_memory():
    """Get current GPU memory usage in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # Returns memory in MB
        return int(result.stdout.strip().split('\n')[0])
    except Exception as e:
        print(f"Error reading GPU memory: {e}")
        return None

def monitor_vram(interval=2, duration=120):
    """
    Monitor VRAM usage over time.
    
    Args:
        interval: Seconds between measurements
        duration: Total monitoring duration in seconds
    """
    print("=" * 60)
    print("VRAM Monitoring Started")
    print("=" * 60)
    print(f"Target: < 40 GB (40960 MB)")
    print(f"Monitoring every {interval}s for {duration}s...")
    print("")
    
    max_memory = 0
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            memory_mb = get_gpu_memory()
            
            if memory_mb is not None:
                memory_gb = memory_mb / 1024
                max_memory = max(max_memory, memory_mb)
                
                # Status indicator
                status = "✓ OK" if memory_gb < 40  else "✗ EXCEEDED"
                
                print(f"[{time.strftime('%H:%M:%S')}] VRAM: {memory_gb:.2f} GB ({memory_mb} MB) | Max: {max_memory/1024:.2f} GB | {status}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print("")
    print("=" * 60)
    print("Monitoring Complete")
    print("=" * 60)
    print(f"Peak VRAM Usage: {max_memory/1024:.2f} GB ({max_memory} MB)")
    
    if max_memory / 1024 < 40:
        print("✓ SUCCESS: Peak VRAM under 40 GB target")
    else:
        print(f"✗ FAILED: Peak VRAM exceeded target by {max_memory/1024 - 40:.2f} GB")
    
    return max_memory / 1024

if __name__ == "__main__":
    # Monitor for 2 minutes (adjust as needed)
    duration = 600000
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except ValueError:
            print(f"Invalid duration: {sys.argv[1]}, using default 120s")
    
    peak_gb = monitor_vram(interval=2, duration=duration)
    sys.exit(0 if peak_gb < 43 else 1)
