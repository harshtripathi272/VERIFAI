"""
RadGraph Model Installer

Extracts the manually downloaded modern-radgraph-xl.tar.gz into the
correct cache directory that radgraph expects.

Usage:
    python scripts/install_radgraph_model.py
    python scripts/install_radgraph_model.py --tar C:\path\to\modern-radgraph-xl.tar.gz
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path
TARGET_DIR = Path("~/elephant_detection/med/dataset/med/modern-radgraph-xl").expanduser()

# Look for the file in the project root first, then Downloads
ROOT_PATH = Path(__file__).parent.parent / "modern-radgraph-xl.tar.gz"
DOWNLOADS_PATH = Path(os.path.expanduser("~")) / "Downloads" / "modern-radgraph-xl.tar.gz"

DEFAULT_TAR = ROOT_PATH if ROOT_PATH.exists() else DOWNLOADS_PATH


def install(tar_path: Path):
    print(f"Source : {tar_path}")
    print(f"Target : {TARGET_DIR}")

    if not tar_path.exists():
        print(f"\n[Error] File not found: {tar_path}")
        print("Download modern-radgraph-xl.tar.gz from:")
        print("https://huggingface.co/StanfordAIMI/RRG_scorers/blob/main/modern-radgraph-xl.tar.gz")
        sys.exit(1)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print("\nExtracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=TARGET_DIR)

    # Verify expected files
    expected = ["config.json", "weights.th", "vocabulary"]
    missing = [f for f in expected if not (TARGET_DIR / f).exists()]
    if missing:
        print(f"[Warning] These expected files are missing: {missing}")
    else:
        print("[OK] All expected model files are present.")

    print(f"\nDone! Model installed to:\n  {TARGET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", type=Path, default=DEFAULT_TAR,
                        help="Path to modern-radgraph-xl.tar.gz")
    args = parser.parse_args()
    install(args.tar)
