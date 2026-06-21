#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper script to set up model symlinks for local running.
This links the large model files from the studio workspace into the repository structure.
"""
import os
import sys
from pathlib import Path

# Target repository structure
REPO_ROOT = Path(__file__).resolve().parent
TARGET_DET_DIR = REPO_ROOT / "models" / "detector"
TARGET_OCR_DIR = REPO_ROOT / "models" / "ocr" / "crnn_ctc_v1"
TARGET_OCR_CKPT_DIR = TARGET_OCR_DIR / "checkpoints"

# Source paths in the teamspace workspace
SRC_DET_WEIGHTS = Path("/teamspace/studios/this_studio/output_cascade_r50_prep/model_final.pth")
SRC_OCR_CKPT = Path("/teamspace/studios/this_studio/data/PartB_fromLS/runs/crnn_ctc_v1/checkpoints/last.pt")
SRC_OCR_META = Path("/teamspace/studios/this_studio/data/PartB_fromLS/runs/crnn_ctc_v1/metadata.json")


def make_symlink(src_path: Path, dest_path: Path):
    if not src_path.exists():
        print(f"Error: Source file not found: {src_path}", file=sys.stderr)
        return False

    if dest_path.exists() or dest_path.is_symlink():
        print(f"Removing existing file/symlink: {dest_path}")
        dest_path.unlink()

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src_path, dest_path)
        print(f"Created symlink: {dest_path} -> {src_path}")
        return True
    except Exception as e:
        print(f"Failed to create symlink {dest_path}: {e}", file=sys.stderr)
        print("Attempting to copy file instead (this might take a while)...")
        try:
            import shutil
            shutil.copy2(src_path, dest_path)
            print(f"Successfully copied: {src_path} to {dest_path}")
            return True
        except Exception as copy_err:
            print(f"Failed to copy file: {copy_err}", file=sys.stderr)
            return False


def main():
    print("Setting up model directory structure...")
    TARGET_DET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_OCR_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    success = True
    success &= make_symlink(SRC_DET_WEIGHTS, TARGET_DET_DIR / "model_final.pth")
    success &= make_symlink(SRC_OCR_CKPT, TARGET_OCR_CKPT_DIR / "last.pt")
    success &= make_symlink(SRC_OCR_META, TARGET_OCR_DIR / "metadata.json")

    if success:
        print("\nAll models set up successfully!")
        print("Structure:")
        print(f"  models/detector/model_final.pth")
        print(f"  models/ocr/crnn_ctc_v1/metadata.json")
        print(f"  models/ocr/crnn_ctc_v1/checkpoints/last.pt")
    else:
        print("\nWarning: Some files could not be linked or copied.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
