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

TARGET_QWEN_BASE_DIR = REPO_ROOT / "models" / "Qwen3.5_model"
TARGET_QWEN_ADAPTER_DIR = REPO_ROOT / "models" / "ocr" / "legal"

# Source paths in the teamspace workspace
SRC_DET_WEIGHTS = Path("/teamspace/studios/this_studio/output_cascade_r50_prep/model_final.pth")
SRC_OCR_CKPT = Path("/teamspace/studios/this_studio/data/PartB_fromLS/runs/crnn_ctc_v1/checkpoints/last.pt")
SRC_OCR_META = Path("/teamspace/studios/this_studio/data/PartB_fromLS/runs/crnn_ctc_v1/metadata.json")

SRC_QWEN_BASE_DIR = Path("/teamspace/studios/this_studio/models/Qwen3.5_model")
SRC_QWEN_ADAPTER_DIR = Path("/teamspace/studios/this_studio/runs/qwen35_legal_ocr_lora_v1/full_train_v3_augmented/checkpoints/best_by_val_cer_clean")


def make_symlink(src_path: Path, dest_path: Path):
    if not src_path.exists():
        print(f"Error: Source path not found: {src_path}", file=sys.stderr)
        return False

    if dest_path.exists() or dest_path.is_symlink():
        print(f"Removing existing path/symlink: {dest_path}")
        if dest_path.is_symlink() or dest_path.is_file():
            dest_path.unlink()
        elif dest_path.is_dir():
            import shutil
            shutil.rmtree(dest_path)

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src_path, dest_path)
        print(f"Created symlink: {dest_path} -> {src_path}")
        return True
    except Exception as e:
        print(f"Failed to create symlink {dest_path}: {e}", file=sys.stderr)
        print("Attempting to copy instead (this might take a while)...")
        try:
            import shutil
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
            print(f"Successfully copied: {src_path} to {dest_path}")
            return True
        except Exception as copy_err:
            print(f"Failed to copy path: {copy_err}", file=sys.stderr)
            return False


def main():
    print("Setting up model directory structure...")
    TARGET_DET_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_OCR_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    success = True
    
    # Detector weights are already tracked by Git LFS in the repo.
    # We only try to symlink if they are missing.
    target_det_file = TARGET_DET_DIR / "model_final.pth"
    if not target_det_file.exists():
        success &= make_symlink(SRC_DET_WEIGHTS, target_det_file)
    else:
        print(f"Detector weights already exist at {target_det_file}, skipping symlink.")

    success &= make_symlink(SRC_OCR_CKPT, TARGET_OCR_CKPT_DIR / "last.pt")
    success &= make_symlink(SRC_OCR_META, TARGET_OCR_DIR / "metadata.json")

    
    success &= make_symlink(SRC_QWEN_BASE_DIR, TARGET_QWEN_BASE_DIR)
    
    target_adapter_file = TARGET_QWEN_ADAPTER_DIR / "adapter_model.safetensors"
    if not target_adapter_file.exists():
        success &= make_symlink(SRC_QWEN_ADAPTER_DIR, TARGET_QWEN_ADAPTER_DIR)
    else:
        print(f"Legal adapter weights already exist at {target_adapter_file}, skipping symlink.")

    if success:
        print("\nAll models set up successfully!")
        print("Structure:")
        print(f"  models/detector/model_final.pth")
        print(f"  models/ocr/crnn_ctc_v1/metadata.json")
        print(f"  models/ocr/crnn_ctc_v1/checkpoints/last.pt")
        print(f"  models/Qwen3.5_model/")
        print(f"  models/ocr/legal/")
    else:
        print("\nWarning: Some files/folders could not be linked or copied.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

