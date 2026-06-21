# -*- coding: utf-8 -*-
"""
Modal deployment script for Arabic Cheque OCR Streamlit Application.

To deploy this app to Modal:
1. Install modal: pip install modal
2. Authenticate: modal setup
3. Create the model volume and upload model weights:
    modal volume create cheque-ocr-models
    modal volume put cheque-ocr-models models/ /
4. Deploy the app:
    modal deploy modal_app.py
"""
import os
import subprocess
import modal

# Define the Modal App name
app = modal.App("arabic-cheque-ocr")

# Create/retrieve a persistent volume for storing model weights
# This prevents putting large files in Git or rebuilds.
volume = modal.Volume.from_name("cheque-ocr-models", create_if_missing=True)

# Build a container image with all needed libraries
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install OS libraries required by OpenCV and building Detectron2
    .apt_install(
        "git",
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "gcc",
        "g++",
    )
    # Install Python dependencies
    .pip_install(
        "streamlit>=1.20.0",
        "opencv-python-headless>=4.5.0",
        "torch>=1.13.0",
        "torchvision>=0.14.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "pandas>=1.5.0",
        "requests>=2.28.0",
    )
    # Build and install Detectron2 from source
    .run_commands(
        "pip install 'git+https://github.com/facebookresearch/detectron2.git'"
    )
)

# Define a local mount for the application source files
# This mounts the current folder into the container during run
src_mount = modal.Mount.from_local_dir(
    os.path.dirname(__file__),
    remote_path="/root/app",
)


@app.function(
    image=image,
    mounts=[src_mount],
    # Mount the volume containing model weights under /root/models.
    # This aligns with the default paths "models/detector/model_final.pth"
    # and "models/ocr/crnn_ctc_v1/checkpoints/last.pt" when running from /root/app.
    volumes={"/root/models": volume},
    timeout=3600,
)
@modal.web_server(port=8501)
def run():
    # Start the Streamlit application
    # We change directory to /root/app so relative imports and relative paths work as expected
    cmd = [
        "streamlit",
        "run",
        "/root/app/app_streamlit.py",
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
    ]
    # Launch Streamlit as a subprocess
    # modal.web_server expects this function to start a process listening on the specified port
    print("Starting Streamlit web server on port 8501...")
    subprocess.Popen(cmd, cwd="/root/app")
