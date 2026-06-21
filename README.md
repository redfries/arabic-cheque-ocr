# Arabic Cheque Localization & Courtesy Amount OCR Pipeline

This repository contains a production-ready, end-to-end computer vision pipeline to localize critical fields on Arabic bank cheques and transcribe the courtesy amount (numerical digits).

The project is split into two primary components:
- **Part A (Field Localization)**: A Cascade R-CNN detector with a ResNet-50-FPN backbone (implemented in Detectron2) that outputs bounding boxes for the **Courtesy Amount** (numerical digits) and the **Legal Amount** (words in Arabic).
- **Part B (Courtesy OCR)**: A CRNN (Convolutional Recurrent Neural Network) + CTC (Connectionist Temporal Classification) model that transcribes the cropped courtesy amount region into standard digits (normalizing Arabic-Indic numerals like `١`, `٢`, `٣` to `1`, `2`, `3`).

---

## Key Features

- **Field Detection (Part A)**: High-precision localization that detects exactly one top-scoring Courtesy box and one top-scoring Legal box per cheque.
- **Robust OCR (Part B)**: CRNN architecture with Bidirectional LSTM sequence modeling trained using CTC loss. Achieves **96.65% Digit Accuracy** and **87.79% Exact Match** on the test dataset.
- **Advanced Preprocessing**: Crop enhancement pipeline including border cleanup (frame line suppression), CLAHE contrast adjustment, morphological background subtraction, ink boosting, and aspect-ratio-preserving height normalization.
- **Streamlit Web Application**: An interactive UI for uploading cheques, visualizing detections, examining step-by-step OCR preprocessing stages, viewing transcription overlays, and downloading predictions.
- **Bulk CLI Runner**: High-performance CLI script to batch process local directories of images and output formatted predictions in CSV and TXT formats.
- **Modal Deployment Ready**: Pre-configured script (`modal_app.py`) to easily deploy the web app to [Modal](https://modal.com) as a serverless web endpoint.

---

## Directory Structure

```text
arabic-cheque-ocr/
├── .gitignore               # Excludes models, datasets, and local outputs from Git
├── README.md                # Project documentation
├── requirements.txt         # Package dependencies
├── pipeline_core.py         # Core pipeline logic (Model classes, Preprocessing, Inference)
├── app_streamlit.py         # Streamlit web application
├── run_cheque_pipeline.py   # Command-line interface for bulk processing
├── eval_iou_metrics.py      # IoU evaluation metrics for Part A
├── main.py                  # Label Studio data downloader
├── setup_models.py          # Helper script to link models locally in the studio
├── modal_app.py             # Modal.com cloud deployment configuration
├── notebooks/               # Training notebooks
│   ├── mainv2.ipynb         # Full detector and OCR training pipeline
│   └── partb.ipynb          # Additional TrOCR experiments
└── models/                  # [GIT-IGNORED] Model weights directory
    ├── detector/
    │   └── model_final.pth
    └── ocr/
        └── crnn_ctc_v1/
            ├── metadata.json
            └── checkpoints/
                └── last.pt
```

---

## Local Setup

### 1. Requirements and Dependencies

Make sure you are using Python 3.8+ (3.10 recommended). Install core dependencies:
```bash
pip install -r requirements.txt
```

#### Installing Detectron2
Detectron2 requires pre-built wheels matching your CUDA and PyTorch versions, or a compilation from source.
- **For CUDA 11.7 + PyTorch 2.0**:
  ```bash
  pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu117/torch2.0/index.html
  ```
- **To build from source (requires gcc/g++ compilers)**:
  ```bash
  pip install 'git+https://github.com/facebookresearch/detectron2.git'
  ```

### 2. Model Setup (Inside the Studio Workspace)

Run the included model setup helper to automatically create directory structures and symlink the models from the studio workspace folders:
```bash
python setup_models.py
```

*Note: For deployment in a clean environment outside this studio, manually place `model_final.pth` under `models/detector/` and the OCR checkpoint directory structure under `models/ocr/` as shown in the directory tree above.*

---

## Usage

### 1. Running the Command Line Interface (CLI)

Use `run_cheque_pipeline.py` to batch-process a single image or a directory of cheque images:

```bash
python run_cheque_pipeline.py \
  --input /path/to/images \
  --out /path/to/output_folder
```

By default, the script uses the weights under `models/`. You can specify custom weights using flags:
```bash
python run_cheque_pipeline.py \
  --input /path/to/images \
  --det-weights /path/to/custom_detector.pth \
  --ocr-ckpt /path/to/custom_ocr.pt \
  --out /path/to/output_folder \
  --det-thresh 0.30 \
  --pad-frac 0.04
```

This runs detection and OCR, draws visual overlays, and creates three main output files inside your output directory:
- `predictions.csv`: Detailed CSV showing bounding boxes, confidence scores, OCR outputs, and status.
- `predictions.txt`: TSV file containing `<image_stem>\t<predicted_digits>`.
- `run_summary.json`: JSON summary of the run statistics.
- `overlays/`, `crops/`, `stages/`: Visual debugging folders.

### 2. Running the Streamlit UI

Start the Streamlit web application:
```bash
streamlit run app_streamlit.py
```

If running on a cloud environment like Lightning.ai, host it externally:
```bash
streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501
```

Open the provided URL, upload your cheque images, and interact with the pipeline. You will be able to download `outputs.zip` containing all overlays and prediction spreadsheets directly from the interface.

---

## Deploying to Modal.com (Cloud Hosting)

The project includes `modal_app.py` for cloud hosting on [Modal](https://modal.com). 

To deploy:
1. **Install Modal**:
   ```bash
   pip install modal
   ```
2. **Authenticate with Modal**:
   ```bash
   modal setup
   ```
3. **Create the Model Volume and Upload Weights**:
   Create a Modal Volume to store the large weights files and upload your local `models/` directory:
   ```bash
   modal volume create cheque-ocr-models
   modal volume put cheque-ocr-models models/ /
   ```
4. **Deploy the App**:
   ```bash
   modal deploy modal_app.py
   ```
   Modal will automatically build the container image, install PyTorch, CUDA, and Detectron2, mount your model weights volume, and provide a public URL hosting your Streamlit UI.
