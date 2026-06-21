<p align="center">
  <strong>Arabic Cheque OCR &amp; Verification Pipeline</strong>
</p>

<p align="center">
  End-to-end field localization, courtesy &amp; legal amount recognition, and cross-verification<br/>
  for Arabic bank cheques — deployed serverlessly on Modal.
</p>

<p align="center">
  <a href="https://redfries--arabic-cheque-ocr-run.modal.run"><img src="https://img.shields.io/badge/Live_Demo-Modal-6C3FC5?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHJ4PSI0IiBmaWxsPSIjNkMzRkM1Ii8+PHRleHQgeD0iNSIgeT0iMTciIGZpbGw9IiNmZmYiIGZvbnQtc2l6ZT0iMTIiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXdlaWdodD0iYm9sZCI+TTwvdGV4dD48L3N2Zz4=" alt="Live Demo"/></a>
  <a href="https://github.com/redfries/arabic-cheque-ocr"><img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github" alt="GitHub"/></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/>
</p>

---

## ✨ Live Demo

> **Try it now →** [redfries--arabic-cheque-ocr-run.modal.run](https://redfries--arabic-cheque-ocr-run.modal.run)
>
> Upload a cheque image or select a built-in sample. The pipeline runs detection, OCR, and verification end-to-end on a GPU-backed Modal container.

---

## Architecture

The system is a **unified three-stage pipeline** — detection, recognition, and verification — running inside a single GPU container.

```mermaid
graph LR
    A["🖼️ Cheque Image"] --> B["🔍 Cascade R-CNN<br/><small>Field Detection</small>"]
    B --> C["✂️ Crop & Preprocess"]
    C --> D["🔢 CRNN + CTC<br/><small>Courtesy OCR</small>"]
    C --> E["📝 Qwen3.5 LoRA<br/><small>Legal OCR</small>"]
    D --> F["✅ Verify"]
    E --> F
    F --> G["📊 Results"]

    style A fill:#1a1a2e,stroke:#e8b760,color:#ece8dc
    style B fill:#16213e,stroke:#e8b760,color:#ece8dc
    style C fill:#0f3460,stroke:#e8b760,color:#ece8dc
    style D fill:#533483,stroke:#e8b760,color:#ece8dc
    style E fill:#533483,stroke:#e8b760,color:#ece8dc
    style F fill:#2b6777,stroke:#e8b760,color:#ece8dc
    style G fill:#1a1a2e,stroke:#e8b760,color:#ece8dc
```

| Stage | Component | Description |
|:------|:----------|:------------|
| **Part A** | Cascade R-CNN (ResNet-50 + FPN) | Localizes exactly one courtesy box and one legal box per cheque |
| **Part B — Courtesy** | CRNN + CTC | Transcribes cropped courtesy digits (Arabic-Indic → standard) |
| **Part B — Legal** | Qwen3.5-0.8B + LoRA | Reads handwritten Arabic legal amount text via vision-language model |
| **Verification** | Rule-based parser + fallback | Converts Arabic text → number, cross-checks with courtesy digits |

---

## Metrics

### Part A — Field Detection

Evaluated on our validation split (177 images):

| Metric | Overall | Courtesy | Legal |
|:-------|--------:|---------:|------:|
| **Mean IoU** | 80.21% | 80.18% | 80.25% |
| **Acc @ IoU ≥ 0.50** | 97.46% | 96.61% | 98.31% |
| **Acc @ IoU ≥ 0.75** | 73.73% | 75.71% | 71.75% |

### Part B — Courtesy Amount OCR

Evaluated on professor test set (598 samples), selected checkpoint `v1_last`:

| Digit Accuracy | Exact Match | Insertions | Deletions | Substitutions |
|---------------:|------------:|-----------:|----------:|--------------:|
| **96.65%** | **87.79%** | 26 | 42 | 19 |

### Part B — Legal Amount OCR

Fine-tuned **Qwen3.5-0.8B** with LoRA adapter. Features:
- Arabic-to-number parser with Levenshtein edit-distance recovery
- Courtesy-guided image enhancement fallback on mismatch
- Border cleanup, auto-contrast, and white padding variants

---

## Screenshots

<p align="center">
  <img src="docs/screenshots/verified_match.png" alt="Verified Match Result" width="800"/>
  <br/><em>Verified Case — Bounding boxes, cropped steps, and matching courtesy/legal values</em>
</p>

<p align="center">
  <img src="docs/screenshots/not_verified_mismatch.png" alt="Not Verified Mismatch Result" width="800"/>
  <br/><em>Not Verified Case — Flagging a discrepancy between OCR digits and parsed legal text</em>
</p>

---

## Directory Structure

```text
arabic-cheque-ocr/
├── README.md
├── requirements.txt
├── pipeline_core.py          # Core pipeline (detection, OCR, verification)
├── app_streamlit.py          # Streamlit web application
├── run_cheque_pipeline.py    # CLI for batch processing
├── modal_app.py              # Modal.com serverless deployment
├── eval_iou_metrics.py       # IoU evaluation metrics (Part A)
├── setup_models.py           # Model symlink helper
├── main.py                   # Label Studio data downloader
├── docs/
│   ├── OCR report.md         # Detailed project report
│   └── screenshots/          # App screenshots
├── notebooks/
│   ├── part_A.ipynb          # Part A: Detector training notebook
│   └── part_B.ipynb          # Part B: OCR training notebook
├── sample_images/            # 25 sample cheque TIFFs for demo
└── models/                   # [GIT-IGNORED] Model weights
    ├── detector/
    │   └── model_final.pth
    └── ocr/
        ├── crnn_ctc_v1/
        │   └── checkpoints/last.pt
        └── legal/            # Qwen3.5 LoRA adapter weights
```

---

## Local Setup

### Prerequisites

- Python 3.11 (recommended)
- CUDA-capable GPU (for Qwen3.5 inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/redfries/arabic-cheque-ocr.git
cd arabic-cheque-ocr

# Install dependencies
pip install -r requirements.txt

# Install Detectron2 (build from source)
pip install 'setuptools<70'
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Set up model symlinks (Lightning.ai studio)
python setup_models.py
```

---

## Usage

### CLI — Batch Processing

```bash
python run_cheque_pipeline.py \
  --input /path/to/images \
  --out /path/to/output \
  --det-thresh 0.30 \
  --pad-frac 0.04
```

**Outputs:** `predictions.csv`, `predictions.txt`, `run_summary.json`, `overlays/`, `crops/`, `stages/`

### Streamlit — Interactive Web UI

```bash
streamlit run app_streamlit.py --server.address 0.0.0.0 --server.port 8501
```

### Modal — Cloud Deployment

```bash
# Install and authenticate
pip install modal
modal setup

# Create volume and upload model weights
modal volume create cheque-ocr-models
modal volume put cheque-ocr-models models/detector /detector
modal volume put cheque-ocr-models models/Qwen3.5_model /Qwen3.5_model

# Deploy
modal deploy modal_app.py
```

The app will be available at the URL Modal provides (currently live at [redfries--arabic-cheque-ocr-run.modal.run](https://redfries--arabic-cheque-ocr-run.modal.run)).

---

## Tech Stack

| Layer | Technology |
|:------|:-----------|
| **Detection** | Detectron2 · Cascade R-CNN · ResNet-50 + FPN |
| **Courtesy OCR** | Custom CRNN + BiLSTM + CTC (PyTorch) |
| **Legal OCR** | Qwen3.5-0.8B Vision-Language + LoRA (PEFT) |
| **Preprocessing** | OpenCV · Pillow · CLAHE · Morphological ops |
| **Web UI** | Streamlit |
| **Deployment** | Modal.com (A10G GPU, serverless) |
| **Training** | PyTorch · Transformers · Hugging Face |

---

## Acknowledgements

This project was developed as a **Master's term project** at **King Fahd University of Petroleum and Minerals (KFUPM)**.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://infinitys.me">Shabaaz Hussain</a></sub>
</p>
