# -*- coding: utf-8 -*-
"""
Streamlit UI for cheque pipeline.

Per image:
- Full cheque overlay with BOTH boxes: courtesy red, legal green, plus predicted digits.
- Courtesy crop.
- Preprocessing stages: raw -> border -> lines_removed -> enhanced -> enhanced_resized.
- Download outputs.zip with all artifacts.
"""
from __future__ import annotations

from pathlib import Path
import base64
import io
from io import BytesIO
import os
import random
import tempfile
import zipfile

from PIL import Image
import streamlit as st

from pipeline_core import PipelineConfig, ChequeOCRPipeline, write_outputs

def load_pil_image(path) -> Image.Image | None:
    try:
        p = Path(path)
        if p.exists():
            return Image.open(p)
    except Exception:
        pass
    return None

def render_image(pil_img, caption="", use_container_width=True):
    if pil_img is None:
        return
    try:
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        width_style = "width: 100%;" if use_container_width else "max-width: 500px;"
        html = f'<img src="data:image/png;base64,{img_str}" style="{width_style} height: auto; border-radius: 4px; margin-bottom: 5px;" alt="{caption}">'
        st.markdown(html, unsafe_allow_html=True)
        if caption:
            st.caption(caption)
    except Exception as e:
        st.error(f"Error rendering image: {e}")

st.set_page_config(page_title="Cheque OCR Pipeline", layout="wide")
st.title("Cheque OCR Pipeline (Detection + OCR)")

# Detect if running inside a Modal container
is_modal = os.path.exists("/root/models")

if is_modal:
    default_det_weights = "/root/models/detector/model_final.pth"
    default_ocr_ckpt = "/root/app/models/ocr/crnn_ctc_v1/checkpoints/last.pt"
    default_legal_ocr_base = "/root/models/Qwen3.5_model"
    default_legal_ocr_ckpt = "/root/app/models/ocr/legal"
else:
    default_det_weights = "models/detector/model_final.pth"
    default_ocr_ckpt = "models/ocr/crnn_ctc_v1/checkpoints/last.pt"
    default_legal_ocr_base = "models/Qwen3.5_model"
    default_legal_ocr_ckpt = "models/ocr/legal"

with st.sidebar:
    st.header("Model paths")
    det_weights = st.text_input("Part A Detectron2 weights (.pth)", value=default_det_weights)
    ocr_ckpt = st.text_input("Part B OCR checkpoint (.pt)", value=default_ocr_ckpt)

    st.header("Detection")
    det_thresh = st.slider("Detector score threshold", 0.01, 0.90, 0.30, 0.01)
    courtesy_class = st.number_input("Courtesy class id", min_value=0, max_value=10, value=0, step=1)
    legal_class = st.number_input("Legal class id (-1 = auto)", min_value=-1, max_value=10, value=-1, step=1)

    st.header("Courtesy Crop + OCR")
    pad_frac = st.slider("Courtesy crop padding", 0.00, 0.20, 0.04, 0.01)
    do_line_cleanup = st.checkbox("Remove long lines", value=True)
    
    st.header("Legal Crop + OCR (Qwen3.5)")
    legal_ocr_ckpt = st.text_input("Part B Legal OCR adapter", value=default_legal_ocr_ckpt)
    legal_ocr_base = st.text_input("Part B Legal OCR base model", value=default_legal_ocr_base)
    legal_pad_frac = st.slider("Legal crop padding", 0.00, 0.20, 0.05, 0.01)
    do_fallback = st.checkbox("Enable fallback enhancements", value=True)
    
    st.header("Output Settings")
    save_debug = st.checkbox("Save debug artifacts", value=True)

st.write("Upload 1 or more cheque images, or select a pre-loaded sample cheque to test the pipeline.")

# Search for sample images
sample_dir = Path(__file__).parent / "sample_images"
sample_files = []
if sample_dir.exists():
    sample_files = sorted(
        list(sample_dir.glob("*.tif")) + 
        list(sample_dir.glob("*.tiff")) + 
        list(sample_dir.glob("*.png")) + 
        list(sample_dir.glob("*.jpg")) + 
        list(sample_dir.glob("*.jpeg"))
    )

if "selected_sample" not in st.session_state:
    if sample_files:
        st.session_state.selected_sample = random.choice(sample_files)
    else:
        st.session_state.selected_sample = None

def select_random_sample():
    if sample_files:
        st.session_state.selected_sample = random.choice(sample_files)

input_mode = st.radio(
    "Select Input Source",
    ["Select a Sample Cheque (Instant Demo)", "Upload Your Own Cheque(s)"]
)

input_files_to_process = []

if input_mode == "Select a Sample Cheque (Instant Demo)":
    if not sample_files:
        st.warning("No sample images found in `sample_images/` directory.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Current Sample: `{st.session_state.selected_sample.name}`")
        with col2:
            st.button("🎲 Pick Another Random Cheque", on_click=select_random_sample)
        
        if st.session_state.selected_sample:
            img = load_pil_image(st.session_state.selected_sample)
            if img:
                render_image(img, caption="Sample Cheque Preview", use_container_width=False)
            input_files_to_process = [st.session_state.selected_sample]
else:
    uploads = st.file_uploader(
        "Upload cheque images",
        type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        input_files_to_process = uploads

run = st.button("Run Pipeline", type="primary", disabled=(not input_files_to_process or not det_weights or not ocr_ckpt or not legal_ocr_ckpt or not legal_ocr_base))

if run:
    legal = None if int(legal_class) < 0 else int(legal_class)

    cfg = PipelineConfig(
        det_weights=det_weights,
        ocr_ckpt=ocr_ckpt,
        det_score_thresh=float(det_thresh),
        courtesy_pred_class=int(courtesy_class),
        legal_pred_class=legal,
        pad_frac=float(pad_frac),
        do_line_cleanup=bool(do_line_cleanup),
        legal_ocr_ckpt=legal_ocr_ckpt,
        legal_ocr_base=legal_ocr_base,
        legal_pad_frac=float(legal_pad_frac),
        do_fallback=bool(do_fallback),
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_dir = td / "inputs"
        out_dir = td / "outputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        input_paths = []
        for f in input_files_to_process:
            if isinstance(f, Path):
                p = in_dir / f.name
                p.write_bytes(f.read_bytes())
            else:
                p = in_dir / f.name
                p.write_bytes(f.getbuffer())
            input_paths.append(p)

        pipe = ChequeOCRPipeline(cfg)

        prog = st.progress(0, text="Running pipeline...")
        results = []
        n = len(input_paths)
        for i, p in enumerate(input_paths, start=1):
            results.append(pipe.process_one(p, out_dir=out_dir, save_debug=save_debug))
            prog.progress(int(i * 100 / n), text=f"Processed {i}/{n}: {p.name}")
        prog.empty()

        csv_path, txt_path = write_outputs(results, out_dir=out_dir)

        st.success(f"Done. Processed {len(results)} images.")
        st.dataframe([{k: r.get(k) for k in ["stem", "ocr_digits", "ocr_legal_text", "verified", "legal_parsed_amount", "courtesy_score", "legal_score", "status"]} for r in results])

        st.divider()
        st.subheader("Per-image visualization")

        for r in results:
            stem = r["stem"]
            with st.expander(stem, expanded=(len(results) <= 2)):
                colA, colB = st.columns([1.2, 1.0])

                with colA:
                    img_orig = load_pil_image(r["image_path"])
                    if img_orig:
                        render_image(img_orig, caption="Original cheque", use_container_width=True)
                    else:
                        st.warning("Original cheque image not found.")

                    overlay = out_dir / "overlays" / f"{stem}.png"
                    img_overlay = load_pil_image(overlay)
                    if img_overlay:
                        render_image(img_overlay, caption="Full cheque with courtesy and legal boxes", use_container_width=True)
                    else:
                        st.warning("Overlay not found.")

                    # Verification Status
                    verified = r.get("verified")
                    if verified:
                        st.success("✅ **Verified (Courtesy matches Legal amount!)**")
                    else:
                        st.error("❌ **Not Verified (Courtesy/Legal mismatch or missing)**")

                    st.markdown(f"**Courtesy Digits:** `{r.get('ocr_digits','')}`")
                    st.markdown(f"**Legal Text (Arabic):** `{r.get('ocr_legal_text','')}`")
                    if r.get("legal_parsed_amount") is not None:
                        st.markdown(f"**Parsed Legal Amount:** `{r.get('legal_parsed_amount')}`")

                    if r.get("ocr_raw"):
                        st.caption(f"Raw courtesy decode: {r.get('ocr_raw')}")
                    dbg = r.get("preprocess_debug") or {}
                    if dbg:
                        st.caption(f"Line cleanup applied: {dbg.get('line_cleanup_applied')} | Ink keep ratio: {dbg.get('ink_keep_ratio')}")

                with colB:
                    st.subheader("Crops & Processing Breakdown")
                    
                    crop = Path(r.get("crop_path") or "")
                    img_crop = load_pil_image(crop)
                    if img_crop:
                        render_image(img_crop, caption="Courtesy crop (raw)", use_container_width=True)
                    else:
                        st.warning("Courtesy crop missing.")

                    # Preprocessing Progression
                    stages_dict = r.get("stage_paths") or {}
                    if stages_dict:
                        st.markdown("**Courtesy Extraction Progression (Processing Stages):**")
                        # The stages are: 'raw', 'border', 'lines_removed', 'enhanced', 'enhanced_resized'
                        stage_labels = {
                            "raw": "1. Raw Crop",
                            "border": "2. Border Removed",
                            "lines_removed": "3. Lines Removed",
                            "enhanced": "4. Enhanced Bin",
                            "enhanced_resized": "5. CRNN Input"
                        }
                        # Display in a sub-grid of columns
                        sub_cols = st.columns(len(stages_dict))
                        for col, (k, path) in zip(sub_cols, stages_dict.items()):
                            with col:
                                img_stage = load_pil_image(path)
                                if img_stage:
                                    render_image(img_stage, caption=stage_labels.get(k, k), use_container_width=True)

                    st.markdown("---")
                    
                    legal_crop = Path(r.get("legal_crop_path") or "")
                    img_legal = load_pil_image(legal_crop)
                    if img_legal:
                        render_image(img_legal, caption="Legal crop (raw)", use_container_width=True)
                    else:
                        st.warning("Legal crop missing.")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.rglob("*"):
                z.write(p, arcname=str(p.relative_to(out_dir)))
        buf.seek(0)

        st.download_button("Download outputs.zip", data=buf.getvalue(), file_name="outputs.zip", mime="application/zip")
        st.download_button("Download predictions.csv", data=csv_path.read_bytes(), file_name="predictions.csv")
        st.download_button("Download predictions.txt", data=txt_path.read_bytes(), file_name="predictions.txt")
