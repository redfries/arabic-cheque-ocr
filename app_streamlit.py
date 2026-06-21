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
import io
import tempfile
import zipfile

import streamlit as st

from pipeline_core import PipelineConfig, ChequeOCRPipeline, write_outputs

st.set_page_config(page_title="Cheque OCR Pipeline", layout="wide")
st.title("Cheque OCR Pipeline (Detection + OCR)")

with st.sidebar:
    st.header("Model paths")
    det_weights = st.text_input("Part A Detectron2 weights (.pth)", value="models/detector/model_final.pth")
    ocr_ckpt = st.text_input("Part B OCR checkpoint (.pt)", value="models/ocr/crnn_ctc_v1/checkpoints/last.pt")

    st.header("Detection")
    det_thresh = st.slider("Detector score threshold", 0.01, 0.90, 0.30, 0.01)
    courtesy_class = st.number_input("Courtesy class id", min_value=0, max_value=10, value=0, step=1)
    legal_class = st.number_input("Legal class id (-1 = auto)", min_value=-1, max_value=10, value=-1, step=1)

    st.header("Crop + OCR")
    pad_frac = st.slider("Courtesy crop padding", 0.00, 0.20, 0.04, 0.01)
    do_line_cleanup = st.checkbox("Remove long lines", value=True)
    save_debug = st.checkbox("Save debug artifacts", value=True)

st.write("Upload 1 or more cheque images. The app shows detection overlay and OCR preprocessing stages.")

uploads = st.file_uploader(
    "Upload cheque images",
    type=["tif", "tiff", "png", "jpg", "jpeg", "bmp", "webp"],
    accept_multiple_files=True,
)

run = st.button("Run", type="primary", disabled=(not uploads or not det_weights or not ocr_ckpt))

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
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_dir = td / "inputs"
        out_dir = td / "outputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        input_paths = []
        for f in uploads:
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
        st.dataframe([{k: r.get(k) for k in ["stem", "ocr_digits", "courtesy_score", "legal_score", "status"]} for r in results])

        st.divider()
        st.subheader("Per-image visualization")

        for r in results:
            stem = r["stem"]
            with st.expander(stem, expanded=(len(results) <= 2)):
                colA, colB = st.columns([1.2, 1.0])

                with colA:
                    st.image(r["image_path"], caption="Original cheque", use_column_width=True)

                    overlay = out_dir / "overlays" / f"{stem}.png"
                    if overlay.exists():
                        st.image(str(overlay), caption="Full cheque with courtesy and legal boxes", use_column_width=True)
                    else:
                        st.warning("Overlay not found.")

                    st.markdown(f"**Digits:** `{r.get('ocr_digits','')}`")
                    if r.get("ocr_raw"):
                        st.caption(f"Raw decode: {r.get('ocr_raw')}")
                    dbg = r.get("preprocess_debug") or {}
                    if dbg:
                        st.caption(f"Line cleanup applied: {dbg.get('line_cleanup_applied')} | Ink keep ratio: {dbg.get('ink_keep_ratio')}")

                with colB:
                    crop = Path(r.get("crop_path") or "")
                    if crop.exists():
                        st.image(str(crop), caption="Courtesy crop (raw)", use_column_width=True)
                    else:
                        st.warning("Courtesy crop missing.")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in out_dir.rglob("*"):
                z.write(p, arcname=str(p.relative_to(out_dir)))
        buf.seek(0)

        st.download_button("Download outputs.zip", data=buf.getvalue(), file_name="outputs.zip", mime="application/zip")
        st.download_button("Download predictions.csv", data=csv_path.read_bytes(), file_name="predictions.csv")
        st.download_button("Download predictions.txt", data=txt_path.read_bytes(), file_name="predictions.txt")
