# -*- coding: utf-8 -*-
"""
CLI runner for cheque pipeline.

Example:
  python run_cheque_pipeline.py --input /path/to/images --det-weights model_final.pth --ocr-ckpt best_digitacc.pt --out out_run
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from pipeline_core import PipelineConfig, ChequeOCRPipeline, list_images, write_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a single image OR a folder of images")
    ap.add_argument("--det-weights", default="models/detector/model_final.pth", help="Detectron2 Part A weights .pth (default: models/detector/model_final.pth)")
    ap.add_argument("--ocr-ckpt", default="models/ocr/crnn_ctc_v1/checkpoints/last.pt", help="Part B OCR checkpoint .pt (default: models/ocr/crnn_ctc_v1/checkpoints/last.pt)")
    ap.add_argument("--out", required=True, help="Output folder for this run")
    ap.add_argument("--det-thresh", type=float, default=0.30, help="Detector score threshold")
    ap.add_argument("--pad-frac", type=float, default=0.04, help="Padding around courtesy bbox (fraction of box size)")
    ap.add_argument("--courtesy-class", type=int, default=0, help="Predicted class id to treat as courtesy")
    ap.add_argument("--legal-class", type=int, default=-1, help="Predicted class id to treat as legal. Use -1 to auto other class.")
    ap.add_argument("--no-line-cleanup", action="store_true", help="Disable long-line removal for OCR crops")
    ap.add_argument("--no-debug", action="store_true", help="Do not save overlay images and stages")
    
    # Legal OCR arguments
    ap.add_argument("--legal-ocr-ckpt", default="models/ocr/legal", help="Part B Legal OCR adapter checkpoint (default: models/ocr/legal)")
    ap.add_argument("--legal-ocr-base", default="models/Qwen3.5_model", help="Part B Legal OCR base model (default: models/Qwen3.5_model)")
    ap.add_argument("--legal-pad-frac", type=float, default=0.05, help="Padding around legal bbox (fraction of box size)")
    ap.add_argument("--no-fallback", action="store_true", help="Disable image enhancement fallback logic for legal amount recognition")
    
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list_images(args.input)
    if len(paths) == 0:
        raise SystemExit(f"No images found under: {args.input}")

    legal = None if int(args.legal_class) < 0 else int(args.legal_class)

    cfg = PipelineConfig(
        det_weights=args.det_weights,
        ocr_ckpt=args.ocr_ckpt,
        det_score_thresh=args.det_thresh,
        courtesy_pred_class=args.courtesy_class,
        legal_pred_class=legal,
        pad_frac=args.pad_frac,
        do_line_cleanup=not args.no_line_cleanup,
        legal_ocr_ckpt=args.legal_ocr_ckpt,
        legal_ocr_base=args.legal_ocr_base,
        legal_pad_frac=args.legal_pad_frac,
        do_fallback=not args.no_fallback,
    )

    pipe = ChequeOCRPipeline(cfg)
    results = pipe.process_many(paths, out_dir=out_dir, save_debug=not args.no_debug)
    csv_path, txt_path = write_outputs(results, out_dir=out_dir)

    (out_dir / "config_used.json").write_text(json.dumps(cfg.__dict__, indent=2), encoding="utf-8")

    print("Done.")
    print("Images:", len(paths))
    print("CSV:", csv_path)
    print("TXT:", txt_path)
    print("Artifacts:", out_dir)


if __name__ == "__main__":
    main()
