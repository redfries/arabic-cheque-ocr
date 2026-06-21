# -*- coding: utf-8 -*-
"""
Cheque OCR pipeline (Part A detector -> Part B OCR).

No GT required.
Detects BOTH courtesy and legal boxes, crops courtesy, runs OCR, saves full visualization artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import re
import time

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("opencv-python (cv2) is required for this pipeline.") from e

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required for this pipeline.") from e


IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(input_path: Union[str, Path]) -> List[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    paths: List[Path] = []
    for ext in IMG_EXTS:
        paths.extend(p.rglob(f"*{ext}"))
    return sorted(paths)


def safe_stem(p: Union[str, Path]) -> str:
    s = Path(p).stem
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
    return s or "image"


def read_image_any(path: Union[str, Path]) -> np.ndarray:
    """Read image robustly for .tif and common formats. Returns BGR uint8."""
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    try:
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PIL is required to read this image type.") from e
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def preprocess_for_detection(
    bgr: np.ndarray,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    clahe_grid: Tuple[int, int] = (8, 8),
    denoise: str = "none",
    median_ksize: int = 3,
) -> np.ndarray:
    """grayscale + CLAHE + optional median, then back to BGR."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=tuple(clahe_grid))
        gray = clahe.apply(gray)

    if denoise == "median":
        if median_ksize < 3 or median_ksize % 2 == 0:
            raise ValueError("median_ksize must be odd and >= 3")
        gray = cv2.medianBlur(gray, int(median_ksize))
    elif denoise == "none":
        pass
    else:
        raise ValueError("denoise must be 'none' or 'median'")

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# -----------------------------
# Crop preprocessing for OCR (stage outputs)
# -----------------------------
def whiten_border(img_u8: np.ndarray, frac: float = 0.03, min_px: int = 2, max_px: int = 8) -> np.ndarray:
    h, w = img_u8.shape
    b = int(round(frac * min(h, w)))
    b = max(min_px, min(max_px, b))
    out = img_u8.copy()
    out[:b, :] = 255
    out[-b:, :] = 255
    out[:, :b] = 255
    out[:, -b:] = 255
    return out


def remove_long_lines(img_u8: np.ndarray, h_kernel: int = 45, v_kernel: int = 25, thr: int = 220) -> np.ndarray:
    x = img_u8.copy()
    mask = (x < thr).astype(np.uint8) * 255

    hk = max(15, int(h_kernel))
    vk = max(15, int(v_kernel))
    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

    hlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hker)
    vlines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vker)

    lines = cv2.bitwise_or(hlines, vlines)
    out = x.copy()
    out[lines > 0] = 255
    return out


def enhance_crop_for_ocr_v3(img_u8_gray: np.ndarray, **cfg) -> np.ndarray:
    x = img_u8_gray.copy()

    median_k = int(cfg.get("median_k", 3))
    if median_k and median_k >= 3:
        if median_k % 2 == 0:
            median_k += 1
        x = cv2.medianBlur(x, median_k)

    inv = 255 - x
    h, w = inv.shape

    k = int(cfg.get("k_open", 31))
    k = max(15, k)
    if k % 2 == 0:
        k += 1
    k = min(k, max(15, (min(h, w) // 2) | 1))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bg = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    ink = cv2.subtract(inv, bg)
    ink = cv2.normalize(ink, None, 0, 255, cv2.NORM_MINMAX)

    enh = 255 - ink

    if bool(cfg.get("clahe", True)):
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh = c.apply(enh)

    ink_boost = float(cfg.get("ink_boost", 1.30))
    if ink_boost != 1.0:
        inv2 = (255.0 - enh.astype(np.float32))
        inv2 = np.clip(inv2 * ink_boost, 0, 255)
        enh = (255.0 - inv2).astype(np.uint8)

    sharpen = float(cfg.get("sharpen", 0.05))
    if sharpen > 0:
        blur = cv2.GaussianBlur(enh, (0, 0), 0.8)
        enh = cv2.addWeighted(enh, 1.0 + sharpen, blur, -sharpen, 0)

    return enh


def resize_keep_aspect(img_u8: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img_u8.shape
    if h == target_h:
        return img_u8
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(img_u8, (new_w, target_h), interpolation=interp)


def preprocess_crop_array(
    img_gray: np.ndarray,
    enh_cfg: dict,
    target_h: int,
    do_line_cleanup: bool = True,
    ink_thr: int = 220,
    min_ink_keep_ratio: float = 0.45,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict[str, Union[bool, float]]]:
    stages: Dict[str, np.ndarray] = {}
    dbg: Dict[str, Union[bool, float]] = {"line_cleanup_applied": False, "ink_keep_ratio": 1.0}

    raw = img_gray.copy()
    stages["raw"] = raw

    border = whiten_border(raw, frac=0.03, min_px=2, max_px=8)
    stages["border"] = border

    chosen = border
    if do_line_cleanup:
        before_ink = float((border < ink_thr).sum())
        lines = remove_long_lines(border, h_kernel=45, v_kernel=25, thr=ink_thr)
        stages["lines_removed"] = lines
        after_ink = float((lines < ink_thr).sum())
        keep_ratio = (after_ink / max(before_ink, 1.0))
        dbg["ink_keep_ratio"] = keep_ratio
        if keep_ratio >= float(min_ink_keep_ratio):
            chosen = lines
            dbg["line_cleanup_applied"] = True
        else:
            chosen = border
            dbg["line_cleanup_applied"] = False

    enh = enhance_crop_for_ocr_v3(chosen, **enh_cfg)
    stages["enhanced"] = enh

    enh_rs = resize_keep_aspect(enh, target_h=target_h)
    stages["enhanced_resized"] = enh_rs

    return stages, enh_rs, dbg


# -----------------------------
# OCR Model: CRNN + CTC
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.act(x + y)


class StrongCRNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.s1 = nn.Sequential(ResBlock(base), ResBlock(base), nn.MaxPool2d(2, 2))
        self.proj2 = nn.Sequential(nn.Conv2d(base, base * 2, 1, bias=False), nn.BatchNorm2d(base * 2))
        self.s2 = nn.Sequential(ResBlock(base * 2), ResBlock(base * 2), nn.MaxPool2d(2, 2))
        self.proj3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 1, bias=False), nn.BatchNorm2d(base * 4))
        self.s3 = nn.Sequential(ResBlock(base * 4), ResBlock(base * 4), nn.MaxPool2d((2, 1), (2, 1)))
        self.s4 = nn.Sequential(ResBlock(base * 4), ResBlock(base * 4), nn.MaxPool2d((2, 1), (2, 1)))

    def forward(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.proj2(x)
        x = self.s2(x)
        x = self.proj3(x)
        x = self.s3(x)
        x = self.s4(x)
        return x


def fmap_to_sequence(fmap: torch.Tensor) -> torch.Tensor:
    B, C, Hf, Wf = fmap.shape
    seq = fmap.permute(0, 3, 1, 2).contiguous()
    return seq.view(B, Wf, C * Hf)


class CRNNHead(nn.Module):
    def __init__(self, in_feat: int, hidden: int, num_classes: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=in_feat,
            hidden_size=hidden,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, seq: torch.Tensor):
        out, _ = self.rnn(seq)
        logits = self.fc(out)
        log_probs = logits.log_softmax(dim=-1)
        return log_probs.permute(1, 0, 2).contiguous()


class CRNN_CTC(nn.Module):
    def __init__(self, num_classes: int, target_h: int = 48, base: int = 64, rnn_hidden: int = 256):
        super().__init__()
        self.encoder = StrongCRNNEncoder(in_ch=1, base=base)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, target_h, 200)
            fmap = self.encoder(dummy)
            in_feat = int(fmap.shape[1] * fmap.shape[2])
        self.head = CRNNHead(in_feat=in_feat, hidden=rnn_hidden, num_classes=num_classes)

    def forward(self, images: torch.Tensor):
        fmap = self.encoder(images)
        seq = fmap_to_sequence(fmap)
        return self.head(seq)


def decode_greedy_ctc(pred_idx_seq: List[int], blank_idx: int, idx2char: Dict[int, str]) -> str:
    out: List[str] = []
    prev = None
    for x in pred_idx_seq:
        if x == blank_idx:
            prev = x
            continue
        if prev == x:
            continue
        out.append(idx2char.get(int(x), ""))
        prev = x
    return "".join(out)


_ARABIC_INDIC = str.maketrans({
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})


def digits_only_normalized(s: str) -> str:
    s = (s or "").translate(_ARABIC_INDIC)
    s = s.replace("*", "").replace("#", "")
    return "".join(ch for ch in s if "0" <= ch <= "9")


def meta_path_from_ckpt(ckpt_path: Path) -> Path:
    return ckpt_path.parent.parent / "metadata.json"


def load_model_from_ckpt(ckpt_path: Union[str, Path], device: torch.device):
    ckpt_path = Path(ckpt_path)
    mp = meta_path_from_ckpt(ckpt_path)
    if not mp.exists():
        raise FileNotFoundError(f"Missing metadata.json for {ckpt_path}: {mp}")

    meta = json.load(open(mp, "r", encoding="utf-8"))
    charset = meta["charset"]
    blank_idx = int(meta.get("blank_idx", 0))
    enh_cfg = meta.get("enh_cfg", {})
    target_h = int(meta.get("target_h", 48))

    idx2char = {i + 1: ch for i, ch in enumerate(charset)}
    idx2char[blank_idx] = ""
    num_classes = len(charset) + 1

    model = CRNN_CTC(num_classes=num_classes, target_h=target_h).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return model, meta, idx2char, blank_idx, enh_cfg, target_h


# -----------------------------
# Detectron2 detector
# -----------------------------
class Detectron2Unavailable(RuntimeError):
    pass


def build_predictor(det_weights: Union[str, Path], score_thresh: float, device_str: str = "auto"):
    try:
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.engine import DefaultPredictor  # type: ignore
        from detectron2 import model_zoo  # type: ignore
    except Exception as e:  # pragma: no cover
        raise Detectron2Unavailable(
            "detectron2 is required for Part A inference. Install detectron2 in your environment."
        ) from e

    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.WEIGHTS = str(det_weights)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(score_thresh)
    cfg.MODEL.DEVICE = device_str

    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def pick_top_box(instances, class_id: int) -> Tuple[Optional[List[float]], Optional[float]]:
    if len(instances) == 0:
        return None, None
    classes = instances.pred_classes.detach().cpu().numpy()
    scores = instances.scores.detach().cpu().numpy()
    boxes = instances.pred_boxes.tensor.detach().cpu().numpy()

    idx = np.where(classes == class_id)[0]
    if len(idx) == 0:
        return None, None
    best = idx[np.argmax(scores[idx])]
    return boxes[best].tolist(), float(scores[best])


def clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1i = max(0, min(int(round(x1)), W - 1))
    y1i = max(0, min(int(round(y1)), H - 1))
    x2i = max(0, min(int(round(x2)), W))
    y2i = max(0, min(int(round(y2)), H))
    if x2i <= x1i:
        x2i = min(W, x1i + 1)
    if y2i <= y1i:
        y2i = min(H, y1i + 1)
    return x1i, y1i, x2i, y2i


# -----------------------------
# Pipeline
# -----------------------------
@dataclass
class PipelineConfig:
    det_weights: str
    ocr_ckpt: str

    det_score_thresh: float = 0.30
    det_device: str = "auto"

    courtesy_pred_class: int = 0
    legal_pred_class: Optional[int] = None  # None => use other class (0<->1)

    pad_frac: float = 0.04
    do_line_cleanup: bool = True


class ChequeOCRPipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.predictor, _ = build_predictor(
            det_weights=cfg.det_weights,
            score_thresh=cfg.det_score_thresh,
            device_str=cfg.det_device,
        )

        self.ocr_model, self.ocr_meta, self.idx2char, self.blank_idx, self.enh_cfg, self.target_h = load_model_from_ckpt(
            cfg.ocr_ckpt, device=self.device
        )

        if cfg.legal_pred_class is None:
            self.legal_class = 1 if int(cfg.courtesy_pred_class) == 0 else 0
        else:
            self.legal_class = int(cfg.legal_pred_class)

    def process_one(self, img_path: Union[str, Path], out_dir: Union[str, Path], save_debug: bool = True) -> Dict:
        t0 = time.time()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "crops").mkdir(exist_ok=True)
        (out_dir / "stages").mkdir(exist_ok=True)
        (out_dir / "overlays").mkdir(exist_ok=True)
        (out_dir / "json").mkdir(exist_ok=True)

        img_path = Path(img_path)
        stem = safe_stem(img_path)

        bgr = read_image_any(img_path)
        bgr_det = preprocess_for_detection(bgr, use_clahe=True, denoise="none")

        det_out = self.predictor(bgr_det)
        inst = det_out["instances"]

        courtesy_box, courtesy_score = pick_top_box(inst, int(self.cfg.courtesy_pred_class))
        legal_box, legal_score = pick_top_box(inst, int(self.legal_class))

        result: Dict = {
            "stem": stem,
            "image_path": str(img_path),
            "courtesy_xyxy": courtesy_box,
            "courtesy_score": courtesy_score,
            "legal_xyxy": legal_box,
            "legal_score": legal_score,
            "overlay_path": str(out_dir / "overlays" / f"{stem}.png"),
            "crop_path": "",
            "stage_paths": {},
            "ocr_raw": "",
            "ocr_digits": "",
            "status": "ok",
            "time_sec": None,
            "preprocess_debug": {},
        }

        overlay_path = Path(result["overlay_path"])
        H, W = bgr.shape[:2]

        def draw_box(vis, box, color_bgr, label):
            if box is None:
                return None
            x1, y1, x2, y2 = box
            x1i, y1i, x2i, y2i = clip_xyxy(x1, y1, x2, y2, W, H)
            cv2.rectangle(vis, (x1i, y1i), (x2i, y2i), color_bgr, 2)
            cv2.putText(vis, label, (x1i, max(25, y1i - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2)
            return (x1i, y1i, x2i, y2i)

        # No boxes
        if courtesy_box is None and legal_box is None:
            result["status"] = "missing_detection"
            if save_debug:
                cv2.imwrite(str(overlay_path), bgr)
            result["time_sec"] = round(time.time() - t0, 4)
            (out_dir / "json" / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        # Courtesy missing
        if courtesy_box is None:
            result["status"] = "missing_courtesy_box"
            if save_debug:
                vis = bgr.copy()
                draw_box(vis, legal_box, (0, 255, 0), f"LEGAL {legal_score:.2f}" if legal_score is not None else "LEGAL")
                cv2.imwrite(str(overlay_path), vis)
            result["time_sec"] = round(time.time() - t0, 4)
            (out_dir / "json" / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result

        # Crop courtesy with padding
        x1, y1, x2, y2 = courtesy_box
        bw = x2 - x1
        bh = y2 - y1
        x1 -= self.cfg.pad_frac * bw
        x2 += self.cfg.pad_frac * bw
        y1 -= self.cfg.pad_frac * bh
        y2 += self.cfg.pad_frac * bh
        x1i, y1i, x2i, y2i = clip_xyxy(x1, y1, x2, y2, W, H)

        crop_bgr = bgr[y1i:y2i, x1i:x2i]
        crop_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        crop_path = out_dir / "crops" / f"{stem}.png"
        cv2.imwrite(str(crop_path), crop_gray)
        result["crop_path"] = str(crop_path)

        # Stage preprocessing
        stages, enh_rs, dbg = preprocess_crop_array(
            crop_gray, enh_cfg=self.enh_cfg, target_h=self.target_h, do_line_cleanup=self.cfg.do_line_cleanup
        )
        result["preprocess_debug"] = dbg

        stage_paths: Dict[str, str] = {}
        for k, im in stages.items():
            sp = out_dir / "stages" / f"{stem}_{k}.png"
            cv2.imwrite(str(sp), im)
            stage_paths[k] = str(sp)
        result["stage_paths"] = stage_paths

        # OCR
        try:
            x = torch.from_numpy(enh_rs).float().div(255.0).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                log_probs = self.ocr_model(x)
                pred = torch.argmax(log_probs, dim=-1).squeeze(1).detach().cpu().numpy().tolist()
            raw = decode_greedy_ctc(pred, self.blank_idx, self.idx2char)
            digits = digits_only_normalized(raw)
            result["ocr_raw"] = raw
            result["ocr_digits"] = digits
        except Exception:
            result["status"] = "ocr_failed"

        # Overlay full cheque
        if save_debug:
            vis = bgr.copy()
            cxy = draw_box(vis, courtesy_box, (0, 0, 255), f"COURTESY {courtesy_score:.2f}" if courtesy_score is not None else "COURTESY")
            draw_box(vis, legal_box, (0, 255, 0), f"LEGAL {legal_score:.2f}" if legal_score is not None else "LEGAL")
            label = result["ocr_digits"] if result.get("ocr_digits") else "(empty)"
            if cxy is not None:
                cx1, cy1, cx2, cy2 = cxy
                cv2.putText(vis, f"PRED: {label}", (cx1, min(H - 10, cy2 + 30)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.imwrite(str(overlay_path), vis)

        result["time_sec"] = round(time.time() - t0, 4)
        (out_dir / "json" / f"{stem}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def process_many(self, inputs: List[Union[str, Path]], out_dir: Union[str, Path], save_debug: bool = True) -> List[Dict]:
        results: List[Dict] = []
        for p in inputs:
            try:
                results.append(self.process_one(p, out_dir=out_dir, save_debug=save_debug))
            except Exception as e:
                results.append({
                    "stem": safe_stem(p),
                    "image_path": str(p),
                    "courtesy_xyxy": None,
                    "courtesy_score": None,
                    "legal_xyxy": None,
                    "legal_score": None,
                    "overlay_path": "",
                    "crop_path": "",
                    "stage_paths": {},
                    "ocr_raw": "",
                    "ocr_digits": "",
                    "status": f"failed: {type(e).__name__}",
                    "time_sec": None,
                    "preprocess_debug": {},
                })
        return results


def write_outputs(results: List[Dict], out_dir: Union[str, Path]):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    csv_path = out_dir / "predictions.csv"
    fields = ["stem", "image_path", "ocr_digits", "ocr_raw", "courtesy_score", "legal_score", "status", "crop_path", "overlay_path"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in fields})

    txt_path = out_dir / "predictions.txt"
    lines = [f"{r.get('stem','')}\t{r.get('ocr_digits','')}" for r in results]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "total": len(results),
        "ok": sum(1 for r in results if r.get("status") == "ok"),
        "missing_detection": sum(1 for r in results if r.get("status") == "missing_detection"),
        "missing_courtesy_box": sum(1 for r in results if r.get("status") == "missing_courtesy_box"),
        "ocr_failed": sum(1 for r in results if r.get("status") == "ocr_failed"),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return csv_path, txt_path
