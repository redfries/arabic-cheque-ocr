import os, json
import numpy as np
import cv2

def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union

def load_gt(val_coco_path):
    coco = json.load(open(val_coco_path,"r"))
    img_by_id = {im["id"]: im for im in coco["images"]}
    gt = {}
    for a in coco["annotations"]:
        gt.setdefault(a["image_id"], {})[a["category_id"]] = a["bbox"]
    return img_by_id, gt

def load_preds(pred_path):
    preds = json.load(open(pred_path,"r"))
    by_img = {}
    for p in preds:
        by_img.setdefault(p["image_id"], {}).setdefault(p["category_id"], []).append(p)

    top = {}
    for iid, d in by_img.items():
        top[iid] = {}
        for cid, plist in d.items():
            best = max(plist, key=lambda x: x.get("score", 0.0))
            top[iid][cid] = best["bbox"]
    return top

def evaluate(val_coco_path, pred_path, img_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_by_id, gt = load_gt(val_coco_path)
    pred = load_preds(pred_path)

    thresholds = [0.50, 0.75, 0.90]
    ious_all = []
    ious_by_class = {0: [], 1: []}
    miss_by_class = {0: 0, 1: 0}
    swap_count = 0
    total_imgs = len(img_by_id)

    worst = []  # (mean_iou_two_classes, image_id)

    for iid in img_by_id:
        gt0 = gt[iid].get(0, None)
        gt1 = gt[iid].get(1, None)
        p0 = pred.get(iid, {}).get(0, None)
        p1 = pred.get(iid, {}).get(1, None)

        if p0 is None:
            miss_by_class[0] += 1
        if p1 is None:
            miss_by_class[1] += 1

        i0 = iou_xywh(p0, gt0) if (p0 is not None and gt0 is not None) else 0.0
        i1 = iou_xywh(p1, gt1) if (p1 is not None and gt1 is not None) else 0.0

        if (p0 is not None) and (p1 is not None) and (gt0 is not None) and (gt1 is not None):
            normal = iou_xywh(p0, gt0) + iou_xywh(p1, gt1)
            swapped = iou_xywh(p0, gt1) + iou_xywh(p1, gt0)
            if swapped > normal:
                swap_count += 1

        ious_by_class[0].append(i0)
        ious_by_class[1].append(i1)
        ious_all.extend([i0, i1])

        worst.append(((i0 + i1) / 2.0, iid))

    mean_iou_all = float(np.mean(ious_all))
    mean_iou_0 = float(np.mean(ious_by_class[0]))
    mean_iou_1 = float(np.mean(ious_by_class[1]))

    def acc_at(t, arr):
        arr = np.array(arr)
        return float(np.mean(arr >= t))

    report = {
        "mean_iou": {"overall": mean_iou_all, "courtesy": mean_iou_0, "legal": mean_iou_1},
        "acc_iou": {
            str(t): {
                "overall": acc_at(t, ious_all),
                "courtesy": acc_at(t, ious_by_class[0]),
                "legal": acc_at(t, ious_by_class[1]),
            } for t in thresholds
        },
        "missing_rate": {
            "courtesy": miss_by_class[0] / total_imgs,
            "legal": miss_by_class[1] / total_imgs,
        },
        "swap_rate_est": swap_count / total_imgs,
        "total_images": total_imgs
    }

    worst_sorted = sorted(worst, key=lambda x: x[0])[:100]
    worst_dir = os.path.join(out_dir, "worst_100")
    os.makedirs(worst_dir, exist_ok=True)

    for mean_iou, iid in worst_sorted:
        fn = img_by_id[iid]["file_name"]
        path = os.path.join(img_root, fn)
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            continue

        def draw(im, bbox, color, label):
            if bbox is None:
                return
            x,y,w,h = map(int, bbox)
            cv2.rectangle(im, (x,y), (x+w,y+h), color, 3)
            cv2.putText(im, label, (x, max(20, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        gt0 = gt[iid].get(0); gt1 = gt[iid].get(1)
        p0 = pred.get(iid, {}).get(0); p1 = pred.get(iid, {}).get(1)

        draw(im, gt0, (0,0,255), "GT Courtesy")
        draw(im, gt1, (0,255,0), "GT Legal")
        draw(im, p0, (0,0,180), "PR Courtesy")
        draw(im, p1, (0,180,0), "PR Legal")

        outp = os.path.join(worst_dir, f"{mean_iou:.3f}_{fn}.png")
        cv2.imwrite(outp, im)

    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    print("Saved report and worst_100 overlays to:", out_dir)
