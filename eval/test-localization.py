"""
Evaluate per-object anomaly-localization segmentors (DRAEM-style) against the
real MVTec test set.

Port of O2MAG `eval/test-localization.py`. For each object, loads the
``<object>.pckl`` checkpoint saved by `train-localization.py`, predicts anomaly
masks on the real MVTec ``test/<defect>`` images, and scores them against the
real ``ground_truth/<defect>/<img>_mask.png`` masks.

Metrics (per defect type and object-mean), computed with the pure-numpy
`au_pro_util.calculate_aupro` and sklearn-free PR curves here:

  image-level: AUROC, AP, F1_max  — scored over the FULL MVTec test set for the
                object, including the real `test/good` images as negatives (this
                is what makes the detection AUROC meaningful).
  pixel-level: AUROC, AP, F1_max, AU-PRO — per defect type.

Output: writes ``<checkpoint_path>/result.csv`` with one row per (object,
defect_type) and a ``mean`` row, plus per-image detail printed to stdout.

Usage
-----
    uv run eval/test-localization.py \
        --mvtec_path /home/luca_piai/big_disk/datasets/mvtec \
        --generated_data_path /home/luca_piai/big_disk/datasets/generated \
        --checkpoint_path eval/checkpoints/localization \
        --categories cable screw transistor leather hazelnut pill tile \
        --device cuda:1
"""

import argparse
import csv
import os

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader

from unet_utils.au_pro_util import calculate_aupro
from unet_utils.data_loader import AnomalyLocalizationTestDataset

CATEGORIES = ["cable", "screw", "transistor", "leather", "hazelnut", "pill", "tile", "carpet", "capsule", "wood"]


def _eval_defect_names(args, obj_name):
    """Resolve defect types for evaluation of one object.

    Returns ``(anomaly_names, all_test_names)`` where:
      * ``anomaly_names`` is the **intersection** of the defect subfolders present
        in the real MVTec ``test/`` set and those present in the generated images
        folder — we can only score a defect we both trained on (generated) and
        have real GT for (MVTec).
      * ``all_test_names`` is ``anomaly_names`` plus the MVTec ``good`` (normal)
        test folder when it exists; those images act as image-level negatives.
    """
    mvtec_test = os.path.join(args.mvtec_path, obj_name, "test")
    gen_test = os.path.join(args.generated_data_path, obj_name, "test")

    mvtec_defects = (
        {
            d
            for d in os.listdir(mvtec_test)
            if os.path.isdir(os.path.join(mvtec_test, d)) and d != "good"
        }
        if os.path.isdir(mvtec_test)
        else set()
    )
    gen_defects = (
        {d for d in os.listdir(gen_test) if os.path.isdir(os.path.join(gen_test, d))}
        if os.path.isdir(gen_test)
        else set()
    )

    anomaly_names = sorted(mvtec_defects & gen_defects)

    all_test_names = list(anomaly_names)
    if os.path.isdir(mvtec_test) and "good" in os.listdir(mvtec_test):
        all_test_names.append("good")
    return anomaly_names, all_test_names


def _metrics_for(anomaly_name, gts, preds):
    """Compute pixel-level AUROC/AP/F1 + AU-PRO for one defect type."""
    # Pixel-level.
    flat_gt = np.concatenate([g.ravel() for g in gts])
    flat_pred = np.concatenate([p.ravel() for p in preds])
    p_auroc, p_ap, p_f1 = _pixel_pr_metrics(flat_gt, flat_pred)

    # AU-PRO (official MVTec PRO: connected-component overlap, integrated to FPR 0.3).
    au_pro, _, _ = calculate_aupro(
        [g.astype(np.float32) for g in gts],
        [p.astype(np.float32) for p in preds],
    )
    return {
        "defect": anomaly_name,
        "n": len(gts),
        "pixel_AUROC": p_auroc,
        "pixel_AP": p_ap,
        "pixel_F1": p_f1,
        "pixel_AUPRO": au_pro,
    }


def _image_level_metrics(grouped_gts, grouped_preds):
    """Image-level AUROC/AP/F1 over the FULL object test set (defects + good).

    Each image's anomaly score is the max of its predicted mask *after* a 21x21
    average-pooling smoothing (O2MAG applies ``F.avg_pool2d(out_mask_sm[:,1:],
    21, stride=1, padding=10)`` before taking the max); the image label is 1 if
    its GT mask has any positive pixel. Including the real MVTec ``test/good``
    images as negatives is what makes a meaningful detection AUROC computable
    (the per-defect loop alone has no negatives, so its AUROC collapses to 0).
    """
    img_scores = []
    img_gt = []
    for defect in grouped_gts:
        for g, p in zip(grouped_gts[defect], grouped_preds[defect]):
            # 21x21 average-pool (zero-padded) matches O2MAG's avg_pool2d.
            smoothed = uniform_filter(p, size=21, mode="constant", cval=0.0)
            img_scores.append(float(smoothed.max()))
            img_gt.append(1.0 if g.max() > 0 else 0.0)
    return _image_pr_metrics(np.array(img_gt), np.array(img_scores))


def _image_pr_metrics(gt, scores):
    """AUROC + AP + best-F1 for a binary image-score problem (numpy-only)."""
    n = len(gt)
    if n == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(-scores)
    gt_s = gt[order]
    tp = 0.0
    fp = 0.0
    n_pos = float(gt.sum())
    n_neg = float(n - n_pos)

    # AUC-ROC via rank sum (Mann-Whitney U).
    order_all = np.argsort(scores)
    ranks = np.empty_like(order_all, dtype=np.float64)
    ranks[order_all] = np.arange(1, n + 1, dtype=np.float64)
    sum_pos = ranks[gt == 1].sum() if n_pos > 0 else 0.0
    u = sum_pos - n_pos * (n_pos + 1.0) / 2.0
    auc = (u / max(n_pos * n_neg, 1e-12)) if n_pos > 0 and n_neg > 0 else 0.0
    auroc = float(auc)

    # Precision-recall + best F1 (sweep thresholds = unique scores).
    best_f1 = 0.0
    ap = 0.0
    prev_rec = 0.0
    for i in range(n):
        tp += gt_s[i]
        if gt_s[i] == 1:
            # recall increased -> accumulate the PR trapezoid area (prec * drec)
            rec = tp / n_pos
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            ap += prec * (rec - prev_rec)
            prev_rec = rec
        else:
            fp += 1
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / n_pos if n_pos > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            best_f1 = max(best_f1, f1)
    # include final threshold (everything positive)
    if n_pos > 0:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    return auroc, float(ap), float(best_f1)


def _pixel_pr_metrics(gt_flat, pred_flat):
    """Pixel-level AUROC + AP + best-F1."""
    gt_flat = gt_flat.astype(np.float32)
    pred_flat = pred_flat.astype(np.float32)
    n = len(gt_flat)
    n_pos = float(gt_flat.sum())
    n_neg = float(n - n_pos)
    auroc = 0.0
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(pred_flat)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        sum_pos = ranks[gt_flat == 1].sum()
        u = sum_pos - n_pos * (n_pos + 1.0) / 2.0
        auroc = float(u / (n_pos * n_neg))

    order = np.argsort(-pred_flat)
    gt_s = gt_flat[order]
    tp = 0.0
    fp = 0.0
    best_f1 = 0.0
    ap = 0.0
    prev_rec = 0.0
    for i in range(n):
        tp += gt_s[i]
        if gt_s[i] == 1:
            rec = tp / n_pos
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            ap += prec * (rec - prev_rec)
            prev_rec = rec
        else:
            fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_pos if n_pos > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    return auroc, float(ap), float(best_f1)


@torch.inference_mode()
def predict_object(args, obj_name, all_test_names, ckpt_path):
    from unet_utils.model_unet import DiscriminativeSubNetwork

    model = DiscriminativeSubNetwork(in_channels=3, out_channels=1)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(args.device).eval()

    dataset = AnomalyLocalizationTestDataset(args, obj_name, all_test_names)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    grouped_gts = {d: [] for d in all_test_names}
    grouped_preds = {d: [] for d in all_test_names}
    for image, mask, defect in dataloader:
        image = image.to(args.device)
        pred = model(image)
        pred = pred.float().cpu().numpy()[:, 0]  # (B, H, W)
        masks_np = mask.numpy()[:, 0]
        for d, m, p in zip(defect, masks_np, pred):
            grouped_gts[d].append(m.astype(np.float32))
            grouped_preds[d].append(p.astype(np.float32))
    return grouped_gts, grouped_preds


def evaluate_object(args, obj_name, anomaly_names, all_test_names, ckpt_path):
    print(f"  eval {obj_name}")
    grouped_gts, grouped_preds = predict_object(
        args, obj_name, all_test_names, ckpt_path
    )
    rows = []
    for defect in anomaly_names:
        gts = grouped_gts.get(defect, [])
        preds = grouped_preds.get(defect, [])
        if not gts:
            continue
        m = _metrics_for(defect, gts, preds)
        rows.append(m)
        print(
            f"    {defect:12s} n={m['n']:>3d} "
            f"pxAUROC={m['pixel_AUROC']:.3f} "
            f"pxAP={m['pixel_AP']:.3f} pxF1={m['pixel_F1']:.3f} "
            f"pxPRO={m['pixel_AUPRO']:.3f}"
        )
    if rows:
        avg_px_auroc = float(np.mean([r["pixel_AUROC"] for r in rows]))
        avg_px_ap = float(np.mean([r["pixel_AP"] for r in rows]))
        avg_px_f1 = float(np.mean([r["pixel_F1"] for r in rows]))
        avg_px_pro = float(np.mean([r["pixel_AUPRO"] for r in rows]))
        print(
            f"    [pixel-avg] AUROC={avg_px_auroc:.3f} AP={avg_px_ap:.3f} "
            f"F1={avg_px_f1:.3f} PRO={avg_px_pro:.3f}"
        )
    i_auroc, i_ap, i_f1 = _image_level_metrics(grouped_gts, grouped_preds)
    print(
        f"    [image-level] AUROC={i_auroc:.3f} AP={i_ap:.3f} F1={i_f1:.3f}"
    )
    return rows, (i_auroc, i_ap, i_f1)


def main():
    parser = argparse.ArgumentParser(
        description="Test per-object anomaly localization segmentors."
    )
    parser.add_argument("--device", type=str, default="cuda:1", help="e.g. cuda:1")
    parser.add_argument(
        "--mvtec_path", type=str, required=True, help="Path to real MVTec dataset"
    )
    parser.add_argument(
        "--generated_data_path",
        type=str,
        required=True,
        help="Path to generated defect images (used for defect-type names)",
    )
    parser.add_argument(
        "--checkpoint_path", default="./eval/checkpoints/localization", type=str
    )
    parser.add_argument("--categories", type=str, nargs="+", default=CATEGORIES)
    parser.add_argument("--bs", type=int, default=64)
    args = parser.parse_args()

    all_rows = []
    image_summary = []  # (obj_name, (auroc, ap, f1))
    for obj_name in args.categories:
        anomaly_names, all_test_names = _eval_defect_names(args, obj_name)
        print(f"{obj_name}: defects={anomaly_names}")

        ckpt = os.path.join(args.checkpoint_path, obj_name + ".pckl")
        if not os.path.isfile(ckpt):
            print(f"  -- no checkpoint for {obj_name} at {ckpt}, skipping")
            continue
        if not anomaly_names:
            print("  -- no common defect types between MVTec and generated, skipping")
            continue
        rows, image_metrics = evaluate_object(
            args, obj_name, anomaly_names, all_test_names, ckpt
        )
        all_rows.extend((obj_name, r) for r in rows)
        image_summary.append((obj_name, image_metrics))

    _write_csv(args, all_rows, image_summary)
    _print_summary(all_rows, image_summary)


def _write_csv(args, all_rows, image_summary):
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    csv_path = os.path.join(args.checkpoint_path, "result.csv")
    fields = [
        "object", "defect", "n",
        "image_AUROC", "image_AP", "image_F1",
        "pixel_AUROC", "pixel_AP", "pixel_F1", "pixel_AUPRO",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for obj_name, r in all_rows:
            row = {"object": obj_name}
            row.update(r)
            w.writerow(row)
        # One object-level row per image-level summary (uses real test/good as
        # negatives, so these are the meaningful detection metrics).
        for obj_name, (auroc, ap, f1) in image_summary:
            w.writerow(
                {
                    "object": obj_name,
                    "defect": "image-level",
                    "image_AUROC": auroc,
                    "image_AP": ap,
                    "image_F1": f1,
                }
            )
    print(f"  wrote {csv_path}")


def _print_summary(all_rows, image_summary):
    metrics = [
        "pixel_AUROC", "pixel_AP", "pixel_F1", "pixel_AUPRO",
    ]
    for metric in metrics:
        vals = [
            r[metric] for _, r in all_rows if isinstance(r.get(metric), float)
        ]
        mean = float(np.mean(vals)) if vals else float("nan")
        print(f"  mean {metric:14s}: {mean:.4f}")
    print("  image-level (full test set incl. good):")
    for label, idx in [("image_AUROC", 0), ("image_AP", 1), ("image_F1", 2)]:
        vals = [m[idx] for _, m in image_summary]
        mean = float(np.mean(vals)) if vals else float("nan")
        print(f"    mean {label:14s}: {mean:.4f}")


if __name__ == "__main__":
    main()
