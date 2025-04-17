import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import jaccard_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from tqdm import tqdm
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation
from medpy.metric.binary import hd, assd

from segmentation.segment import post_process_predictions
from segmentation.utils import get_binary_class_mask, get_instance_mask
from segmentation.improved_unet import ImprovedUNet
from segmentation.dataset import H5SegmentationDataset
from segmentation.config import CLASS_NAMES, NUM_CLASSES

def compute_boundary_iou(pred_mask, true_mask, class_idx, boundary_width=3):
    pred_bin = get_binary_class_mask(pred_mask, class_idx)
    true_bin = get_binary_class_mask(true_mask, class_idx)

    pred_boundary = find_boundaries(pred_bin, mode='inner')
    true_boundary = find_boundaries(true_bin, mode='inner')

    pred_dil = binary_dilation(pred_boundary, iterations=boundary_width)
    true_dil = binary_dilation(true_boundary, iterations=boundary_width)

    intersection = np.logical_and(pred_boundary, true_dil).sum() + \
                   np.logical_and(true_boundary, pred_dil).sum()

    union = pred_boundary.sum() + true_boundary.sum()
    return intersection / (union + 1e-6)

def compute_boundary_f1(pred_mask, true_mask, class_idx, boundary_width=3):
    pred_bin = get_binary_class_mask(pred_mask, class_idx)
    true_bin = get_binary_class_mask(true_mask, class_idx)

    pred_boundary = find_boundaries(pred_bin, mode='inner')
    true_boundary = find_boundaries(true_bin, mode='inner')

    pred_dil = binary_dilation(pred_boundary, iterations=boundary_width)
    true_dil = binary_dilation(true_boundary, iterations=boundary_width)

    tp = np.logical_and(pred_boundary, true_dil).sum()
    fp = np.logical_and(pred_boundary, np.logical_not(true_dil)).sum()
    fn = np.logical_and(true_boundary, np.logical_not(pred_dil)).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def compute_dice(pred_mask, true_mask, class_idx):
    pred_bin = get_binary_class_mask(pred_mask, class_idx)
    true_bin = get_binary_class_mask(true_mask, class_idx)
    intersection = (pred_bin * true_bin).sum()
    return (2. * intersection) / (pred_bin.sum() + true_bin.sum() + 1e-6)

def compute_hausdorff_and_assd(pred_mask, true_mask, class_idx):
    pred_bin = get_binary_class_mask(pred_mask, class_idx)
    true_bin = get_binary_class_mask(true_mask, class_idx)

    if pred_bin.sum() == 0 or true_bin.sum() == 0:
        return None, None

    return hd(pred_bin, true_bin), assd(pred_bin, true_bin)

def match_instances(pred_mask, gt_mask, iou_thresh=0.5):
    matched = set()
    tp = 0
    fp = 0
    fn = 0

    pred_ids = np.unique(pred_mask)
    gt_ids = np.unique(gt_mask)
    pred_ids = pred_ids[pred_ids != 0]
    gt_ids = gt_ids[gt_ids != 0]

    for pid in pred_ids:
        p = pred_mask == pid
        ious = []
        for gid in gt_ids:
            if gid in matched:
                continue
            g = gt_mask == gid
            intersection = np.logical_and(p, g).sum()
            union = np.logical_or(p, g).sum()
            iou = intersection / union if union > 0 else 0
            ious.append((iou, gid))
        ious.sort(reverse=True)
        if ious and ious[0][0] >= iou_thresh:
            matched.add(ious[0][1])
            tp += 1
        else:
            fp += 1

    fn = len(gt_ids) - len(matched)
    return tp, fp, fn

def compute_ap(tp, fp, fn):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def compute_map(pred_mask, gt_mask, thresholds=np.arange(0.5, 1.0, 0.05)):
    aps = []
    for thresh in thresholds:
        tp, fp, fn = match_instances(pred_mask, gt_mask, iou_thresh=thresh)
        ap = compute_ap(tp, fp, fn)
        aps.append(ap)
    return np.mean(aps)

def run_evaluation(checkpoint_path="checkpoints/improved_unet_best.pth", 
                   h5_path="data/train_data.h5",
                   output_dir="evaluation_results"):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ImprovedUNet(n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = H5SegmentationDataset(h5_path, is_training=False, tissue_threshold=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []

    boundary_ious = [[] for _ in range(NUM_CLASSES)]
    boundary_f1s = [[] for _ in range(NUM_CLASSES)]
    hausdorff_dists = [[] for _ in range(NUM_CLASSES)]
    assd_dists = [[] for _ in range(NUM_CLASSES)]
    dice_scores = [[] for _ in range(NUM_CLASSES)]
    instance_map_scores = [[] for _ in range(NUM_CLASSES)]
    instance_precision = [[] for _ in range(NUM_CLASSES)]
    instance_recall = [[] for _ in range(NUM_CLASSES)]
    absolute_count_error = [[] for _ in range(NUM_CLASSES)]
    percentage_count_error = [[] for _ in range(NUM_CLASSES)]

    print("Running evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            raw_pred = torch.softmax(outputs.squeeze(0).cpu(), dim=0).numpy()
            h, w = raw_pred.shape[1:]
            slide_map = {0: (0, 0, w, h)}
            pred_np = post_process_predictions([raw_pred], slide_map, (h, w))

            mask_np = masks.cpu().numpy()[0]

            all_preds.append(pred_np)
            all_targets.append(mask_np)

            for cls in range(1, NUM_CLASSES):
                biou = compute_boundary_iou(pred_np, mask_np, cls)
                bf1 = compute_boundary_f1(pred_np, mask_np, cls)
                dice = compute_dice(pred_np, mask_np, cls)
                boundary_ious[cls].append(biou)
                boundary_f1s[cls].append(bf1)
                dice_scores[cls].append(dice)

                hd_dist, assd_dist = compute_hausdorff_and_assd(pred_np, mask_np, cls)
                if hd_dist is not None:
                    hausdorff_dists[cls].append(hd_dist)
                if assd_dist is not None:
                    assd_dists[cls].append(assd_dist)

                pred_inst, pred_count = get_instance_mask(pred_np, cls)
                gt_inst, gt_count = get_instance_mask(mask_np, cls)
                pred_count, gt_count = int(pred_count), int(gt_count)
                map_score = compute_map(pred_inst, gt_inst)
                instance_map_scores[cls].append(map_score)

                tp, fp, fn = match_instances(pred_inst, gt_inst)
                prec = tp / (tp + fp + 1e-6)
                rec = tp / (tp + fn + 1e-6)
                instance_precision[cls].append(prec)
                instance_recall[cls].append(rec)
                absolute_count_error[cls].append(float(abs(pred_count - gt_count)))
                if gt_count > 0:
                    pct_error = 100.0 * abs(pred_count - gt_count) / gt_count
                    percentage_count_error[cls].append(pct_error)

    y_true = np.concatenate([y.flatten() for y in all_targets])
    y_pred = np.concatenate([y.flatten() for y in all_preds])

    acc = accuracy_score(y_true, y_pred)
    miou = jaccard_score(y_true, y_pred, average='macro')
    class_iou = jaccard_score(y_true, y_pred, average=None, labels=range(NUM_CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(NUM_CLASSES), zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    for i in range(NUM_CLASSES):
        print(f"\nClass: {CLASS_NAMES[i]}")
        print(f"  IoU: {class_iou[i]:.4f}")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall: {recall[i]:.4f}")
        print(f"  F1: {f1[i]:.4f}")
        if dice_scores[i]:
            print(f"  Dice: {np.mean(dice_scores[i]):.4f}")
        if boundary_ious[i]:
            print(f"  Boundary IoU: {np.mean(boundary_ious[i]):.4f}")
        if boundary_f1s[i]:
            print(f"  Boundary F1: {np.mean(boundary_f1s[i]):.4f}")
        if hausdorff_dists[i]:
            print(f"  Hausdorff: {np.mean(hausdorff_dists[i]):.2f}")
        if assd_dists[i]:
            print(f"  ASSD: {np.mean(assd_dists[i]):.2f}")
        if instance_map_scores[i]:
            print(f"  Instance mAP: {np.mean(instance_map_scores[i]):.4f}")
        if instance_precision[i]:
            print(f"  Instance Precision: {np.mean(instance_precision[i]):.4f}")
        if instance_recall[i]:
            print(f"  Instance Recall: {np.mean(instance_recall[i]):.4f}")
        if absolute_count_error[i]:
            print(f"  Abs Count Error: {np.mean(absolute_count_error[i]):.2f}")
        if percentage_count_error[i]:
            print(f"  % Count Error: {np.mean(percentage_count_error[i]):.2f}%")

    df = pd.DataFrame({
        "Class": CLASS_NAMES,
        "IoU": class_iou,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Dice": [np.mean(d) if d else None for d in dice_scores],
        "BoundaryIoU": [np.mean(b) if b else None for b in boundary_ious],
        "BoundaryF1": [np.mean(b) if b else None for b in boundary_f1s],
        "Hausdorff": [np.mean(h) if h else None for h in hausdorff_dists],
        "ASSD": [np.mean(a) if a else None for a in assd_dists],
        "Instance_mAP": [np.mean(m) if m else None for m in instance_map_scores],
        "Instance_Precision": [np.mean(p) if p else None for p in instance_precision],
        "Instance_Recall": [np.mean(r) if r else None for r in instance_recall],
        "Abs_Count_Error": [np.mean(e) if e else None for e in absolute_count_error],
        "%_Count_Error": [np.mean(p) if p else None for p in percentage_count_error],
        "Support": support
    })

    df.to_csv(os.path.join(output_dir, "class_metrics.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    return df

if __name__ == "__main__":
    run_evaluation()
