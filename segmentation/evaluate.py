import torch
from torch.utils.data import DataLoader
import numpy as np
from dataset import H5SegmentationDataset
from improved_unet import ImprovedUNet
from sklearn.metrics import jaccard_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def run_evaluation(checkpoint_path="checkpoints/improved_unet_best.pth", 
                  h5_path="data/train_data.h5", 
                  num_classes=5,
                  output_dir="evaluation_results"):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.load(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    dataset = H5SegmentationDataset(h5_path, is_training=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []
    class_names = ['Background', 'Tubuli', 'Vein', 'Artery', 'Glomeruli']

    print("Running evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get main output if using deep supervision
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(masks.cpu().numpy())

    y_true = np.concatenate([y.flatten() for y in all_targets])
    y_pred = np.concatenate([y.flatten() for y in all_preds])

    acc = accuracy_score(y_true, y_pred)
    miou = jaccard_score(y_true, y_pred, average='macro')
    class_iou = jaccard_score(y_true, y_pred, average=None, labels=range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print(f"\nEvaluation Results:")
    print(f"  - Overall Pixel Accuracy: {acc:.4f}")
    print(f"  - Mean IoU: {miou:.4f}")
    
    print("\nClass-wise Results:")
    for i in range(num_classes):
        print(f"  - {class_names[i]}:")
        print(f"    - IoU: {class_iou[i]:.4f}")
        print(f"    - Precision: {precision[i]:.4f}")
        print(f"    - Recall: {recall[i]:.4f}")
        print(f"    - F1-Score: {f1[i]:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'IoU': class_iou,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    metrics_df.to_csv(f"{output_dir}/class_metrics.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='IoU', data=metrics_df)
    plt.title('IoU Scores by Class')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/iou_scores.png")
    
    with open(f"{output_dir}/evaluation_summary.txt", 'w') as f:
        f.write(f"Overall Pixel Accuracy: {acc:.4f}\n")
        f.write(f"Mean IoU: {miou:.4f}\n\n")
        f.write("Class-wise Results:\n")
        for i in range(num_classes):
            f.write(f"{class_names[i]}:\n")
            f.write(f"  IoU: {class_iou[i]:.4f}\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall: {recall[i]:.4f}\n")
            f.write(f"  F1-Score: {f1[i]:.4f}\n\n")
    
    print(f"\nEvaluation complete. Results saved to {output_dir}/")
    return miou

if __name__ == "__main__":
    run_evaluation()