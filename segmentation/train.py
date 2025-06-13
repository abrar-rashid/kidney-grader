from datetime import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from segmentation.improved_unet import ImprovedUNet, CombinedLoss
from segmentation.dataset import H5SegmentationDataset, safe_collate
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

def train_model(h5_path='data/train_data.h5', 
                num_classes=5, 
                batch_size=32, 
                epochs=50, 
                lr=3e-4, 
                save_dir='checkpoints',
                checkpoint_path=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # derive model name from checkpoint path or set a default name
    if checkpoint_path:
        model_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    else:
        model_name = "current_model"
    
    dataset = H5SegmentationDataset(h5_path, is_training=True, tissue_threshold=0.05)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=safe_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=safe_collate)
    
    model = ImprovedUNet(n_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    # check for existing checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"Resuming training from checkpoint: {checkpoint_path}, starting at epoch {start_epoch+1}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights only from: {checkpoint_path}")
    else:
        print("Training from scratch.")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = CombinedLoss(num_classes=num_classes, alpha=0.5, beta=0.5, gamma=2.0)(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), avg_loss=running_loss / (batch_idx + 1))
        
        train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                main_output = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = CombinedLoss(num_classes=num_classes, alpha=0.5, beta=0.5, gamma=2.0)(main_output, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        # save best  model without overwriting, without model state dict etc
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_best_{timestamp}.pth"))
            print(f"Saved new best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # save regular checkpoints with state dicts
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f"checkpoint_{model_name}_epoch{epoch+1}_{timestamp}.pth"))

        # early stopping condition
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train_model(
        checkpoint_path='checkpoints/segmentation/kidney_grader_unet.pth',
        epochs=200,
        save_dir='checkpoints'
    )
