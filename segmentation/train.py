import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from segmentation.improved_unet import ImprovedUNet, CombinedLoss
from segmentation.dataset import H5SegmentationDataset, safe_collate
import os
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_model(h5_path='data/train_data.h5', 
                num_classes=5, 
                batch_size=8, 
                epochs=50, 
                lr=3e-4, 
                save_dir='checkpoints'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = H5SegmentationDataset(h5_path, is_training=True, tissue_threshold=0.05)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=safe_collate)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=safe_collate)
    
    model = ImprovedUNet(n_classes=num_classes).to(device)
    criterion = CombinedLoss(num_classes=num_classes, alpha=0.5, beta=0.5, gamma=2.0)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training")
        
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
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
                loss = criterion(main_output, masks)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "improved_unet_best.pth"))
            print(f"Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pth"))

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    train_model()
