import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from app.models.capsule_resnet import CapsuleResNetSegNet
from app.models.dedswin import DEDSWINNet
from app.utils.data_loader import LitsDataset
from app.utils.metrics import get_all_metrics
import time

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(loader):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            outputs = model(images)
            # Binary mask segmentation (Liver + Tumor)
            # Standard Dice + Cross-Entropy Loss
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        if i % 10 == 0:
            print(f"Batch {i} Loss: {loss.item():.4f}")
            
    return total_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            # Batch average metrics
            batch_metrics = [get_all_metrics(p, t) for p, t in zip(preds, targets_np)]
            all_metrics.extend(batch_metrics)
            
    # Calculate means
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum([m[key] for m in all_metrics]) / len(all_metrics)
    
    return avg_metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Select architecture (User requirement: Two Models)
    model = DEDSWINNet(num_classes=3).to(device)
    
    # Dataset and Loader
    train_set = LitsDataset("data/raw/lits", mode='train')
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    
    # Dice + Cross-Entropy Loss (Standard for medical imaging imbalance)
    criterion = nn.CrossEntropyLoss() # In production: combine with DiceLoss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    
    num_epochs = 50
    print(f"Starting Training on LiTS-2017 Dataset...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Time: {time.time()-start_time:.2f}s")
        
        if (epoch + 1) % 5 == 0:
            # val_metrics = validate(model, val_loader, device)
            # print(f"Validation Metrics: {val_metrics}")
            torch.save(model.state_dict(), f"data/processed/checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
