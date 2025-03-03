import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import numpy as np

# Config
DATA_PATH = "./data/animals-10"  
BATCH_SIZE = 32
EPOCHS = 20  
NUM_CLASSES = 10
VAL_SPLIT = 0.2  

# Path checking
if not os.path.exists(DATA_PATH):
    print(f"Error: {DATA_PATH} not found!")
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

# Transformation 
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Validation data
full_dataset = datasets.ImageFolder(DATA_PATH, transform=train_transform)
train_size = int((1 - VAL_SPLIT) * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model creating
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# Loss function
class_counts = [len(os.listdir(os.path.join(DATA_PATH, cls))) for cls in full_dataset.classes]
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) 

# Training
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Епоха {epoch+1}/{EPOCHS}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss/len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = correct / len(val_dataset)
    
    print(f"Validation | Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Saving best trained model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
    
    scheduler.step() 

# Saving model to animal_classifier.pth
torch.save(model.state_dict(), "animal_classifier.pth")
