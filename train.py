import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
class SpectrumDataset(Dataset):
        def __init__(self, inputs_dir, labels_dir, transform=None):
            self.inputs_dir = inputs_dir
            self.labels_dir = labels_dir
            self.transform = transform

            self.input_files = sorted([f for f in os.listdir(inputs_dir) if f.endswith('.png')])
        
        def __len__(self):
            return len(self.input_files)
        
        def __getitem__(self, idx):
            input_filename = self.input_files[idx]
            number = input_filename.split('_')[1].split('.')[0]
            
            img_path = os.path.join(self.inputs_dir, input_filename)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
            

            label_path = f".\dataset\labels\label_{number}.pt"
            label = torch.load(label_path)
            label = label.view(-1)
            
            return image, label

if __name__ == '__main__':

    input_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225])
    ])

    inputs_dir = './dataset/inputs'
    labels_dir = './dataset/labels'
    dataset = SpectrumDataset(inputs_dir, labels_dir, transform=input_transform)

    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


    num_elements = 99

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    model.fc = nn.Linear(2048, num_elements)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()  # shape: [batch, 99]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            print("train_loss: ", loss.item())

            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'model.pth')
    print("학습 완료. 모델을 저장했습니다.")