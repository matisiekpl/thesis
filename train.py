import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms, models
import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from PIL import Image
from sys import platform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 3
LR = 0.001
DATASET_PART = 1
INPUT = "/kaggle/input/bone-marrow-cell-classification/bone_marrow_cell_dataset"
CLASSES = ['ABE','FGC', 'LYI']

if platform == "darwin":
    DEVICE = "mps"
    
if os.path.isdir('dataset'):
    INPUT = 'dataset'
    

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes()
        self.images = self.load_images()

    def find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        if len(CLASSES) > 0:
            classes = CLASSES
        classes = sorted(classes)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx

    def load_images(self):
        images = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for root, _, filenames in os.walk(class_path):
                for filename in filenames:
                    images.append((os.path.join(root, filename), self.class_to_idx[class_name]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_path, label = self.images[idx]
            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            img_path, label = self.images[0]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, label

def train(experiment_name, model_name):
    experiment_path = f'./experiments/{experiment_name}'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_file = open(f'{experiment_path}/log.txt', 'w')
    
    def log(text):
        print(text)
        log_file.write(text + '\n')
        
    
    log(f'Experiment: {experiment_name}')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(root_dir=INPUT, transform=transform)
    log(f'Classes: {dataset.classes}')
    log(f'Number of images: {len(dataset)}')
    
    operating_size = int(DATASET_PART * len(dataset))
    operating_rest_size = len(dataset) - operating_size
    operating_dataset, _ = random_split(dataset, [operating_size, operating_rest_size])
    
    train_size = int(0.8 * len(operating_dataset))
    val_size = len(operating_dataset) - train_size
    train_dataset, val_dataset = random_split(operating_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    if model_name == 'efficientnet_b5':
        model = models.efficientnet_b5(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    device = torch.device(DEVICE)
    model.to(device)
    
    training_loss_history = []
    validation_loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        training_loss_history.append(running_loss)

        log(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {running_loss/len(train_loader)}')

        model.eval()
        val_loss = 0.0
        correct = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        validation_loss_history.append(val_loss)


        val_loss /= len(val_loader)
        accuracy = correct / len(val_dataset)
        f1_val = f1_score(y_true_val, y_pred_val, average='weighted')
        
        log(classification_report(y_true_val, y_pred_val, target_names=dataset.classes))
        log(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, Accuracy: {accuracy}, F1: {f1_val}')
        
        cf_matrix = confusion_matrix(y_true_val, y_pred_val)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in dataset.classes], columns = [i for i in dataset.classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(f'{experiment_path}/confusion_matrix.png')

    plt.cla()
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.savefig(f'{experiment_path}/loss.png')

    log_file.close()
    print("Training complete.")

if __name__ == '__main__':
    train('efficientnet_b5', 'efficientnet_b5')