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
DRY = False
INPUT = "/kaggle/input/bone-marrow-cell-classification/bone_marrow_cell_dataset"
CLASSES = [
    'NGS',
    'EBO',
    'LYT',
    'ART',
    'PMO',
    'BLA',
    'NGB',
    'PLM',
    'MYB',
]
BATCH_SIZE = 16

if platform == "darwin":
    DEVICE = "mps"

if os.path.isdir('dataset'):
    INPUT = 'dataset'

names = {
    'ABE': 'Nieprawidłowy eozynofil',
    'ART': 'Artefakt',
    'BAS': 'Bazofil',
    'BLA': 'Blast',
    'EBO': 'Erytroblast',
    'EOS': 'Eozynofil',
    'FGC': 'Fagocyt',
    'HAC': 'Włochata komórka',
    'KSC': 'Cienie komórkowe',
    'LYI': 'Niedojrzały limfocyt',
    'LYT': 'Limfocyt',
    'MMZ': 'Metamielocyt',
    'MON': 'Monocyte',
    'MYB': 'Monocyt',
    'NGB': 'Krwinka biała pałeczkowata',
    'NGS': 'Segmentowany neutrofil',
    'NIF': 'Brak rozpoznania',
    'OTH': 'Inna komórka',
    'PEB': 'Proerytroblast',
    'PLM': 'Komórka plazmatyczna',
    'PMO': 'Promielocyt',
}


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes, self.class_to_idx = self.find_classes()
        self.images = self.load_images()

    def find_classes(self):
        classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(
            os.path.join(self.root_dir, d))]
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
                    images.append((os.path.join(root, filename),
                                  self.class_to_idx[class_name]))
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
        log_file.flush()

    log(f'Experiment: {experiment_name}')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    dataset = CustomDataset(root_dir=INPUT, transform=transform)
    log(f'Classes: {dataset.classes}')
    log(f'Number of images: {len(dataset)}')

    operating_size = int(DATASET_PART * len(dataset))
    operating_rest_size = len(dataset) - operating_size
    operating_dataset, _ = random_split(
        dataset, [operating_size, operating_rest_size])

    train_size = int(0.8 * len(operating_dataset))
    val_size = len(operating_dataset) - train_size
    train_dataset, val_dataset = random_split(
        operating_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

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

    training_accuracy_history = []
    validation_accuracy_history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0

        if not DRY:
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                running_correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            training_loss_history.append(running_loss)
            running_accuracy = running_correct / len(train_dataset)
            training_accuracy_history.append(running_accuracy)

        log(f'Epoch {epoch+1}/{EPOCHS}, Training Loss: {running_loss/len(train_loader)}')

        model.eval()
        val_loss = 0.0
        val_correct = 0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        validation_loss_history.append(val_loss)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        validation_accuracy_history.append(val_accuracy)
        f1_val = f1_score(y_true_val, y_pred_val, average='weighted')

        log(classification_report(y_true_val,
            y_pred_val, target_names=dataset.classes))
        log(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}, F1: {f1_val}')

        cf_matrix = confusion_matrix(y_true_val, y_pred_val)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[
                             f'{names[i]} ({i})' for i in dataset.classes], columns=[f'{names[i]} ({i})' for i in dataset.classes])
        plt.cla()
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True)
        plt.title('Macierz pomyłek')
        plt.savefig(f'{experiment_path}/confusion_matrix.png',
                    bbox_inches="tight")

        plt.cla()
        plt.title('Wykres funkcji straty od epoki')
        plt.plot(training_loss_history, label='Strata treningu')
        plt.plot(validation_loss_history, label='Strata walidacji')
        plt.legend()
        plt.savefig(f'{experiment_path}/loss.png', bbox_inches="tight")

        plt.cla()
        plt.title('Wykres dokładności od epoki')
        plt.plot(training_accuracy_history,
                 label='Dokładność dla danych treningowych')
        plt.plot(validation_accuracy_history,
                 label='Dokładność dla danych walidacyjnych')
        plt.legend()
        plt.savefig(f'{experiment_path}/acc.png', bbox_inches="tight")

        torch.save(model.state_dict(), f'{experiment_path}/model.pth')

        result_file = open(f'{experiment_path}/result.txt', 'w')
        result_file.write(classification_report(
            y_true_val, y_pred_val, target_names=[
                f'{names[i]} ({i})' for i in dataset.classes]))
        result_file.close()

    log_file.close()
    print("Training complete.")


if __name__ == '__main__':
    train('efficientnet_b5', 'efficientnet_b5')
