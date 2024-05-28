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
import math

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
    'PMO',
    'BLA',
    'NGB',
    'PLM',
    'MYB',
    'EOS',
    'MON',
    'PEB'
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
    'MON': 'Monocyt',
    'MYB': 'Mielocyt',
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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.ToPILImage(),
    transforms.RandomEqualize(1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225]),
])


def train(experiment_name, model_name, epochs=EPOCHS):
    for c in CLASSES:
        print(f'Using {c}: {names[c]}')

    experiment_path = f'./experiments/{experiment_name}'
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
    log_file = open(f'{experiment_path}/log.txt', 'w')

    def log(text):
        print(text)
        log_file.write(text + '\n')
        log_file.flush()

    log(f'Experiment: {experiment_name}')

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

    if model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'efficientnet_b3':
        model = models.efficientnet_b3(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'efficientnet_b4':
        model = models.efficientnet_b4(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'efficientnet_b5':
        model = models.efficientnet_b5(weights='DEFAULT')
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(dataset.classes))
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    device = torch.device(DEVICE)
    model.to(device)

    x1 = []
    x2 = []

    training_loss_history = []
    validation_loss_history = []

    training_f1_history = []
    validation_f1_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0

        if not DRY:
            y_true_train = []
            y_pred_train = []
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                _, predicted = torch.max(outputs, 1)
                running_correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                y_true_train.extend(labels.cpu().numpy())
                y_pred_train.extend(predicted.cpu().numpy())

                running_loss += loss.item()
            training_loss_history.append(running_loss)
            running_accuracy = running_correct / len(train_dataset)
            x1.append(epoch+1)
            f1_train = f1_score(y_true_train, y_pred_train, average='weighted')
            training_f1_history.append(f1_train)

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
        x2.append(epoch+1)

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        f1_val = f1_score(y_true_val, y_pred_val, average='weighted')
        validation_f1_history.append(f1_val)

        log(classification_report(y_true_val,
            y_pred_val, target_names=dataset.classes))
        log(f'Epoch {epoch+1}/{EPOCHS}, Validation Loss: {val_loss}, Val Accuracy: {val_accuracy}, F1: {f1_val}')

        plt.clf()
        plt.figure(figsize=(5, 7))
        plt.title('Wykres funkcji straty od epoki')
        plt.plot(x1, training_loss_history, label='Strata treningu')
        plt.plot(x2, validation_loss_history, label='Strata walidacji')
        plt.legend()
        plt.xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
        plt.xlabel('Epoka')
        plt.ylabel('Strata')
        plt.savefig(f'{experiment_path}/loss.eps',
                    bbox_inches="tight", format='eps')

        plt.clf()
        plt.figure(figsize=(5, 7))
        plt.title('Wykres F1 od epoki')
        plt.plot(x1, training_f1_history,
                 label='F1 dla danych treningowych')
        plt.plot(x2, validation_f1_history,
                 label='F1 dla danych walidacyjnych')
        plt.legend()
        plt.ylim(0, 1)
        plt.xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
        plt.xlabel('Epoka')
        plt.ylabel('F1')
        plt.savefig(f'{experiment_path}/f1.eps',
                    bbox_inches="tight", format='eps')

        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].plot(x1, training_loss_history, label='Strata treningu')
        ax[0].plot(x2, validation_loss_history, label='Strata walidacji')
        ax[0].legend()
        ax[0].set_title('Wykres funkcji straty od epoki')
        ax[0].set_xlabel('Epoka')
        ax[0].set_ylabel('Strata')
        ax[0].set_xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
        ax[1].plot(x1, training_f1_history,
                   label='F1 dla danych treningowych')
        ax[1].plot(x2, validation_f1_history,
                   label='F1 dla danych walidacyjnych')
        ax[1].legend()
        ax[1].set_title('Wykres F1 od epoki')
        ax[1].set_xlabel('Epoka')
        ax[1].set_ylabel('F1')
        ax[1].set_ylim(0, 1)
        ax[1].set_xticks(range(math.floor(min(x2)), math.ceil(max(x2))+1))
        plt.savefig(f'{experiment_path}/combined.eps',
                    bbox_inches="tight", format='eps')

        cf_matrix = confusion_matrix(y_true_val, y_pred_val)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[
                             f'{names[i]} ({i})' for i in dataset.classes], columns=[f'{names[i]} ({i})' for i in dataset.classes])
        plt.clf()
        plt.figure(figsize=(12, 7))
        sn.heatmap(df_cm, annot=True, fmt='.3f')
        plt.title('Macierz pomyłek')
        plt.savefig(f'{experiment_path}/confusion_matrix.eps',
                    bbox_inches="tight", format='eps')

        torch.save(model.state_dict(), f'{experiment_path}/model.pth')

        result_file = open(f'{experiment_path}/result.txt', 'w')
        result_file.write(classification_report(
            y_true_val, y_pred_val, target_names=[
                f'{names[i]} ({i})' for i in dataset.classes]))
        result_file.close()

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'{experiment_path}/checkpoint.pt')

    log_file.close()
    print("Training complete.")


if __name__ == '__main__':
    train('efficientnet_b0', 'efficientnet_b0')
    train('efficientnet_b1', 'efficientnet_b1')
    train('efficientnet_b2', 'efficientnet_b2')
    # train('efficientnet_b3', 'efficientnet_b3')
    train('efficientnet_b4', 'efficientnet_b4')
    # train('efficientnet_b5', 'efficientnet_b5')
    # train('densenet121', 'densenet121')
    # train('densenet169', 'densenet169')
    # train('densenet201', 'densenet201')
    # train('resnet18', 'resnet18')
    # train('vgg16', 'vgg16')
    # train('vgg19', 'vgg19')
