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
from train import CustomDataset, transform, INPUT, names
from sklearn.preprocessing import StandardScaler

dataset = CustomDataset(root_dir=INPUT, transform=transform)
model = models.efficientnet_b0(weights='DEFAULT')
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
model.load_state_dict(torch.load(
    'experiments/efficientnet_b0/model.pth', map_location=torch.device('cpu')))

print(dataset.classes)


image = Image.open('dataset/MMZ/0001-1000/MMZ_00006.jpg').convert('RGB')
# image = Image.open('sample/x6.png').convert('RGB')
# image = Image.open('cells/Im060_1/cell_12.png').convert('RGB')
image = transform(image)

scaler = StandardScaler()

model.eval()
with torch.no_grad():
    outputs = model(image.unsqueeze(0))
    # outputs = outputs[:, 1:]
    print(outputs.shape)
    _, predicted = torch.max(outputs, 1)
    for i, p in enumerate(outputs[0]):
        # print(f'{names[dataset.classes[i]]}: {p}')
        percent = torch.nn.functional.softmax(outputs, dim=1)[0][i] * 100
        print(f'{names[dataset.classes[i]]}: {percent.item():.4f}%')
    print(
        f'Predicted: {names[dataset.classes[predicted]]}')
