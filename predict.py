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
model = models.efficientnet_b5(weights='DEFAULT')
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
model.load_state_dict(torch.load(
    'experiments/efficientnet_b5/model.pth', map_location=torch.device('cpu')))


# image = Image.open('dataset/LYT/1001-2000/LYT_01006.jpg').convert('RGB')
image = Image.open('x.png').convert('RGB')
image = transform(image)

scaler = StandardScaler()

model.eval()
with torch.no_grad():
    outputs = model(image.unsqueeze(0))
    print(outputs)
    _, predicted = torch.max(outputs, 1)
    for i, p in enumerate(outputs[0]):
        print(f'{names[dataset.classes[i]]}: {p}')
    print(
        f'Predicted: {names[dataset.classes[predicted]]}')
