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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2

# dataset = CustomDataset(root_dir=INPUT, transform=transform)
# model = models.efficientnet_b0(weights='DEFAULT')
# num_ftrs = model.classifier[1].in_features
# model.classifier = nn.Linear(num_ftrs, len(dataset.classes))
# model.load_state_dict(torch.load(
#     'experiments/efficientnet_b0/model.pth', map_location=torch.device('cpu')))

dataset = CustomDataset(root_dir=INPUT, transform=transform)
model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(dataset.classes))
model.load_state_dict(torch.load(
    'experiments/resnet18/model.pth', map_location=torch.device('cpu')))

print(dataset.classes)

# filename = 'cell2.png'
# filename = 'sample/x6.png'
# filename = 'sample/29001-29424/NGS_29009.jpg'
filename = 'dataset/PLM/0001-1000/PLM_00010.jpg'
image = Image.open(filename).convert('RGB')
# image = Image.open('sample/x6.png').convert('RGB')
# image = Image.open('cells/Im060_1/cell_12.png').convert('RGB')
transformed_image = transform(image)

scaler = StandardScaler()

model.eval()
# with torch.no_grad():
cam = FullGrad(model=model, target_layers=[model.layer4[-1]])

outputs = model(transformed_image.unsqueeze(0))

grayscale_cam = cam(input_tensor=transformed_image.unsqueeze(
    0), targets=[ClassifierOutputTarget(10)], aug_smooth=True, eigen_smooth=False)

image = cv2.imread(filename)
resized = cv2.resize(image, (224, 224))

grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(
    np.array(resized, np.float32)/255, grayscale_cam, use_rgb=True)

# outputs = outputs[:, 1:]
print(outputs.shape)
_, predicted = torch.max(outputs, 1)
for i, p in enumerate(outputs[0]):
    # print(f'{names[dataset.classes[i]]}: {p}')
    percent = torch.nn.functional.softmax(outputs, dim=1)[0][i] * 100
    print(f'{names[dataset.classes[i]]}: {percent.item():.4f}%')
print(
    f'Predicted: {names[dataset.classes[predicted]]}')

plt.imsave('cam.png', visualization)

o = 0.3
combinated = cv2.addWeighted(resized, o, visualization, 1-o, 1)

plt.imsave('combinated.png', cv2.cvtColor(combinated, cv2.COLOR_BGR2RGB))
