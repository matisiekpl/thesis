from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from torchvision import models
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64
import os

from train import transform, names

app = Flask(__name__)
CORS(app)

classes = ['BLA', 'EBO', 'EOS', 'LYT', 'MON',
           'MYB', 'NGB', 'NGS', 'PEB', 'PLM', 'PMO']
# model = models.resnet18(weights='DEFAULT')
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, len(classes))
# model.load_state_dict(torch.load(
# 'experiments/resnet18/model.pth', map_location=torch.device('cpu')))


def get_model(model_name):
    class Dataset:
        def __init__(self, classes):
            self.classes = classes
    dataset = Dataset(classes)
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

    model.load_state_dict(torch.load(
        f'experiments/{model_name}/model.pth', map_location=torch.device('cpu')))
    return model


@app.route('/models')
def list_models():
    model_names = os.listdir('experiments')
    model_names = [
        model_name for model_name in model_names if not model_name.startswith('.')]
    results = []
    for model_name in model_names:
        with open(f'experiments/{model_name}/result.txt', 'r') as f:
            results.append({
                'name': model_name,
                'result': f.read()
            })
    return results


@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if 'file' not in request.files:
        return 'No file part'

    # Validate
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return 'Invalid file'
    image_stream = BytesIO(file.read())
    I = Image.open(image_stream).convert('RGB')
    image = transform(I)
    resized = I.resize((224, 224))

    # Predict
    model = get_model(model_name)
    model.eval()
    outputs = model(image.unsqueeze(0))
    result = {}
    for i, p in enumerate(outputs[0]):
        percent = torch.nn.functional.softmax(outputs, dim=1)[0][i] * 100
        print(f'{names[classes[i]]}: {percent.item():.4f}%')
        result[names[classes[i]]] = percent.item()

    # Generate CAM

    target_layers = []
    if model_name in ['resnet18', 'resnet50', 'resnet101']:
        target_layers = [model.layer4[-1]]

    if len(target_layers) > 0:
        cam = FullGrad(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=image.unsqueeze(
            0), targets=[ClassifierOutputTarget(10)], aug_smooth=True, eigen_smooth=False)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(
            np.array(resized, np.float32)/255, grayscale_cam, use_rgb=True)
        plt.imsave('cam.png', visualization)
        encoded_cam = base64.b64encode(open('cam.png', 'rb').read())
    else:
        encoded_cam = None

    output = {
        'predictions': result,
    }
    if encoded_cam is not None:
        output['cam'] = encoded_cam.decode('utf-8'),
    return output


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists("dist/" + path):
        return send_from_directory('dist', path)
    else:
        return send_from_directory('dist', 'index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
