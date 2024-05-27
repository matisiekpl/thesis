from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request
from flask_cors import CORS
from torchvision import models
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import base64

from train import transform, names

app = Flask(__name__)
CORS(app)

classes = ['BLA', 'EBO', 'EOS', 'LYT', 'MON',
           'MYB', 'NGB', 'NGS', 'PEB', 'PLM', 'PMO']
model = models.resnet18(weights='DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(classes))
model.load_state_dict(torch.load(
    'experiments/resnet18/model.pth', map_location=torch.device('cpu')))


@app.route('/predict/<revision>', methods=['POST'])
def predict(revision):
    if 'file' not in request.files:
        return 'No file part'

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

    # file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    # resized = cv2.resize(cv2.imdecode(
    #     file_bytes, cv2.IMREAD_COLOR), (224, 224))

    cam = FullGrad(model=model, target_layers=[model.layer4[-1]])
    model.eval()
    # with torch.no_grad():
    outputs = model(image.unsqueeze(0))
    result = {}
    for i, p in enumerate(outputs[0]):
        percent = torch.nn.functional.softmax(outputs, dim=1)[0][i] * 100
        print(f'{names[classes[i]]}: {percent.item():.4f}%')
        result[names[classes[i]]] = percent.item()
    grayscale_cam = cam(input_tensor=image.unsqueeze(
        0), targets=[ClassifierOutputTarget(10)], aug_smooth=True, eigen_smooth=False)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(
        np.array(resized, np.float32)/255, grayscale_cam, use_rgb=True)
    plt.imsave('cam.png', visualization)

    encoded_cam = base64.b64encode(open('cam.png', 'rb').read())

    return {
        'predictions': result,
        'cam': encoded_cam.decode('utf-8'),
    }


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
