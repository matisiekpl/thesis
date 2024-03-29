from io import BytesIO

import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request
from torchvision import models

from train import CustomDataset, transform, INPUT, names

app = Flask(__name__)

dataset = CustomDataset(root_dir=INPUT, transform=transform)
model = models.efficientnet_b5(weights='DEFAULT')
num_features = model.classifier[1].in_features
model.classifier = nn.Linear(num_features, len(dataset.classes))
model.load_state_dict(torch.load(
    'experiments/efficientnet_b5/model.pth', map_location=torch.device('cpu')))


@app.route('/predict/<model>', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return 'Invalid file'
    image_stream = BytesIO(file.read())
    image = transform(Image.open(image_stream))

    model.eval()
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))
        result = {}
        for i, p in enumerate(outputs[0]):
            percent = torch.nn.functional.softmax(outputs, dim=1)[0][i] * 100
            print(f'{names[dataset.classes[i]]}: {percent.item():.4f}%')
            result[names[dataset.classes[i]]] = percent.item()
        return result


if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
