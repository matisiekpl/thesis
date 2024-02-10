import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolo
from dataset import TuberculosisDataset
from loss import YoloLoss

seed = 123
torch.manual_seed(seed)

LEARNING_RATE = 2e-5
DEVICE = "mps"
BATCH_SIZE = 16
WEIGHT_DECAY = 0
EPOCHS = 1000


class Compose(object):
    def __init__(self, t):
        self.transforms = t

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(), ])

model = Yolo(S=7).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = YoloLoss()
train_dataset = TuberculosisDataset(transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)

for epoch in range(EPOCHS):
    mean_loss = []
    for idx, (l, r) in enumerate(tqdm(train_loader)):
        x, y = l.to(DEVICE), r.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Mean loss: {sum(mean_loss) / len(mean_loss)}")
