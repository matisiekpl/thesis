import torch
from torchvision import transforms
import torchstain
import cv2
import matplotlib.pyplot as plt

target1 = cv2.cvtColor(cv2.imread(
    "dataset/BLA/0001-1000/BLA_00001.jpg"), cv2.COLOR_BGR2RGB)
target2 = cv2.cvtColor(cv2.imread(
    "cell0.png"), cv2.COLOR_BGR2RGB)
target3 = cv2.cvtColor(cv2.imread(
    "cell0.png"), cv2.COLOR_BGR2RGB)

T = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomEqualize(1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x*255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
# normalizer.fit(T(target1))

norm1, H1, E1 = normalizer.normalize(
    I=T(target1), stains=True, Io=240, alpha=1, beta=0.15)
norm2, H2, E2 = normalizer.normalize(
    I=T(target3), stains=True, Io=240, alpha=1, beta=0.15)


fig, ax = plt.subplots(5, 2, figsize=(5, 15))

ax[0, 0].imshow(target1)
ax[0, 0].set_title("Oryginał A")

ax[0, 1].imshow(target3)
ax[0, 1].set_title("Oryginał B")

ax[1, 0].imshow(norm1)
ax[1, 0].set_title("Normalizowany A")

ax[1, 1].imshow(norm2)
ax[1, 1].set_title("Normalizowany B")

ax[2, 0].imshow(H1)
ax[2, 0].set_title("Hematoksylina A")

ax[2, 1].imshow(H2)
ax[2, 1].set_title("Hematoksylina B")

ax[3, 0].imshow(E1)
ax[3, 0].set_title("Eozyna A")

ax[3, 1].imshow(E2)
ax[3, 1].set_title("Eozyna B")

ax[4, 0].imshow(T(target1).int().permute(1, 2, 0))
ax[4, 0].set_title("Wyrównany histogram A")

ax[4, 1].imshow(T(target3).int().permute(1, 2, 0))
ax[4, 1].set_title("Wyrównany histogram B")

plt.savefig('stain.png')
