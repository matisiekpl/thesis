import torch
import os
from xml.etree import ElementTree
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TuberculosisDataset(torch.utils.data.Dataset):
    def __init__(self, directory='dataset', S=7, B=2, transform=None):
        self.directory = directory
        self.S = S
        self.B = B
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.directory)) / 2

    def __getitem__(self, index, single=False):
        image_filename = os.path.join(self.directory, f'tuberculosis-phone-{(index + 1):04d}.jpg')
        metadata_filename = os.path.join(self.directory, f'tuberculosis-phone-{(index + 1):04d}.xml')

        boxes = []
        root = ElementTree.parse(metadata_filename)
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            boxes.append([
                int(bbox.find('xmin').text),
                int(bbox.find('ymin').text),
                int(bbox.find('xmax').text),
                int(bbox.find('ymax').text),
            ])
        boxes = torch.tensor(boxes)
        image = Image.open(image_filename)

        if single:
            return image, boxes

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, 5))

        for box in boxes:
            x, y, width, height = box.tolist()
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S
            )

            if label_matrix[i, j, 1] == 0:
                label_matrix[i, j, 1] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[1:] = box_coordinates

        return image, label_matrix

    def show(self, index):
        image, boxes = self.__getitem__(index, True)
        fig, ax = plt.subplots()
        ax.imshow(image)
        for box in boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
        plt.show()


d = TuberculosisDataset()
# print(d.__len__())
# print(d.__getitem__(0))

d.show(0)
