import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding):
        super(CNNBlock, self).__init__()
        self.convolution = nn.Conv2d(input_channels, output_channels,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=padding,
                                     bias=False)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.convolution(x)))


class Yolo(nn.Module):
    def __init__(self, S, input_channels=3):
        super(Yolo, self).__init__()
        self.input_channels = input_channels
        self.darknet = self.create_darknet()
        self.fcs = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (2 * 5)),
        )

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def create_darknet(self):
        return nn.Sequential(
            CNNBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            CNNBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            CNNBlock(192, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # 4 repeats
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),

            # End of repeats
            CNNBlock(512, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # 2 repeats
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),

            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )
