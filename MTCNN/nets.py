import torch
import torch.nn as nn
import numpy as np

class PNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,10,3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3)
        )
        self.cls = nn.Conv2d(32,1,1)
        self.box = nn.Conv2d(32,4,1)

    def forward(self,x):
        layer = self.layer(x)
        cls = torch.sigmoid(self.cls(layer))
        box = self.box(layer)

        return cls,box

class RNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,28,3,1,1,bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(28,48,3,1,0,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(48,64,2,1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3*3*64,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3*3*64,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        layer = self.layer(x)
        layer = layer.reshape(-1,64*3*3)
        cls = torch.sigmoid(self.fc1(layer))
        box = self.fc2(layer)

        return cls, box


class ONet(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(3,32,3,1,0,bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1),
            nn.Conv2d(32,64,3,1,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.Conv2d(64,64,3,1,0,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,2,1,bias=False)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(3*3*128,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(3*3*128,256),
            nn.ReLU(),
            nn.Linear(256,4)
        )

        self.fc3 = nn.Sequential(
            nn.Linear(3*3*128,256),
            nn.ReLU(),
            nn.Linear(256,10)
        )

    def forward(self,x):
        layer = self.layer(x)
        layer = layer.reshape(-1,128*3*3)
        cls = torch.sigmoid(self.fc1(layer))
        box = self.fc2(layer)
        kp = self.fc3(layer)

        return cls,box,kp


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = PNet().to(device)
    input1 = torch.randn(1,3,12,12).to(device)
    output1 = model(input1)
    a=  output1[0]
    b = output1[1].cpu()
    print(output1[0].shape)
    print(output1[1].shape)

    input2 = torch.randn(1,3,24,24).to(device)
    model = RNet().to(device)
    output2 = model(input2)
    print(output2[0].shape)
    print(output2[1].shape)

    input3 = torch.randn(1,3,48,48).to(device)
    model = ONet().to(device)
    output3 = model(input3)
    print(output3[0].shape)
    print(output3[1].shape)
    print(output3[2].shape)





