import torch

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn 
import matplotlib.pyplot as plt

# Define the transformation to be applied to the images
transform = transforms.Compose([
    transforms.Resize(256),                         # Resize the input image to 256x256
    transforms.CenterCrop(224),                     # Crop the center 224x224 region
    transforms.ToTensor(),                          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class AlexNet(nn.Module):

    def __init__(self, num_classes: int=1000) -> None:
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
        
        self.layer6 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU()
        )

        self.layer7 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(),
        )

        self.layer8 = nn.Sequential(
            nn.Linear(2048, num_classes),
        )

        self.loss = nn.CrossEntropyLoss()


    def forward(self, x, y):
        print("PRESHAPE:", x.shape)
        x = self.layer1(x)
        print("post layer 1: ", x.shape)
        x = self.layer2(x)
        print("post layer 2: ", x.shape)
        x = self.layer3(x)
        print("post layer 3: ", x.shape)
        x = self.layer4(x)
        print("post layer 4: ", x.shape)
        x = self.layer5(x)
        print("post layer 5: ", x.shape)
        x = self.layer6(x)
        print("post layer 6: ", x.shape)
        x = self.layer7(x)
        print("post layer 7: ", x.shape)
        logits = self.layer8(x)            
        print("post layer 8: ", x.shape)
        loss = self.loss(logits, y)
         
        
        return logits, loss 
        
 

