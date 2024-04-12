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

# Load the ImageNet dataset
imagenet_dataset = datasets.ImageNet(root='path_to_imagenet_dataset', split='train', transform=transform)

# Create a DataLoader to iterate over the dataset in batches
batch_size = 32
imagenet_dataloader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True)

# Example usage:
for images, labels in imagenet_dataloader:
    # Process the batch of images and labels here
    plt.imshow(images[0])
    plt.show()


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class AlexNet(nn.Module):

    def __init__(self) -> None:
        super().__init__()


        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.ReLU(),
        )
        
        self.layer6 = nn.Sequential(
            nn.Linear(1, 2048),
            nn.ReLU(),
        )

        self.layet7 = nn.Sequential(
            nn.Linear(2048, 1000),
            nn.ReLU(),
        )

        def forward(x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = self.layer6(x)
            x = self.layer7(x)

             

