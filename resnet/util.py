import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar100_data(batch_size=64):
    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images with mean and std
    ])

    # Load CIFAR-100 training dataset
    trainset = torchvision.datasets.CIFAR100(root='../data', train=True,
                                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)

    # Load CIFAR-100 test dataset
    testset = torchvision.datasets.CIFAR100(root='../data', train=False,
                                             download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False) 

    return trainloader, testloader
