import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import AlexNet
import os

learning_rate = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
grad_clip = 5

def get_data():

    data_transforms = {
    "train": transforms.Compose([transforms.ToTensor()]),
    "val": transforms.Compose([transforms.ToTensor()]),
    "test": transforms.Compose([transforms.ToTensor()]),
    }

    data_dir = "../data/tiny-224"
    num_workers = {"train": 2, "val": 0, "test": 0}
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ["train", "val", "test"]
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers[x])
        for x in ["train", "val", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    print(dataloaders["train"])

    return dataloaders["train"], dataloaders["test"] 


def train(model, data):

    train_loader, test_loader = get_data()
    
class Trainer:
    
    def __init__(self, model, train_data, test_data):
        
        self.model = model
        self.model.to(device)
        self.train_data = train_data
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(self):
        model = self.model 
         
        model.train()
        
        for epoch in range(epochs):
            
            for batch in self.train_data:

                # Move data to device
                batch = [t.to(device) for t in batch]
                x, y = batch 
                print(x.shape, y.shape) 
                # Forward pass 
                logits, self.loss = model(x, y) 
                
                # Backward pass
                model.zero_grad()
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                self.optimizer.step()
                
                # Log training progress
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {self.loss.item()}')
                
                
if __name__ == '__main__': 
    model = AlexNet(10)
    train_data, test_data = get_data()
    
    trainer = Trainer(model, train_data, test_data)
    trainer.train()
    
    # Save model 
    torch.save(model.state_dict(), 'model.pth') 
    
