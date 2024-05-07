import torch
import torchvision
import torchvision.transforms as transforms
from model import AlexNet

learning_rate = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10
grad_clip = 5

def get_data():
    train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.ToTensor())
    test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms.ToTensor())
    
    return train, test


def train(model, data):

    train_loader, test_loader = get_data()
    
class Trainer:
    
    def __init__(self, model, train_data):
        
        self.model = model
        self.model.to(device)
        self.train_data = train_data
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    def train(self):
        model = self.model 
        
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        
        model.train()
        data_iter = iter(train_loader) 
        
        for epoch in range(epochs):
           
            # Get next minibatch of data 
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
                
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
    train_data, _ = get_data()
    
    trainer = Trainer(model, train_data)
    trainer.train()
    
    # Save model 
    torch.save(model.state_dict(), 'model.pth') 
    
