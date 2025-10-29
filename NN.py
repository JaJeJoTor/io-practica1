from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score





class NN(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def forward_sin_softamx(self, x):

        flat = self.flatten(x)
        return self.linear_relu(flat)
    

    def forward(self, x):
        return nn.Softmax(dim= 1)(self.forward_sin_softamx(x))


    def train_(self, datasets_train, datasets_test, epochs, batchsize=64):

        train = DataLoader(datasets_train, batch_size= batchsize, shuffle= True, num_workers= 0, pin_memory= True, drop_last= True)
        test = DataLoader(datasets_test, batch_size= 1024, num_workers= 0, pin_memory= True)

        optimizer = Adam(self.parameters(), lr= 1e-4)
        criterion = nn.CrossEntropyLoss(reduction= 'mean')

        self.to('cuda')

        losses_train = []
        losses_val = []
        n_batches = len(train)
        
        for e in tqdm(range(epochs)):
            
            self.train()
            loss_train = 0
            loss_val = 0

            for x, y in train:

                optimizer.zero_grad()

                x, y = x.to('cuda'), y.to('cuda')

                y_pred = self.forward_sin_softamx(x)

                loss = criterion(y_pred, y)

                loss_train += loss.item()

                loss.backward()

                optimizer.step()

            self.eval()

            for x, y in test:
                
                x, y = x.to('cuda'), y.to('cuda')

                y_pred = self.forward_sin_softamx(x)

                loss = criterion(y_pred, y)
                loss_val += loss.item()
                
            losses_train.append(loss_train / n_batches)
            losses_val.append(loss_val / n_batches)
            print(f"Epoch {e+1}/{epochs} | Train Loss: {losses_train[-1]:.4f} | "
              f"Val Loss: {losses_val[-1]:.4f} | Accuracy: {self.accuracy():.2f}%")
        
        plt.figure()
        plt.plot(losses_train, label="Train Loss", color="blue") 
        plt.plot(losses_val, label='Validation Loss', color='orange')   
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title('Entrenamiento: pérdida train vs validación')  
        plt.legend()
        plt.show()
            
        return losses_train, losses_val
    

    def accuracy(self, test):

        self.eval()  # modo evaluación
        correct = 0
        total = 0

        test = DataLoader(test, batch_size= 1024, num_workers= 8, pin_memory= True)

        with torch.no_grad():  # no necesitamos gradientes en test
            for images, labels in test:
                images, labels = images.to('cuda'), labels.to('cuda')
                
                outputs = self(images)
                y_pred = torch.argmax(outputs, dim= 1)

                total += labels.size(0)
                correct += (y_pred == labels).sum().item()

        return correct / total
        

if __name__ == '__main__':

    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # Transformación de imagen a tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    model = NN(28*28)
    model.train_(train_dataset, test_dataset, 15)

    torch.save(model.state_dict(), 'modelo.pth')