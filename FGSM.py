import torch

from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from NN import NN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fgsm_attack(model: NN, criterion, images, labels, epsilon=0.05):
    # Establecemos que a la hora de hacer loss.backward() también calcule 
    # el gradiente del input (la imagen) con la propiedad requires_grad = True.
    images.requires_grad = True

    # Predecimos y calculamos el gradiente.
    labels_pred = model.forward_sin_softamx(images)
    loss = criterion(labels_pred, labels)
    model.zero_grad()
    loss.backward()

    # Modificamos la imagen con el método de FGSM y lo normalizamos entre -1 y 1.
    modification = epsilon * images.grad.data.sign()
    modified_images= images + modification
    modified_images = torch.clamp(modified_images, -1, 1)
    return modified_images


def get_adversarial_images(model: NN, dataset, criterion= nn.CrossEntropyLoss(reduction='mean'), epsilon= 0.05):

    modified_images = None
    labels = None
    # Hacemos ataque.
    for images, labels_ in dataset:
        images, labels_ = images.to(device), labels_.to(device)
        if modified_images is None:
            modified_images = fgsm_attack(model, criterion, images, labels_, epsilon=epsilon)
            labels = labels_
        else:
            modified_images = torch.cat([modified_images, fgsm_attack(model, criterion, images, labels_, epsilon=epsilon)], dim= 0)
            labels = torch.cat([labels, labels_], dim= 0)
    
    return modified_images, labels
        
    
if __name__ == "__main__":
    # Cargamos el modelo.
    model = NN(28*28)
    model.to(device)
    model.load_state_dict(torch.load("modelo.pth",map_location=device, weights_only=True))
    model.eval()

    # Cargamos el dataset.
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_dataset = DataLoader(test_dataset, batch_size= 256, pin_memory= True)
    
    # Calculamos los distintos accuracies para distintas epsilons.
    
# TODO: Gráficas y mirar a partir de qué épsilon se nota.
    
    for epsilon in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:

        criterion = nn.CrossEntropyLoss(reduction='mean')

        modified_images, labels = get_adversarial_images(model, test_dataset, criterion, epsilon)
        with torch.no_grad():
            outputs = model(modified_images)
            y_pred = torch.argmax(outputs, dim= 1)
            total = labels.size(0)
            correct = (y_pred == labels).sum().item()
        
        # Calculamos el accuracy.
        accuracy = correct / total
        if epsilon == 0.0:
                print(f'Accuracy original: {accuracy}')
        else:
            print(f'Accuracy {epsilon}: {accuracy}')