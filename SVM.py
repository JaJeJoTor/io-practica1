from sklearn.svm import SVC
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV

# Transformación de imagen a tensor
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

# Datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
X_test, y_test = test_dataset.data.numpy(), test_dataset.targets.numpy()

X_train, X_test = X_train.reshape(len(X_train), -1), X_train.reshape(len(X_test), -1)

params_grid = {'C': [0.1, 0.5, 1], 'gamma' : [0.001, 0.01, 0.1, 1]}
clf = SVC(random_state= 42)
grid = GridSearchCV(clf, params_grid, scoring= 'accuracy', cv= 4, verbose= 1, n_jobs= -1)
grid.fit(X_train, y_train)
print(grid.best_params_)


