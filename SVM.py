from sklearn.svm import SVC
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score
import joblib


# Transformación de imagen a tensor
transform = transforms.Compose([transforms.Normalize((0.5,), (0.5,))])

print('Cargando datos...')
# Datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train, y_train = train_dataset.data.numpy(), train_dataset.targets.numpy()
X_test, y_test = test_dataset.data.numpy(), test_dataset.targets.numpy()

X_train, X_test = X_train.reshape(len(X_train), -1) / 255, X_test.reshape(len(X_test), -1) / 255

# Hemos buscado en internet los mejores parámetros para este problema
clf = SVC(kernel='rbf', gamma= 0.05, C= 5, random_state= 42, max_iter= 1000)

print('Entrenando...')
# Entrenamos y predecimos para test
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Printeamos métricas
print(f'Accuracy: {accuracy_score(y_test, y_pred)} || F1: {f1_score(y_test, y_pred, average= "weighted")}')

# Guardamos el modelo
joblib.dump(clf, 'SVC.pkl')

