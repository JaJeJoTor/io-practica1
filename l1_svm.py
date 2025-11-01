from random import randint
import numpy as np
import matplotlib.pyplot as plt
from Parte_1 import *

def generar_datos_svm(n_samples=100, random_state=42):
    """
    Esta función genera un conjunto de datos sintéticos en 2 dimensiones
    que pertenecen a dos categorías linealmente separables.
    """

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, random_state=random_state, cluster_std=2.5, center_box=(-10.0, 10.0))
    y = np.where(y == 0, -1, 1)  # Convertir etiquetas a -1 y 1
    return X, y

def generate_svm_matrix(X, y, C):
    """
    Esta función genera la matriz Símplex para resolver el problema de
    minimización del L1-SVM.
    """

    I, J = X.shape

    # Generamos una matriz inicial de ceros y rellenamos la primera fila
    simplex_matrix = np.zeros(shape = (I+1, 2*J + 2*I + 4))
    simplex_matrix[0][0] = 1
    simplex_matrix[0, 1:1 + 2*J] = 0.5
    simplex_matrix[0, 3+2*J:-1] = C  # Hiperparámetro de regularización C

    # Rellenamos cada fila, donde cada punto impone una restricción
    for i in range(1, simplex_matrix.shape[0]):
        i_data = i - 1

        for j in range(1, simplex_matrix.shape[1]):
            j = j-1
            if j < J:
                j_data = j
                simplex_matrix[i][j+1] = -(y[i_data] * X[i_data][j_data])  # Parámetros w+
            elif J <= j < 2*J:
                j_data = j - J
                simplex_matrix[i][j+1] = y[i_data] * X[i_data][j_data]  # Parámetros w-
            elif 2*J <= j < 2*J + 1:
                j_data = j - 2*J
                simplex_matrix[i][j+1] = -y[i_data]  # Parámetros b+
            elif 2*J + 1 <= j < 2*J + 2:
                j_data = j - (2*J + 1)
                simplex_matrix[i][j+1] = y[i_data]  # Parámetros b-

        simplex_matrix[i][2*J + 3 + i_data] = -1  # Parámetros ξ
        simplex_matrix[i][-1] = -1

    simplex_matrix[1:, 2 * J + I + 3:-1] = np.eye(I)  # Variables de holgura

    return simplex_matrix

def solve_simplex_svm(X, y, C, print_sol = True, plot=True, print_acc=True):
    """
    Esta función resuelve el problema de minimización del L1-SVM utilizando el
    método del Símplex.
    """

    def plot_result():
        """
        Muestra en un gráfico 2D la separación del conjunto de datos realizada
        por el L1-SVM.
        """

        # Dibuja los puntos de cada clase
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Clase 1')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Clase -1')

        # Rango de cada dimensión
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

        # Hiperplano separador del L1-SVM
        x = np.linspace(x1_min, x1_max, 200)
        y_decision = -w1 / w2 * x - b / w2

        # Dibuja el hiperplano separador
        plt.plot(x, y_decision, 'k-', label='Hiperplano separador')

        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Separación del L1-SVM")
        plt.legend(fontsize=8, loc='upper right')

        plt.show()
    
    def get_accuracy():
        """Calcula el accuracy obtenido por la separación del L1-SVM."""

        # Calcula la predicción
        pred = w1 * X[:, 0] + w2 * X[:, 1] + b
        y_pred = np.sign(pred)

        # Puntos sobre el hiperplano separador
        y_pred[y_pred == 0] == 1
        
        accuracy = np.sum(y == y_pred) / len(y)
        return accuracy

    # Generamos la matriz Símplex del conjunto de datos
    simplex_svm_matrix = generate_svm_matrix(X, y, C=C)
    engine = Simplex(simplex_svm_matrix)

    # Resolvemos con Símplex Combinado
    try:
        engine.combined_solver()
    except Exception as e:
        print("Error durante la resolución del problema:", e)

    # Obtenemos los parámetros del L1-SVM
    punto = engine.point

    w1 = punto[0] - punto[2]
    w2 = punto[1] - punto[3]
    b = punto[4] - punto[5]

    if print_sol:
        print("w1:", w1)
        print("w2:", w2)
        print("b:", b)

    if plot:
        plot_result()
    
    if print_acc:
        accuracy = get_accuracy()
        print(f'Accuracy = {accuracy}')

    del engine


