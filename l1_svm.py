from random import randint
import numpy as np
import matplotlib.pyplot as plt
from Parte_1 import *


def generar_datos_svm(n_samples=100, random_state=42):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2, random_state=random_state, cluster_std=2.5, center_box=(-10.0, 10.0))
    y = np.where(y == 0, -1, 1)  # Convertir etiquetas a -1 y 1
    return X, y

def generate_svm_matrix(X, y, C):

    I, J = X.shape

    simplex_matrix = np.zeros(shape = (I+1, 2*J + 2*I + 4))
    simplex_matrix[0][0] = 1  # Coeficiente de la función objetivo
    simplex_matrix[0, 1:1 + 2*J] = 0.5
    simplex_matrix[0, 3+2*J:-1] = C



    for i in range(1, simplex_matrix.shape[0]):

        i_data = i - 1

        for j in range(1, simplex_matrix.shape[1]):
            j = j-1
            if j < J:
                j_data = j
                simplex_matrix[i][j+1] = -(y[i_data] * X[i_data][j_data])
            elif J <= j < 2*J:
                j_data = j - J
                simplex_matrix[i][j+1] = y[i_data] * X[i_data][j_data]
            elif 2*J <= j < 2*J + 1:
                j_data = j - 2*J
                simplex_matrix[i][j+1] = -y[i_data]
            elif 2*J + 1 <= j < 2*J + 2:
                j_data = j - (2*J + 1)
                simplex_matrix[i][j+1] = +y[i_data]

        simplex_matrix[i][2*J + 3 + i_data] = -1
        simplex_matrix[i][-1] = -1

    simplex_matrix[1:, 2 * J + I + 3:-1] = np.eye(I)

    return simplex_matrix

def solve_simplex_svm(X, y, C, print_sol = True, plot=True):

    def plot_result():
        plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Clase 1')
        plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Clase -1')

        x = np.linspace(np.min(X) - 1, np.max(X) + 1, 100)
        y_decision_boundary = -w1 / w2 * x - b / w2

        plt.plot(x, y_decision_boundary, 'k-', label='Frontera de decisión')

        plt.xlim(-10, 10)
        plt.ylim(-10, 10)

        plt.show()

    simplex_svm_matrix = generate_svm_matrix(X, y, C=C)
    engine = Simplex(simplex_svm_matrix)

    try:
        engine.combined_solver()
    except Exception as e:
        print("Error durante la resolución del problema:", e)

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

    del engine


