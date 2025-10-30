from Parte_1 import *
import numpy as np

def generar_datos_svm(n_samples=10, n_features=2, random_state=42):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=n_features, random_state=random_state)
    y = np.where(y == 0, -1, 1)  # Convertir etiquetas a -1 y 1
    return X, y

C = 1

X, y = generar_datos_svm(n_samples=2, n_features=1)

I, J = X.shape

simplex_matrix = np.zeros(shape = (I+1, 2*J + 2*I + 4))
simplex_matrix[0][0] = 1  # Coeficiente de la función objetivo
simplex_matrix[0, 1:1 + 2*J] = -0.5
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

engine = Simplex(simplex_matrix)
engine.solver(type="primal", printear_intermedios=True)