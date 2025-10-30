import numpy as np
from numpy.linalg import inv
from copy import deepcopy
import matplotlib.pyplot as plt

WITH = 60

class Simplex:
    def __init__(self, matrix):
        self.original_matrix = deepcopy(matrix)
        self.current_matrix = deepcopy(matrix)

        self.n, self.m = matrix.shape
        self.dimension_punto = self.m - self.n - 1
        self.n_restricciones = self.n - 1

        self.base = [self.m - self.n - 1 + i for i in range(self.n_restricciones)]

        self.last_solution_history = None
        self.history = []
        self.point = [0] * self.dimension_punto

    @property
    def c_submatrix(self):
        return self.current_matrix[0,1:-1]

    @property
    def A_submatrix(self):
        return self.current_matrix[1:,1:-1]

    @property
    def b_submatrix(self):
        return self.current_matrix[1:, -1]

    @property
    def objective_value(self):
        return self.current_matrix[0, -1]

    def __simplex_iterate(self):
        """Esta funion espera que hayas actualizado la base siguiendo los criterios de una de las versiones del algoritmo del simplex."""

        def get_base_change_matrix():
            base_change_matrix = np.zeros(shape=(self.n, self.n))
            base_change_matrix[0, 0] = 1

            B = self.A_submatrix[:, self.base]
            B_inv = inv(B)

            base_change_matrix[0, 1:] = -self.c_submatrix[self.base].T @ B_inv
            base_change_matrix[1:, 1:] = B_inv

            return base_change_matrix

        base_change_matrix = get_base_change_matrix()
        new_matrix = base_change_matrix @ self.current_matrix
        return new_matrix

    def iterate(self, type, return_extra_info=False):

        def update_base_primal():
            B_in = int(np.argmin(self.c_submatrix))
            assert self.c_submatrix[B_in] < 0, "No compatible con simplex"

            cocients = self.b_submatrix / self.A_submatrix[:, B_in]
            cocients[cocients < 0] = np.nan

            # elegir la fila de salida (índice de fila)
            row_out = int(np.nanargmin(cocients))
            # reemplazar la variable básica en esa fila por la entrante
            self.base[row_out] = B_in

            self.base = sorted(self.base)

        def update_base_dual():
            # elegir la fila con b mínimo
            row_out = np.argmin(self.b_submatrix)
            B_out = self.base[row_out]

            total_vars = self.n_restricciones + self.dimension_punto
            nonbase = np.delete(np.arange(total_vars), self.base)

            numerators = np.delete(self.c_submatrix, self.base)
            denominators = np.delete(self.A_submatrix[row_out, :], self.base)

            mask = denominators < 0
            if not np.any(mask):
                raise AssertionError("No hay entrada posible en simplex dual (denominadores no negativos)")

            cocients = np.abs(numerators[mask] / denominators[mask])
            arg_cociente_minimo = int(np.argmin(cocients))
            B_in = int(nonbase[mask][arg_cociente_minimo])

            # sustituir en la fila correspondiente
            self.base[row_out] = B_in

            self.base = sorted(self.base)

        if type == "primal":
            update_base_primal()
        elif type == "dual":
            update_base_dual()
        else:
            raise ValueError("Tipo de iteracion no reconocido, use 'primal' o 'dual'")

        self.current_matrix = self.__simplex_iterate()

        base = self.base.copy()
        arguments = self.b_submatrix.copy()
        objective_value = self.objective_value

        if return_extra_info:
            point = [0] * self.dimension_punto
            # asignar valores del b_submatrix usando correspondencia fila -> variable básica
            for row_idx, basic in enumerate(self.base):
                if basic < self.dimension_punto:
                    point[basic] = float(self.b_submatrix[row_idx])
            self.point = point
            return base, arguments, point, objective_value
        else:
            return None



    def solver(self, type, printear_intermedios = True, return_history = False):


        def append_history():
            self.history.append({
                "iteration" : len(self.history),
                "base": np.array(self.base, dtype=int),
                "arguments": np.array(self.b_submatrix, dtype=float),
                "point": np.array(point, dtype=float),
                "objective_value": np.array(self.objective_value, dtype=float)
            })
        def is_feasible():

            feasible = True

            for i in range(self.n_restricciones):
                if not (np.transpose(point) @ self.A_submatrix[i, :self.dimension_punto] <= self.b_submatrix[i]):
                    feasible = False

            return feasible
        def printear_paso():
            print("-"*WITH)
            print("\n- Iteracion", len(self.history)-1)
            print(f"- Base actual:", np.array(base, dtype=int))
            print("- Argumentos actuales:", np.array(arguments, dtype=float))
            print("- Punto actual:", np.array(point, dtype=float))
            print("- Valor objetivo actual:", np.array(objective_value, dtype=float))
            print("\n - Matriz actual:")
            print(np.round(self.current_matrix, 2))

            print("\n - Matriz original:")
            print(np.round(self.original_matrix))
            print("-"*WITH)

        def primal_stop_condition():
            return all(self.c_submatrix >= 0)

        def dual_stop_condition():
            return all(self.b_submatrix >= 0)

        if type == "primal":
            condition = primal_stop_condition
        elif type == "dual":
            condition = dual_stop_condition
        else:
            raise ValueError("Tipo de solver no reconocido, use 'primal' o 'dual'")



        point = self.point

        if type == "primal" and not is_feasible():
            return None


        append_history()

        if printear_intermedios:
            base = self.base.copy()
            arguments = self.b_submatrix.copy()
            objective_value = self.objective_value

            printear_paso()


        while not condition():

            if type == "primal":
                base, arguments, point, objective_value = self.iterate(type = "primal", return_extra_info = True)
            else:
                base, arguments, point, objective_value = self.iterate(type = "dual", return_extra_info = True)

            append_history()

            if printear_intermedios:
                printear_paso()


        if return_history:
            return self.history
        else:
            return None

    # language: python
    def combined_solver(self):

        def is_feasible():
            point = [0] * self.dimension_punto
            feasible = True
            for i in range(self.n_restricciones):
                if not (np.transpose(point) @ self.A_submatrix[i, :self.dimension_punto] <= self.b_submatrix[i]):
                    feasible = False
            return feasible

        if is_feasible():
            print("ITERANDO CON EL PRIMAL".center(WITH, "-"))
            self.solver(type="primal", printear_intermedios=True)
        else:
            # invertir solamente la fila de coeficientes actual y restaurarla luego
            self.original_matrix[0, 1:] = -self.original_matrix[0, 1:]
            self.current_matrix[0, 1:] = -self.current_matrix[0, 1:]
            print("ITERANDO CON EL DUAL".center(WITH, "-"))
            self.solver(type="dual", printear_intermedios=True)
            # restaurar original
            self.original_matrix[0, 1:] = -self.original_matrix[0, 1:]
            self.current_matrix[0, 1:] = -self.current_matrix[0, 1:]
            print("ITERANDO CON EL PRIMAL".center(WITH, "-"))
            self.solver(type="primal", printear_intermedios=True)

    def transform_to_dual(self):
        dual_matrix = np.zeros(shape=(self.dimension_punto + 1, (self.n_restricciones) * 2 + 2))
        dual_matrix[0, 0] = 1
        dual_matrix[0, 1:self.dimension_punto + 1] = -self.original_matrix[1:, -1].T # b del primal a c del dual
        dual_matrix[1:, -1] = self.original_matrix[0, 1:self.dimension_punto + 1].T # c del primal a b del dual (El cambio de signo ya se hace al definir la matriz primal)
        dual_matrix[1:, 1:self.dimension_punto + 1] = -self.original_matrix[1:, 1:self.dimension_punto + 1].T # A del primal a A del dual (le metemos un - para pasar de >= a <= directamente)
        dual_matrix[1:, self.n_restricciones + 1:-1]= np.identity(self.n_restricciones)  # variables de holgura

        self.primal_original_matrix = deepcopy(self.original_matrix)
        self.__init__(dual_matrix)


    def get_primal_solution_from_dual_solution(self, printear = True):
        dual_base = self.base
        dual_non_base = np.delete(np.arange(self.n_restricciones + self.dimension_punto), dual_base)

        def mapping(var_index):
            # las variables duales son holgura en el primal
            if var_index < self.dimension_punto:
                return var_index - self.dimension_punto
            else:
                return var_index - self.dimension_punto

        primal_base = list(map(mapping, dual_non_base))

        aux_engine = Simplex(self.primal_original_matrix)
        aux_engine.base = primal_base
        base, _, point, _ = aux_engine.iterate(type = "primal", return_extra_info=True)

        if printear:
            print("-"*WITH)
            print("Solución del problema primal obtenida desde la solución del dual:")
            print("Base primal:", np.array(primal_base, dtype=int))
            print("Punto primal:", np.array(point, dtype=float))
            # Se puede obterner del primal, pero lo obtenemos del dual puesto que es un resultado del teorema de dualidad fuerte
            objective_value = self.objective_value
            print("Valor objetivo:", np.array(objective_value, dtype=float))
            print("-"*WITH)







    def plot_sol_history(self, size = 10):

        x1_max = size
        x2_max = size


        x1 = np.linspace(0, x1_max, 500)
        x2 = np.linspace(0, x2_max, 500)

        feasible_region = np.ones(shape=(len(x1), len(x2)), dtype=bool)

        for r in range(self.A_submatrix.shape[0]):
            restriccion = self.original_matrix[r + 1, 1:self.dimension_punto + 1]
            t_ind = self.original_matrix[r + 1, -1]

            for i in range(len(x2)):
                for j in range(len(x1)):
                    if feasible_region[i][j]:
                        eval_condition = restriccion[0] * x1[j] + restriccion[1] * x2[i] <= t_ind
                        feasible_region[i][j] = eval_condition

        plt.imshow(feasible_region, cmap="Spectral_r", origin="lower", extent=[0, x2_max, 0, x1_max], )


        points_list = [dict["point"] for dict in self.history]
        filtered_point_list = []

        for point in points_list:
            if not filtered_point_list:
                filtered_point_list.append(point)
            else:
                if any(filtered_point_list[-1] != point):
                    filtered_point_list.append(point)

        points_list = np.array(filtered_point_list)

        x_points = points_list[:, 0]
        y_points = points_list[:, 1]

        for i in range(len(points_list) - 1):
            plt.arrow(
                x_points[i], y_points[i],  # Inicio de la flecha
                x_points[i + 1] - x_points[i],  # Desplazamiento en x
                y_points[i + 1] - y_points[i],  # Desplazamiento en y
                head_width=0.04 * size, head_length=0.03 * size, length_includes_head=True, color='orange'
            )

            ax = plt.gca()

            ax.text(
                x_points[i] + 0.5 * (x_points[i + 1] - x_points[i]), y_points[i] + 0.5 * (y_points[i + 1] - y_points[i]),
                str(i + 1),
                color='white',
                fontsize=12,
                ha="center",
                va="center",
            )


        plt.scatter(x_points, y_points,c = 'blue')

        gradient_vector = -self.original_matrix[0,1:self.dimension_punto+1]
        norm = np.linalg.norm(gradient_vector)
        unit_vector = gradient_vector / norm
        scale = min(x2_max, x1_max)
        scaled_vector = unit_vector * scale

        from matplotlib.patches import FancyArrowPatch

        arrow = FancyArrowPatch(
            (0, 0),  # Inicio de la flecha
            (scaled_vector[0], scaled_vector[1]),  # Fin de la flecha
            arrowstyle="->",  # Estilo de flecha
            color="white",  # Color de la flecha
            linewidth=10,  # Ancho de la línea
            alpha=0.1,  # Transparencia
            mutation_scale=100,  # Escala de la cabeza de la flecha
        )

        ax = plt.gca()

        ax.text(
            scaled_vector[0] * 0.85, scaled_vector[1] * 0.85,
            "Gradiente",
            fontsize=12,
            ha="center",
            va="center",
        )

        ax.add_patch(arrow)

        # plotear recta tangente al punto óptimo
        optimal_point = points_list[-1]
        vector = (-gradient_vector[1], gradient_vector[0])

        t = np.linspace(-size, size, 500)
        x_tangent = optimal_point[0] + vector[0] * t
        y_tangent = optimal_point[1] + vector[1] * t

        plt.plot(x_tangent, y_tangent, color='green', linestyle='--', label='Recta tangente en el punto óptimo')

        plt.xlim(0, x2_max)
        plt.ylim(0, x1_max)

        plt.show()


# --------------------------

from Parte_1 import *
import numpy as np

def generar_datos_svm(n_samples=10, n_features=2, random_state=42):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=n_features, random_state=random_state)
    y = np.where(y == 0, -1, 1)  # Convertir etiquetas a -1 y 1
    return X, y

C = 1

X, y = generar_datos_svm(n_samples=2, n_features=2)

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

engine = Simplex(simplex_matrix)
engine.combined_solver()


plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Clase 1')
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red', label='Clase -1')
plt.show()

punto = engine.point

w1 = punto[0] - punto[2]
w2 = punto[1] - punto[3]
b = punto[4] - punto[5]

x = np.linspace(np.min(X)-1, np.max(X)+1, 100)
y_decision_boundary = (w1* x + b)






plt.plot(x, y_decision_boundary, 'k-', label='Frontera de decisión')