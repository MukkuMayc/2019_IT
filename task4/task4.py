import matplotlib.pyplot as plt
import numpy as np
from input_data import n, m, X

# Задание 4
# Входные данные: X, shape=(N, M). N - число точек, M - число координат
# 1. Генератор точек Парето (P) фронта
# 2. Нарисовать X, P в 'полярных' координатах (M осей)
# Общие замечание: все по возможности векторизовать в numpy до 2D матриц

# Вектор x' называется оптимальным по Парето, если не существует x такого, что
# f_i(x') <= f_i(x) для всех i = 1, ..., k, и f_i(x) < f_i(x') хотя бы для
# одного i. 

# 1. Берём X, 
# 2. Проходим по каждому столбцу, проверяем по Парето
# Как проверять по Парето:
# 1) берём столбец из X, пока не закончатся, когда пройдём по всем, истина
# 2) если хотя бы одна координата исходного вектора > другого, 
# то первый шаг, иначе ложь
# 3. если вектор удовлетворяет Парето, возвращаем его


def is_pareto(idx, X):
    for i, col in enumerate(X.T):
        if i != idx:
            if not np.any(X[:,idx] > col):
                return False
    return True

def get_pareto_indices(X):
    for idx in range(X.shape[1]):
        if is_pareto(idx, X):
            yield idx

def get_pareto_vectors(X):
    for idx, col in enumerate(X.T):
        if is_pareto(idx, X):
            yield col

def get_not_pareto_vectors(X):
    for idx, col in enumerate(X.T):
        if not is_pareto(idx, X):
            yield col


if __name__ == "__main__":
    vec_pareto = get_pareto_vectors(X)
    vec_nopareto = get_not_pareto_vectors(X)

    fig, axes = plt.subplots(ncols=2, subplot_kw=dict(polar=True))

    theta = 2 * np.pi * np.arange(0, 1 + 1 / m, 1 / m)

    for vec in vec_pareto:
        axes[0].plot(theta, np.append(vec, vec[0]))

    for vec in vec_nopareto:
        axes[1].plot(theta, np.append(vec, vec[0]))

    plt.show()