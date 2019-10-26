# Входные данные: X, shape=(N, M). N - число точек, M - число координат
import numpy as np

n = 10
m = 3
scaler = 20

X = np.trunc(np.random.rand(n, m) * scaler)