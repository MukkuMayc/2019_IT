# Входные данные: X, shape=(N, M). N - число точек, M - число координат
import numpy as np

n = 1000
m = 10
scaler = 20

X = np.trunc(np.random.rand(n, m) * scaler)