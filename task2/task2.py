import matplotlib.pyplot as plt
import numpy as np

import random
from math import pi, exp, ceil
import math

def gauss_dist_2d(x, y, sigma=1.0):
    sigma_2 = sigma * sigma
    return 1 / pi * sigma_2 * exp(-(x * x + y * y) / (2 * sigma_2))


def add_noise(img, rate=5):
    img[::rate, ::rate, :] = 1
    return


def get_kernel(window_size, x_mu=0, y_mu=0, sigma=1.0, normalize=True):
    kernel = np.ndarray((window_size, window_size))
    x_0 = window_size / 2 - 0.5 - x_mu
    y_0 = window_size / 2 - 0.5 - y_mu

    for i, row in enumerate(kernel):
        for j, _ in enumerate(row):
            kernel[i, j] = gauss_dist_2d(j - x_0, i - y_0, sigma)

    if normalize:
        kernel /= kernel.sum()

    return kernel


def filter(img, window_size=3, x_mu=0, y_mu=0, sigma=1.0):
    img2 = np.zeros_like(img)

    kernel = get_kernel(window_size, x_mu, y_mu, sigma, False)

    x_low, x_up = 0, img.shape[1] - 1
    y_low, y_up = 0, img.shape[1] - 1

    p = window_size // 2
    for k in range(img.shape[2]): # foreach color channel
        for i in range(0, img.shape[0]): # foreach row
            for j in range(0, img.shape[1]): # foreach column
                i1, i2 = max(i - p, y_low), min(i + p + 1, y_up)
                j1, j2 = max(j - p, x_low), min(j + p + 1, x_up)
                window = img[i1:i2, j1:j2, k]

                # чтобы избежать тёмных линий по краям, будем добавлять
                # нулевые элементы до нужной формы
                if j2 - j1 < window_size:
                    reshape_hor = np.zeros((i2 - i1, window_size - j2 + j1))
                    if j2 == x_up:
                        window = np.concatenate((window, reshape_hor), axis=1)
                    else:
                        window = np.concatenate((reshape_hor, window), axis=1)

                if i2 - i1 < window_size:
                    reshape_ver = np.zeros((window_size - i2 + i1, window_size))
                    if i2 == y_up:
                        window = np.concatenate((window, reshape_ver), axis=0)
                    else:
                        window = np.concatenate((reshape_ver, window), axis=0)

                # теперь нужно обработать ядро, чтобы добавленные элементы
                # не влияли на результат

                ceil_vec = np.vectorize(ceil)
                kernel1 = kernel * ceil_vec(window)
                kernel1 /= kernel1.sum()

                img2[i, j, k] = (kernel1 * window).sum()
    return img2


def main():
    window_size = 5
    x_mu = 0
    y_mu = 0
    sigma = 5

    img = plt.imread("img.png")[:, :, :3]
    add_noise(img)
    img2 = filter(img, window_size, x_mu, y_mu, sigma)

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(img)
    axs[1].imshow(img2)
    plt.show()


if __name__ == "__main__":
    main()