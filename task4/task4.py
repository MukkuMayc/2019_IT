import matplotlib.pyplot as plt
import numpy as np
from input_data import n, m, scaler, X


def is_pareto(idx, X):
    for i, vec in enumerate(X):
        if i != idx and not np.any(X[idx, :] > vec):
            return False
    return True


def split_indices_by_pareto(X):
    pareto = []
    not_pareto = []
    for idx in range(X.shape[0]):
        if is_pareto(idx, X):
            pareto.append(idx)
        else:
            not_pareto.append(idx)
    return pareto, not_pareto


if __name__ == "__main__":
    indices_pareto, indices_not_pareto = split_indices_by_pareto(X)

    # чтобы на графе получился замкнутая линия, добавляем ещё столбец
    X = np.append(X, np.reshape(X[:, 0], (n, 1)), axis=1)

    vecs_pareto, vecs_not_pareto = X[indices_pareto, :], X[indices_not_pareto, :]
    theta = 2 * np.pi * np.arange(m + 1, dtype=int) / m

    _, axes = plt.subplots(ncols=3, subplot_kw=dict(polar=True))

    for ax in axes:
        ax.set_rmax(scaler)
        ax.set_thetagrids(np.arange(0, 360, 360 / m), labels=(np.arange(m) + 1))
        ax.grid(True)

    axes[0].set_title('Pareto vectors')
    for vec in vecs_pareto:
        axes[0].plot(theta, vec)

    axes[1].set_title('Not Pareto vectors')
    for vec in vecs_not_pareto:
        axes[1].plot(theta, vec)

    axes[2].set_title('All')
    for vec in X:
        axes[2].plot(theta, vec)

    plt.show()