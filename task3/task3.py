import numpy as np
import math

def get_indices(N, n_batches, split_ratio):
    N -= 1
    batch_len = N * (split_ratio + 1) / (split_ratio * n_batches + 1) 
    step = (N - batch_len) / (n_batches - 1)
    inds = np.array([0, batch_len - step, batch_len])
    for _ in range(n_batches):
        yield np.array(np.round(inds), int)
        inds += step


def main():
    for inds in get_indices(100, 5, 0.25):
        print(inds)


if __name__ == "__main__":
    main()