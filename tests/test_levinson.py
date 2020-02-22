import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from bsseval.utils import block_levinson


def block_sym_toeplitz(L):

    assert L.ndim == 3  # 3D tensor
    assert L.shape[1] == L.shape[2]  # square matrix
    assert np.allclose(L[0], L[0].T)  # 1st block has to be symmetric

    n_blocks, n_dim, _ = L.shape

    T = np.zeros((n_blocks * n_dim, n_blocks * n_dim), dtype=L.dtype)

    for b in range(n_blocks):
        for d in range(n_blocks - b):
            T[(b + d) * n_dim : (b + d + 1) * n_dim, d * n_dim : (d + 1) * n_dim] = L[b]
            if b > 0:
                T[
                    d * n_dim : (d + 1) * n_dim, (b + d) * n_dim : (b + d + 1) * n_dim
                ] = L[b].T

    return T


def toeplitz_blocks(L):

    n_blocks, n_dim, _ = L.shape

    T = np.zeros((n_blocks * n_dim, n_blocks * n_dim), dtype=L.dtype)

    for r in range(n_dim):
        for c in range(n_dim):
            T[
                r * n_blocks : (r + 1) * n_blocks, c * n_blocks : (c + 1) * n_blocks
            ] = toeplitz(L[:, r, c], L[:, c, r])

    return T


def block_permutation(T, n_blocks, n_dim):

    assert T.shape[0] == T.shape[1]
    assert T.shape[0] == n_blocks * n_dim

    return T.reshape((n_blocks, n_dim, n_blocks, n_dim)).reshape(*T.shape, order="F")


def direct(L, y):
    return np.linalg.solve(block_sym_toeplitz(L), y)


def chrono(L, y, func, n_repeat=100):

    t = time.perf_counter()
    for n in range(n_repeat):
        func(L, y)
    t = time.perf_counter() - t

    return t / n_repeat


n_blocks = 512
n_dim = 4

L = np.random.randn(n_blocks, n_dim, n_dim)
L[0] = L[0] + L[0].T  # make symmetric
y = np.random.randn(n_blocks * n_dim, 10)

# outer structure Toeplitz, inner structure arbitrary
T = block_sym_toeplitz(L)
# outer structure arbitrary, inner structure toeplitz
T2 = toeplitz_blocks(L)

assert np.allclose(T, T.T), "Error: matrix is not symmetric"
assert np.allclose(
    T, block_permutation(T2, n_dim, n_blocks)
), "Error: structures do not commute"
assert np.allclose(
    block_permutation(T, n_blocks, n_dim), T2
), "Error: structures do not commute in reverse"
assert np.allclose(
    T, block_permutation(block_permutation(T, n_blocks, n_dim), n_dim, n_blocks)
), "Error: not invertible structure"

plt.matshow(T)
plt.show()

x1 = np.linalg.solve(T, y)
x2 = block_levinson(L, y)

print("Error:", np.max(np.abs(x1 - x2)))

n_repeat = 3
print(
    "time direct (without matrix creation):",
    chrono(T, y, np.linalg.solve, n_repeat=n_repeat),
)
print(
    "time direct (including matrix creation):", chrono(L, y, direct, n_repeat=n_repeat)
)
print("time levinson2:", chrono(L, y, block_levinson, n_repeat=n_repeat))
