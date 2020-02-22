import numpy as np
from scipy.optimize import linear_sum_assignment


def pad_or_truncate(audio_reference, audio_estimates):
    """Pad or truncate estimates by duration of references:
    - If reference > estimates: add zeros at the and of the estimated signal
    - If estimates > references: truncate estimates to duration of references

    Parameters
    ----------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    Returns
    -------
    references : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing true reference sources
    estimates : np.ndarray, shape=(nsrc, nsampl, nchan)
        array containing estimated sources
    """
    est_shape = audio_estimates.shape
    ref_shape = audio_reference.shape
    if est_shape[1] != ref_shape[1]:
        if est_shape[1] >= ref_shape[1]:
            audio_estimates = audio_estimates[:, : ref_shape[1], :]
        else:
            # pad end with zeros
            audio_estimates = np.pad(
                audio_estimates,
                [(0, 0), (0, ref_shape[1] - est_shape[1]), (0, 0)],
                mode="constant",
            )

    return audio_reference, audio_estimates


def linear_sum_assignment_with_inf(cost_matrix):
    """
    Solves the permutation problem efficiently via the linear sum
    assignment problem.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

    This implementation was proposed by @louisabraham in
    https://github.com/scipy/scipy/issues/6900
    to handle infinite entries in the cost matrix.
    """
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        cost_matrix = cost_matrix.copy()
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder

    return linear_sum_assignment(cost_matrix)


def block_levinson(L, y):
    """
    Parameters
    ----------
    L: ndarray (n_blocks * n_dim, n_dim)
        The first block-column of the matrix
    y: ndarray (n_blocks * n_dim, n_rhs)
        One or multiple right hand sides

    Returns
    -------
    The result of solving BlockToeplitz(L) x = y
    """

    assert L.ndim == 3

    n_blocks, n_dim, _ = L.shape

    assert L.shape[1] == L.shape[2]
    assert y.shape[0] == n_blocks * n_dim

    L = L.reshape((n_blocks * n_dim, n_dim))

    # Get the bottom block row of the matrix
    B = np.zeros((n_dim, n_blocks * n_dim), dtype=L.dtype)
    for b in range(0, n_blocks):
        B[:, b * n_dim : (b + 1) * n_dim] = L[
            (n_blocks - b - 1) * n_dim : (n_blocks - b) * n_dim, :
        ]

    f = np.zeros((n_blocks * n_dim, n_dim), dtype=L.dtype)
    b = np.zeros((n_blocks * n_dim, n_dim), dtype=L.dtype)
    x = np.zeros(n_blocks * n_dim, dtype=L.dtype)
    x = np.zeros_like(y)

    f[:n_dim, :] = np.linalg.inv(L[:n_dim, :])
    b[-n_dim:, :] = f[:n_dim, :]
    x[:n_dim] = f[:n_dim] @ y[:n_dim]

    for n in range(1, n_blocks):
        f_s = f[:n_dim * (n + 1), :]
        b_s = b[-n_dim * (n + 1):, :]
        x_s = x[:n_dim * (n + 1)]

        ef = B[:, -(n + 1) * n_dim :] @ f_s
        eb = L[: (n + 1) * n_dim, :].T @ b_s
        ex = B[:, -(n + 1) * n_dim :] @ x_s

        A = np.linalg.inv(
            np.vstack((np.hstack((np.eye(n_dim), eb)), np.hstack((ef, np.eye(n_dim)))))
        )
        fb = np.column_stack((f_s, b_s))
        f_s[:, :], b_s[:, :] = fb @ A[:, :n_dim], fb @ A[:, n_dim:]

        x_s[:] += b_s @ (y[n * n_dim : (n + 1) * n_dim] - ex)

    return x
