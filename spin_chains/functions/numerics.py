from numba import jit
import numpy as np


@jit(nopython=True, parallel=True)
def eigenvalue_numba(alpha, n, k):
    summands = np.array(
        [
            (2 * np.cos(2 * np.pi * j * k / n)) / (np.power(j, alpha))
            for j in range(1, n, 1)
        ]
    )
    lambda_k = np.sum(summands)
    return lambda_k


def eigenvalue(alpha, n, k):
    exp_positive = np.exp(1j * 2 * np.pi * k / n)
    exp_negative = np.exp(-1j * 2 * np.pi * k / n)
    lambda_k = float(
        mp.nstr(
            mp.re(
                mp.polylog(alpha, exp_positive)
                + mp.polylog(alpha, exp_negative)
                - mp.lerchphi(exp_positive, alpha, n)
                - mp.lerchphi(exp_negative, alpha, n)
            )
        )
    )
    return lambda_k


@jit(nopython=True, parallel=True)
def s_parameter(q, alpha, n):
    summands = np.array(
        [
            1
            / np.power(
                (eigenvalue_numba(alpha, n, n) - eigenvalue_numba(alpha, n, k)), q
            )
            for k in range(1, n, 1)
        ]
    )
    summation = np.sum(summands)
    return summation / n


@jit(nopython=True, parallel=False)
def delta_n(alpha, n):
    return s_parameter(1, alpha, n) / np.sqrt(n * s_parameter(2, alpha, n))


@jit(nopython=True, parallel=True)
def s_k_parameter(k, q, alpha, n):
    summands = np.array(
        [
            0
            if i in [k, n - k]
            else 1
            / np.power(
                (eigenvalue_numba(alpha, n, n) - eigenvalue_numba(alpha, n, k)), q
            )
            for i in range(1, n + 1, 1)
        ]
    )
    summation = np.sum(summands)
    return summation / n


@jit(nopython=True, parallel=False)
def delta_k(gamma, k, alpha, n):
    a = ((gamma * s_k_parameter(k, 2, alpha, n)) - np.power(gamma, 2)) / (
        2 * s_k_parameter(k, 2, alpha, n)
    )
    b = np.sqrt(
        np.power(a, 2)
        + ((2 * np.power(gamma, 2)) / (n * s_k_parameter(k, 2, alpha, n)))
    )
    return a - b


@jit(nopython=True, parallel=True)
def first_order_fidelity_correction(t, gamma, alpha, n):
    d_n = delta_n(alpha, n)
    summands = np.array(
        [
            np.cos(
                t
                * (
                    (gamma * eigenvalue_numba(alpha, n, n))
                    - (gamma * eigenvalue_numba(alpha, n, j))
                    - delta_k(gamma, j, alpha, n)
                )
            )
            * np.cos(t * d_n)
            # * (np.power(n * delta_k(gamma, j, alpha, n) * delta_n(alpha, n), 2) / 2)
            * (
                (np.power(n * gamma * delta_k(gamma, j, alpha, n) * d_n, 2))
                / (
                    (2 * np.power(gamma, 2))
                    + (
                        n
                        * s_k_parameter(j, 2, alpha, n)
                        * np.power(delta_k(gamma, j, alpha, n), 2)
                    )
                )
                / 2
            )
            for j in range(n // 2 + 1, n, 1)
        ]
    )
    summation = np.sum(summands)
    return summation


@jit(nopython=True, parallel=True)
def qst_fidelity(t, gamma, alpha, n):
    d_n = delta_n(alpha, n)
    summands = np.array(
        [
            np.cos(
                t
                * (
                    (gamma * eigenvalue_numba(alpha, n, n))
                    - (gamma * eigenvalue_numba(alpha, n, j))
                    - delta_k(gamma, j, alpha, n)
                )
            )
            * np.cos(t * d_n)
            # * (np.power(n * delta_k(gamma, j, alpha, n) * delta_n(alpha, n), 2) / 2)
            * (
                (np.power(n * gamma * delta_k(gamma, j, alpha, n) * d_n, 2))
                / (
                    (2 * np.power(gamma, 2))
                    + (
                        n
                        * s_k_parameter(j, 2, alpha, n)
                        * np.power(delta_k(gamma, j, alpha, n), 2)
                    )
                )
                / 2
            )
            for j in range(n // 2 + 1, n, 1)
        ]
    )
    summation = np.sum(summands)
    return summation
