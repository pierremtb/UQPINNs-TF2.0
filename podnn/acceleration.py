"""Compiled and parallelized functions."""

import warnings
import numpy as np
from numba import jit, prange


# Disable bad division warning when summing up squares
warnings.filterwarnings("ignore", category=RuntimeWarning)


@jit(nopython=True, parallel=True)
def loop_vdot(n_s, U_tot, U_tot_sq, V, v_pred_hifi):
    """Return mean, std from parallelized dot product between V an v"""
    # pylint: disable=not-an-iterable
    for i in prange(n_s):
        # Computing one snapshot
        U = V.dot(v_pred_hifi[i])
        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2
    return U_tot, U_tot_sq


@jit(nopython=True, parallel=True)
def loop_vdot_t(n_s, n_t, U_tot, U_tot_sq, V, v_pred_hifi):
    """Return mean, std from parallelized dot product between V an v (w/ t)."""
    # pylint: disable=not-an-iterable
    v_pred_hifi = np.ascontiguousarray(v_pred_hifi)
    for i in prange(n_s):
        # Computing one snapshot
        s = n_t * i
        e = n_t * (i + 1)
        v_pred_hifi_i = np.ascontiguousarray(v_pred_hifi[s:e].T)
        U = V.dot(v_pred_hifi_i)
        # Building the sum and the sum of squaes
        U_tot += U
        U_tot_sq += U**2
    return U_tot, U_tot_sq


@jit(nopython=True, parallel=True)
def loop_u(u, X_U, U, U_no_noise, x, mu_lhs, u_noise=0., x_noise=0.):
    """Return the inputs/snapshots matrices from parallel computation."""
    # pylint: disable=not-an-iterable

    n_xyz = x.shape[0]
    n_p = mu_lhs.shape[0]

    for i in prange(n_p):
        mu_lhs_i = mu_lhs[i]
        if x_noise > 0.:
            mu_lhs_i += np.random.normal(0., x_noise*np.std(mu_lhs_i), mu_lhs_i.shape[0])
        for j in prange(n_xyz):
            idx = i * n_p + j
            U_ij_no_noise = u(x[j], 0, mu_lhs[i])
            X_U[idx, :] = np.hstack((mu_lhs_i, x[j]))
            U_ij = u(x[j], 0, mu_lhs_i)
            U[idx, :] = U_ij
            U_no_noise[idx, :] = U_ij_no_noise
        # if u_noise > 0.:
        #     U[i] += u_noise*np.std(U[i])*np.random.randn(U[i].shape[0], U[i].shape[1])
    return X_U, U, U_no_noise


@jit(nopython=True, parallel=True)
def loop_u_t(u, n_t, n_v, n_xyz, n_h,
             X_v, U, U_struct, X, mu_lhs, t_min, t_max, u_noise=0.):
    """Return the inputs/snapshots matrices from parallel computation (w/ t)."""
    # Creating the time steps
    t = np.linspace(t_min, t_max, n_t)
    tT = t.reshape((n_t, 1))
    # pylint: disable=not-an-iterable
    for i in prange(mu_lhs.shape[0]):
        # Getting the snapshot times indices
        s = n_t * i
        e = n_t * (i + 1)

        # Setting the regression inputs (t, mu)
        X_v[s:e, :] = np.hstack((tT, np.ones_like(tT)*mu_lhs[i]))

        # Calling the analytical solution function
        Ui = np.zeros((n_v, n_xyz, n_t))
        for j in range(n_t):
            Uij = u(X, t[j], mu_lhs[i])
            if u_noise > 0.:
                Uij += u_noise*np.std(Uij)*np.random.randn(Uij.shape[0], Uij.shape[1])
            Ui[:, :, j] = Uij

        U[:, s:e] = Ui.reshape((n_h, n_t))
        U_struct[:, :, i] = U[:, s:e]
    return X_v, U, U_struct


@jit(nopython=True, parallel=True)
def lhs(n, samples):
    """Borrowed, parallelized __lhscentered() from pyDOE."""

    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    u = np.random.rand(samples, n)
    a = cut[:samples]
    b = cut[1:samples + 1]
    rdpoints = np.zeros(u.shape)
    for j in prange(n):
        rdpoints[:, j] = u[:, j]*(b-a) + a

    # Make the random pairings
    H = np.zeros(rdpoints.shape)
    for j in prange(n):
        order = np.random.permutation(np.arange(samples))
        H[:, j] = rdpoints[order, j]

    return H
