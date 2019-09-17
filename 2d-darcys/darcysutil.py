import numpy as np
import tensorflow as tf
import time
from datetime import datetime
from pyDOE import lhs
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("utils")
from plotting import newfig, savefig, saveResultDir


# Exact relation between K and u
def k_vanGenuchten(u):
    alpha = 0.1
    n = 1.885
    m = 1.0 - 1.0/n
    s = (1.0 + (alpha*np.abs(u))**n)**(-m)
    K = np.sqrt(s)*(1.0 - (1.0 - s**(1.0/m))**m)**2
    return K

def scarcify(X, u, N):
    idx = np.random.choice(X.shape[0], N, replace=False)
    return X[idx, :], u[idx, :]

def prep_data(path, N_u, N_b, N_f, L_1, L_2, noise=0.0):

    # Reading external data [t is 100x1, usol is 256x100, x is 256x1]
    data = np.load(path)

    # X is x 1, k is 10000 x 1, u is 10000 x 1
    X = data["X"]
    k = data["k"]
    u = data["u"]

    # Boundary points
    x1_b1 = np.zeros(N_b)[:, None]
    x2_b1 = L_2 * np.random.random(N_b)[:, None]
    X_b1 = np.hstack((x1_b1, x2_b1))
    x1_b2 = L_1 * np.random.random(N_b)[:, None]
    x2_b2 = np.zeros(N_b)[:, None]
    X_b2 = np.hstack((x1_b2, x2_b2))
    x1_b3 = L_1 * np.ones(N_b)[:, None]
    x2_b3 = L_2 * np.random.random(N_b)[:, None]
    X_b3 = np.hstack((x1_b3, x2_b3))   
    x1_b4 = L_1 * np.random.random(N_b)[:, None]
    x2_b4 = L_2 * np.ones(N_b)[:, None]
    X_b4 = np.hstack((x1_b4, x2_b4))
    X_b = np.hstack((X_b1, X_b2))
    X_b = np.hstack((X_b, X_b3))
    X_b = np.hstack((X_b, X_b4))

    # Collocation points
    X1_f = L_1 * np.random.random(N_f)[:, None]
    X2_f = L_2 * np.random.random(N_f)[:, None]
    X_f = np.hstack((X1_f, X2_f))
   
    # Getting the training data
    # idx_u = np.random.choice(u.shape[0], N_u, replace=False)
    # X_u_train = np.zeros((N_u, 2))
    # u_train = np.zeros((N_u, 1))
    # for i in range(N_u):
    #     X_u_train[i, :] = X[idx_u[i], :]
    #     u_train[i, :] = u[idx_u[i]]

    X_u_train, u_train = scarcify(X, u, N_u)

    ub = np.array([L_1, L_2])
    lb = np.array([0., 0.])

    Exact_u = u
    Exact_k = k

    return X, Exact_u, Exact_k, X_u_train, u_train, X_f, X_b, ub, lb

def plot_inf_cont_results(X_star, u_star, k_star, kkk, uuu, fff, ub, lb, save_path=None, save_hp=None):

    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    XX, YY = np.meshgrid(x, y)

    K_plot = griddata(X_star, k_star.flatten(), (XX, YY), method="cubic")
    U_plot = griddata(X_star, u_star.flatten(), (XX, YY), method="cubic")
    kkk_mu_pred = np.mean(kkk, axis=1)
    kkk_Sigma_pred = np.var(kkk, axis=1)
    uuu_mu_pred = np.mean(uuu, axis=1)    
    uuu_Sigma_pred = np.var(uuu, axis=1)
    fff_mu_pred = np.mean(fff, axis=1)    
    fff_Sigma_pred = np.var(fff, axis=1)

    K_mu_plot = griddata(X_star, kkk_mu_pred.flatten(), (XX, YY), method="cubic")
    U_mu_plot = griddata(X_star, uuu_mu_pred.flatten(), (XX, YY), method="cubic")
    F_mu_plot = griddata(X_star, fff_mu_pred.flatten(), (XX, YY), method="cubic")
    K_Sigma_plot = griddata(X_star, kkk_Sigma_pred.flatten(), (XX, YY), method="cubic")
    U_Sigma_plot = griddata(X_star, uuu_Sigma_pred.flatten(), (XX, YY), method="cubic")
    F_Sigma_plot = griddata(X_star, fff_Sigma_pred.flatten(), (XX, YY), method="cubic")

    fig1 = plt.figure(2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, K_plot, cmap="viridis")
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel("$x_1$", fontsize=15)
    plt.ylabel("$x_2$", fontsize=15)  
    plt.title("Exact $k(x_1,x_2)$", fontsize=15)

    plt.subplot(2, 2, 2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, K_mu_plot, cmap="viridis")
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel("$x_1$", fontsize=15)
    plt.ylabel("$x_2$", fontsize=15)  
    plt.title("Prediction $k(x_1,x_2)$", fontsize=15)

    plt.subplot(2, 2, 3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, np.abs(K_plot - K_mu_plot), cmap="viridis")
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel("$x_1$", fontsize=15)
    plt.ylabel("$x_2$", fontsize=15)  
    plt.title("Error of $k(x_1,x_2)$", fontsize=15)
    
    plt.subplot(2, 2, 4)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, np.abs(K_plot - K_mu_plot) / K_plot, cmap="viridis")
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel("$x_1$", fontsize=15)
    plt.ylabel("$x_2$", fontsize=15)  
    plt.title("Relative error of $k(x_1,x_2)$", fontsize=15)

    u = uuu
    k = kkk
    u_mu = np.mean(u, axis=1)
    u = np.zeros((10000, 500))
    for i in range(500):
        u[:, i] = u_mu
        
    u = u.reshape(1, -1)
    k = k.reshape(1, -1)
    idx = np.random.choice(5000000, 1000, replace=False)
    u_p = u[:, idx]
    k_p = k[:, idx]


    u = np.linspace(-10., -4.,  1000)
    k = k_vanGenuchten(u)

    fig2 = plt.figure(10,  figsize=(6, 4))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)   
    plt.plot(u_p, k_p, "bo") 
    plt.plot(u, k, "r-", label="Exact", linewidth=2)
    ax = plt.gca()
    plt.xlabel("$u$", fontsize=11)
    plt.ylabel("$K(u)$", fontsize=11)
        

    if save_path != None and save_hp != None:
       saveResultDir(save_path, save_hp, figs=[fig1, fig2])

    else:
       fig1.show()
       fig2.show()

