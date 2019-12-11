import scipy.io
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


def scarcify(X, u, N): 
    idx = np.random.choice(X.shape[0], N, replace=False)
    return X[idx, :], u[idx, :]

def prep_data(path, N_i, N_b, N_f, noise=0.0, noise_is_gaussian=True):

    # Reading external data [t is 100x1, usol is 256x100, x is 256x1]
    data = scipy.io.loadmat(path)

    # Flatten makes [[]] into []
    # [:,None] makes it a column vector
    # t is T x 1, x is N x 1
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]

    # Keeping the 2D data for the solution data (T x N)
    Exact_u = np.real(data['usol']).T

    # Meshing x and t in 2D (256,100)
    X, T = np.meshgrid(x, t)

    # Preparing the inputs (meshed as X, T) for pred as X_star
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Preparing the testing u_star
    u_star = Exact_u.flatten()[:, None]
                
    # Getting the domain bounds 
    lb = X_star.min(axis=0)
    ub = X_star.max(axis=0)

    # Getting the initial conditions (t=0)
    # X_i = np.hstack((X[0:1, :].T, T[0:1, :].T))
    X_u_train, u_train = scarcify(X_star, u_star, 1000)
    # u_i = Exact_u[0:1, :].T    
    # u_train = u_star    
    # X_i, u_i_0 = scarcify(X_i, u_i, N_i)
    # if noise_is_gaussian or noise == 0.0:
    #     u_i = u_i_0 + \
    #         noise*np.std(u_i_0)*np.random.randn(u_i_0.shape[0], u_i_0.shape[1])
    # else:
    #     # Fabricating a non-additive, non gaussian noise
    #     x_i = X_i[:, 0:1]
    #     error = 1.0/np.exp(3.0*(abs(x_i))) * \
    #         np.random.normal(0, noise, x_i.size)[:, None]
    #     u_i = -np.sin(np.pi*(x_i + 2*error)) + error
    u_train = u_train + \
        noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    # else:
    #     # Fabricating a non-additive, non gaussian noise
    #     x_i = X_u_train[:, 0:1]
    #     error = 1.0/np.exp(3.0*(abs(x_i))) * \
    #         np.random.normal(0, noise, x_i.size)[:, None]
    #     u_train = -np.sin(np.pi*(x_i + 2*error)) + error
    # # Getting the lowest boundary conditions (x=-1) 
    # X_bl = np.hstack((X[:, 0:1], T[:, 0:1]))
    # u_bl = Exact_u[:, 0:1]
    # X_bl, u_bl = scarcify(X_bl, u_bl, N_b // 2)
    # # Getting the highest boundary conditions (x=1) 
    # X_bh = np.hstack((X[:, -1:], T[:, -1:]))
    # u_bh = Exact_u[:, -1:]
    # X_bh, u_bh = scarcify(X_bh, u_bh, N_b // 2)
    
    # Stacking them in multidimensional tensors for training
    # (X_u_train is for now the continuous boundaries)
    # X_u_train = np.vstack([X_i, X_bl, X_bh])
    # u_train = np.vstack([u_i, u_bl, u_bh])

    # Generating the x and t colloc points for f, with each having a N_f size
    # We pointwise add and multiply to spread the LHS over the 2D domain
    X_f_train = lb + (ub-lb)*lhs(2, N_f)

    # Plot the exact initial condition with the data for the initial condition
    # plt.figure(1, figsize=(6, 4))
    # plt.xticks(fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.plot(X[0:1, :].T, Exact_u[0:1, :].T, "b-", label="Exact", linewidth=2)
    # plt.plot(X_i[:, 0:1], u_i, "kx", label="Non-noisy initial condition",
    #          alpha=1.)
    # plt.legend(loc="upper right", frameon=False, prop={'size': 11})
    # plt.gca()
    # plt.xlim(-1.0, 1.0)
    # plt.xlabel("$x$", fontsize=11)
    # plt.ylabel("$u(0, x)$", fontsize=11)
    # savefig(f"init-" + datetime.now().strftime('%Y%m%d-%H%M%S'))
    
    return x, t, X, T, Exact_u, X_star, u_star, \
        X_u_train, u_train, X_f_train, ub, lb

def plot_inf_cont_results(X_star, U_pred, Sigma_pred, X_u_train, u_train, Exact_u,
  X, T, x, t, save_path=None, save_hp=None):

  # Interpolating the results on the whole (x,t) domain.
  # griddata(points, values, points at which to interpolate, method)
  # U_pred = griddata(X_star, u_pred, (X, T), method='cubic')

  # Creating the figures
  fig, ax = newfig(1.0, 1.1)
  ax.axis('off')

  X_u = X_u_train[:, 0:1]
  T_u = X_u_train[:, 1:2]
  Y_u = u_train
  Exact = Exact_u

  idx25 = np.where(X_u_train[:, 1] == 0.25)[0]
  idx50 = np.where(X_u_train[:, 1] == 0.50)[0]
  idx75 = np.where(X_u_train[:, 1] == 0.75)[0]

  ####### Row 0: u(t,x) ##################    
  gs0 = gridspec.GridSpec(1, 2)
  gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
  ax = plt.subplot(gs0[:, :])
  
  h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                extent=[t.min(), t.max(), x.min(), x.max()], 
                origin='lower', aspect='auto')
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  fig.colorbar(h, cax=cax)
  
  ax.plot(T_u, X_u, 'kx', label = 'Data (%d points)' % (Y_u.shape[0]), markersize = 4, clip_on = False)
  
  line = np.linspace(x.min(), x.max(), 2)[:,None]
  ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
  ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
  ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
  
  ax.set_xlabel('$t$')
  ax.set_ylabel('$x$')
  ax.legend(frameon=False, loc = 'best')
  ax.set_title('$u(t,x)$', fontsize = 10)


  ####### Row 1: u(t,x) slices ##################    
  gs1 = gridspec.GridSpec(1, 3)
  gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
  
  ax = plt.subplot(gs1[0, 0])
  ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
  ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
  ax.plot(X_u[idx25], Y_u[idx25], 'kx', linewidth = 2, label = 'Data')
  lower = U_pred[25,:] - 2.0*np.sqrt(Sigma_pred[25,:])
  upper = U_pred[25,:] + 2.0*np.sqrt(Sigma_pred[25,:])
  plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
  ax.set_xlabel('$x$')
  ax.set_ylabel('$u(t,x)$')    
  ax.set_title('$t = 0.25$', fontsize = 10)
  ax.axis('square')
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])
  
  ax = plt.subplot(gs1[0, 1])
  ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
  ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
  ax.plot(X_u[idx50], Y_u[idx50], 'kx', linewidth = 2, label = 'Data')
  lower = U_pred[50,:] - 2.0*np.sqrt(Sigma_pred[50,:])
  upper = U_pred[50,:] + 2.0*np.sqrt(Sigma_pred[50,:])
  plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
  ax.set_xlabel('$x$')
  ax.set_ylabel('$u(t,x)$')
  ax.axis('square')
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])
  ax.set_title('$t = 0.50$', fontsize = 10)
  ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
  
  ax = plt.subplot(gs1[0, 2])
  ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
  ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
  ax.plot(X_u[idx75], Y_u[idx75], 'kx', linewidth = 2, label = 'Data')
  lower = U_pred[75,:] - 2.0*np.sqrt(Sigma_pred[75,:])
  upper = U_pred[75,:] + 2.0*np.sqrt(Sigma_pred[75,:])
  plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                    facecolor='orange', alpha=0.5, label="Two std band")
  ax.set_xlabel('$x$')
  ax.set_ylabel('$u(t,x)$')
  ax.axis('square')
  ax.set_xlim([-1.1,1.1])
  ax.set_ylim([-1.1,1.1])    
  ax.set_title('$t = 0.75$', fontsize = 10)


  # savefig('./Prediction')
  

  # fig, ax = newfig(1.0)
  # ax.axis('off')
  
  # #############       Uncertainty       ##################
  # gs2 = gridspec.GridSpec(1, 2)
  # gs2.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
  # ax = plt.subplot(gs2[:, :])
  
  # h = ax.imshow(Sigma_pred.T, interpolation='nearest', cmap='rainbow', 
  #               extent=[t.min(), t.max(), x.min(), x.max()], 
  #               origin='lower', aspect='auto')
  # divider = make_axes_locatable(ax)
  # cax = divider.append_axes("right", size="5%", pad=0.05)
  # fig.colorbar(h, cax=cax)
  # ax.set_xlabel('$t$')
  # ax.set_ylabel('$x$')
  # ax.legend(frameon=False, loc = 'best')
  # ax.set_title('Variance of $u(t,x)$', fontsize = 10)

  if save_path != None and save_hp != None:
      saveResultDir(save_path, save_hp)

  else:
    plt.show()


#%%
