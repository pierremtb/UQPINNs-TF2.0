#%%
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join("..", ".."))
from podnn.advneuralnetwork import AdvNeuralNetwork
from podnn.metrics import re_mean_std, re
from podnn.mesh import create_linear_mesh
from podnn.logger import Logger
from podnn.advneuralnetwork import NORM_MEANSTD, NORM_NONE
from podnn.plotting import figsize, savefig


#%% Datagen
N_tst = 300
x_tst = np.linspace(-6, 6, N_tst).reshape(-1, 1)
D = 1
y_tst = x_tst**3

N = 20
lb = int(2/(2*6) * N_tst)
ub = int((2+2*4)/(2*6) * N_tst)
# idx = np.random.choice(x_tst[lb:ub].shape[0], N, replace=False)
idx = np.array([ 58, 194, 192,  37,  55, 148,  77, 144, 197, 190,  15,  97, 171,
        91, 100, 188,   8,  63,  98,  78])
x = x_tst[lb + idx]
y = y_tst[lb + idx]
# noise_std = 0.01*u_train.std(0)
noise_std = 9
y = y + noise_std*np.random.randn(y.shape[0], y.shape[1])

X_dim = 1
Y_dim = 1
Z_dim = 1 
h_layers = [20, 20]
h_layers_t = [20]
layers_p = [X_dim+Z_dim, *h_layers, Y_dim]
layers_q = [X_dim+Y_dim, *h_layers, Z_dim]
layers_t = [X_dim+Y_dim, *h_layers_t, 1]
layers = (layers_p, layers_q, layers_t)


model = AdvNeuralNetwork(layers, (X_dim, Y_dim, Z_dim),
                          0.001, 1.5, 0.1, 1, 5, NORM_MEANSTD)
epochs = 15000
logger = Logger(epochs, 1000)
logger.set_val_err_fn(lambda: {})
model.fit(x, y, epochs, logger)

# Predict and restruct
u_pred, u_pred_var = model.predict(x_tst)
u_pred_sig = np.sqrt(u_pred_var)

lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
plt.fill_between(x_tst.ravel(), upper.ravel(), lower.ravel(), 
                    facecolor='C0', alpha=0.3, label=r"$2\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.plot(x_tst, u_pred, "b-", label=r"$\hat{u}_*(x)$")
plt.scatter(x, y, c="r", label=r"$u_T(x)$")
plt.plot(x_tst, y_tst, "r--", label=r"$u_*(x)$")
plt.ylim((y_tst.min(), y_tst.max()))
plt.xlabel("$x$")
plt.legend()
plt.tight_layout()
plt.savefig(f"uq-toy-uqpinn.pdf", bbox_inches='tight', pad_inches=0)
