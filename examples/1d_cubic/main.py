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


x_star = np.linspace(-6, 6, 100).reshape(100, 1)
u_star = x_star**3

N = 20
lb = int(2/(2*6) * 100)
ub = int((2+2*4)/(2*6) * 100)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 3
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

# plt.plot(x_star, u_star)
# plt.scatter(x_train, u_train)
# plt.show()
# exit(0)

X_dim = 1
Y_dim = 1
Z_dim = 1 
h_layers = [50, 50, 50, 50]
h_layers_t = [50, 50, 50]
layers_p = [X_dim+Z_dim, *h_layers, Y_dim]
layers_q = [X_dim+Y_dim, *h_layers, Z_dim]
layers_t = [X_dim+Y_dim, *h_layers_t, 1]
layers = (layers_p, layers_q, layers_t)

model = AdvNeuralNetwork(layers, (X_dim, Y_dim, Z_dim),
                            0.0001, 1.0, 1e0, 1, 1, NORM_MEANSTD)
                            
model.summary()

epochs = 30000
logger = Logger(epochs, 1000)

def get_val_err():
    u_pred = model.predict_sample(x_star)
    return {
        "RE": re(u_pred, u_star),
    }
    # U_val_pred, _ = regnn.predict(X_U_val)
    # U_val_pred = model.predict_sample(X_U_val)
    # return {
    #     "RE": re(U_val_pred[:, 0], U_val[:, 0])
    # }
logger.set_val_err_fn(get_val_err)

# Training
model.fit(x_train, u_train, epochs, logger)

# Predict and restruct
u_pred, u_pred_sig = model.predict(x_star)
plt.plot(x_star, u_star)
plt.scatter(x_train, u_train)
plt.plot(x_star, u_pred, "r--")
lower = u_pred - 2 * u_pred_sig
upper = u_pred + 2 * u_pred_sig
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0], 
                    facecolor='orange', alpha=0.5, label=r"$2\sigma_{T,hf}(x)$")
plt.show()

# U_pred, U_pred_sig = model.predict(X_U_test)
# print(X_U_test)
# print(X_U_test.shape)
# print(U_pred.shape)