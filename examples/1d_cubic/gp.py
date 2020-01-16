import sys
import os
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join("..", ".."))
from podnn.advneuralnetwork import AdvNeuralNetwork
from podnn.metrics import re_mean_std, re
from podnn.mesh import create_linear_mesh
from podnn.logger import Logger
from podnn.advneuralnetwork import NORM_MEANSTD, NORM_NONE
from podnn.plotting import figsize


x_star = np.linspace(-6, 6, 100).reshape(100, 1)
u_star = x_star**3

N = 20
lb = int(2/(2*6) * 100)
ub = int((2+2*4)/(2*6) * 100)
idx = np.random.choice(x_star[lb:ub].shape[0], N, replace=False)
x_train = x_star[lb + idx]
u_train = u_star[lb + idx]
noise_std = 6
u_train = u_train + noise_std*np.random.randn(u_train.shape[0], u_train.shape[1])

data = (x_train, u_train)

# plt.plot(x_star, u_star)
# plt.scatter(x_train, u_train)
# plt.show()
# exit(0)
# 
# 
from typing import Tuple, Optional
from pathlib import Path
import datetime
import io
import gpflow
from gpflow.config import default_float
import warnings
warnings.filterwarnings('ignore')
from gpflow.utilities import print_summary

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, u_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_star, u_star))

batch_size = N
num_features = 1
prefetch_size = N // 2
shuffle_buffer_size = N // 2
num_batches_per_epoch = N // batch_size

original_train_dataset = train_dataset
train_dataset = train_dataset.repeat()\
                    .prefetch(prefetch_size)\
                    .shuffle(buffer_size=shuffle_buffer_size)\
                    .batch(batch_size)

# Instantiate a Gaussian Process model
# kernel = gpflow.kernels.RBF(variance=2.)
kernel = gpflow.kernels.Matern52()
model = gpflow.models.GPR(data=(x_train, u_train), kernel=kernel, mean_function=None)

# likelihood = gpflow.likelihoods.Gaussian()
# num_features = 1
# inducing_variable = np.linspace(0, 10, num_features).reshape(-1, 1)
# model = gpflow.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable)
print_summary(model)  # same as print_summary(model, fmt="simple")
model.likelihood.variance.assign(0.01)
model.kernel.lengthscale.assign(0.3)
optimizer = gpflow.optimizers.Scipy()

def objective_closure():
    return - model.log_marginal_likelihood()

opt_logs = optimizer.minimize(objective_closure,
                        model.trainable_variables,
                        options=dict(maxiter=100))
print_summary(model)
# optimizer = tf.optimizers.Adam()

# def optimization_step(model: gpflow.models.SVGP, batch: Tuple[tf.Tensor, tf.Tensor]):
#     with tf.GradientTape(watch_accessed_variables=False) as tape:
#         tape.watch(model.trainable_variables)
#         obj = - model.elbo(*batch)
#         grads = tape.gradient(obj, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

# def simple_training_loop(model: gpflow.models.SVGP, epochs: int = 1, logging_epoch_freq: int = 10):
#     batches = iter(train_dataset)
#     tf_optimization_step = tf.function(optimization_step, autograph=False)
#     for epoch in range(epochs):
#         for _ in range(num_batches_per_epoch):
#             tf_optimization_step(model, next(batches))

#         epoch_id = epoch + 1
#         if epoch_id % logging_epoch_freq == 0:
#             tf.print(f"Epoch {epoch_id}: ELBO (train) {model.elbo(*data)}")

# simple_training_loop(model, epochs=90000, logging_epoch_freq=200)

# Make the prediction on the meshed x-axis (ask for MSE as well)
u_pred, u_pred_var = model.predict_f(x_star)
u_pred_samples = model.predict_f_samples(x_star, 10)

fig = plt.figure(figsize=figsize(1, 1, scale=2.5))
lower = u_pred - 3 * np.sqrt(u_pred_var)
upper = u_pred + 3 * np.sqrt(u_pred_var)
plt.fill_between(x_star[:, 0], lower[:, 0], upper[:, 0], 
                    facecolor='C0', alpha=0.3, label=r"$3\sigma_{T}(x)$")
# plt.plot(x_star, u_pred_samples[:, :, 0].numpy().T, 'C0', linewidth=.5)
plt.scatter(x_train, u_train, c="r", label=r"$u_T(x)$")
plt.plot(x_star, u_star, "r--", label=r"$u_*(x)$")
plt.plot(x_star, u_pred, label=r"$\hat{u}_*(x)$")
plt.legend()
plt.xlabel("$x$")
# plt.show()
plt.savefig("results/gp.pdf")

# U_pred, U_pred_sig = model.predict(X_U_test)
# print(X_U_test)
# print(X_U_test.shape)
# print(U_pred.shape)