import numpy as np
import tensorflow as tf
import sys
import json
import os
from scipy.interpolate import griddata

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

# %% LOCAL IMPORTS

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from logger import Logger
from burgersutil import prep_data, plot_inf_cont_results
from advneuralnetwork import AdvNeuralNetwork

# %% HYPER PARAMETERS
# plt.savefig("./Initial.png", dpi=600)
if len(sys.argv) > 1:
    # if False:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Data size on the initial condition solution
    hp["N_i"] = 50
    # Collocation points on the boundaries
    hp["N_b"] = 100
    # Collocation points on the domain
    hp["N_f"] = 10000
    # Dimension of input, output and latent variable
    hp["X_dim"] = 1
    hp["Y_dim"] = 1
    hp["T_dim"] = 1
    hp["Z_dim"] = 1
    # DeepNNs topologies
    hp["layers_P"] = [hp["X_dim"]+hp["T_dim"] + hp["Z_dim"],
                      50, 50, 50, 50,
                      hp["Y_dim"]]
    hp["layers_Q"] = [hp["X_dim"]+hp["T_dim"] + hp["Y_dim"],
                      50, 50, 50, 50,
                      hp["Z_dim"]]
    hp["layers_T"] = [hp["X_dim"]+hp["T_dim"]+hp["Y_dim"],
                      50, 50, 50,
                      1]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 10000
    hp["tf_lr"] = 0.0001
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = None
    # Setting up the quasi-newton LBGFS optimizer (set nt_epochs=0 to cancel)
    # hp["nt_epochs"] = 500
    # hp["nt_lr"] = 1.0
    # hp["nt_ncorr"] = 50
    # Loss coefficients
    hp["lambda"] = 1.5
    hp["beta"] = 1.0
    # MinMax switching
    hp["k1"] = 1
    hp["k2"] = 5
    # Batch size
    # hp["batch_size_u"] = hp["N_i"] + hp["N_b"]
    hp["batch_size_u"] = 1000
    hp["batch_size_f"] = hp["N_f"]
    # Noise on initial data
    hp["noise"] = 0.0
    hp["noise_is_gaussian"] = False
    # Logging
    hp["log_frequency"] = 100

# %% DEFINING THE MODEL


class BurgersInformedNN(AdvNeuralNetwork):
    def __init__(self, hp, logger, X_f, ub, lb):
        super().__init__(hp, logger, ub, lb)

        # Separating the collocation coordinates and normalizing
        x_f = X_f[:, 0:1]
        t_f = X_f[:, 1:2]

        self.x_mean = x_f.mean(0)
        self.x_std = x_f.std(0)
        self.t_mean = t_f.mean(0)
        self.t_std = t_f.std(0)
        self.x_f = self.tensor(self.normalize_x(x_f))
        self.t_f = self.tensor(self.normalize_t(t_f))

        self.Jacobian_X = 1 / self.x_std
        self.Jacobian_T = 1 / self.t_std

    def normalize_x(self, x):
        return (x - self.x_mean) / self.x_std

    def normalize_t(self, t):
        return (t - self.t_mean) / self.t_std

    def normalize(self, X):
        x = self.normalize_x(X[:, 0:1])
        t = self.normalize_t(X[:, 1:2])
        return np.concatenate((x, t), axis=1)

    def f(self, x):
        return tf.zeros_like(x)
    
    @tf.function
    def model_r(self, XZ_f):
        return self.tensor(0.)
        x_f = XZ_f[:, 0:1]
        t_f = XZ_f[:, 1:2]
        z_prior = XZ_f[:, 2:3]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_f)
            tape.watch(t_f)
            X = tf.concat([x_f, t_f], axis=1)
            u = self.model_p(tf.concat([X, z_prior], axis=1))
            u_x = tape.gradient(u, x_f)
        u_t = tape.gradient(u, t_f)
        u_xx = tape.gradient(u_x, x_f)
        del tape
        f = self.f(x_f)
        r = (self.Jacobian_T) * u_t + \
            (self.Jacobian_X) * u * u_x - \
            (0.01/np.pi) * (self.Jacobian_X ** 2) * u_xx - f
        return r

    # Fetches a mini-batch of data
    def fetch_minibatch(self, X_u, u, X_f):
        N_u = X_u.shape[0]
        N_f = X_f.shape[0]
        idx_u = np.random.choice(N_u, self.batch_size_u, replace=False)
        idx_f = np.random.choice(N_f, self.batch_size_f, replace=False)
        X_u_batch = self.tensor(X_u[idx_u, :])
        X_f_batch = self.tensor(X_f[idx_f, :])
        u_batch = self.tensor(u[idx_u, :])
        return X_u_batch, u_batch, X_f_batch

    # The training function
    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.normalize(X_u)
        X_f = tf.concat([self.x_f, self.t_f], axis=1).numpy()

        self.logger.log_train_opt("Adam")
        for epoch in range(self.epochs):
            X_u_batch, u_batch, X_f_batch = self.fetch_minibatch(X_u, u, X_f)
            z_u, z_f = self.generate_latent_variables()
            loss_G, loss_KL, loss_recon, loss_PDE, loss_T = \
                    self.optimization_step(X_u_batch, u_batch, X_f_batch, z_u, z_f)
            loss_str = f"KL_loss: {loss_KL:.2e}," + \
                       f"Recon_loss: {loss_recon:.2e}," + \
                       f"PDE_loss: {loss_PDE:.2e}," \
                       f"T_loss: {loss_T:.2e}"
            self.logger.log_train_epoch(epoch, loss_G, custom=loss_str)

        self.logger.log_train_end(self.epochs)

    def predict_f(self, X_star):
        # Center around the origin
        X_star_norm = self.tensor(self.normalize(X_star))
        # Predict
        z_f = self.tensor(np.random.randn(X_star.shape[0], self.Z_dim))
        f_star = self.model_r(tf.concat([X_star_norm, z_f], axis=1))
        return f_star

    def predict(self, X_star, X, T):
        N_samples = 500
        samples_mean = np.zeros((X_star.shape[0], N_samples))
        for i in range(0, N_samples):
            samples_mean[:, i:i+1] = self.predict_sample(X_star)

        XT = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

        # Compare mean and variance of the predicted samples
        # as prediction and uncertainty
        U_pred = np.mean(samples_mean, axis=1)
        U_pred = griddata(XT, U_pred.flatten(), (X, T), method='cubic')
        Sigma_pred = np.var(samples_mean, axis=1)
        Sigma_pred = griddata(XT, Sigma_pred.flatten(), (X, T), method='cubic')

        return U_pred, Sigma_pred


# %% TRAINING THE MODEL

# Getting the data
path = os.path.join(eqnPath, "data", "burgers_shock.mat")
x, t, X, T, Exact_u, X_star, u_star, X_u_train, u_train, \
    X_f, ub, lb = prep_data(path, hp["N_i"], hp["N_b"], hp["N_f"],
                            noise=hp["noise"],
                            noise_is_gaussian=hp["noise_is_gaussian"])

# Creating the model
logger = Logger(hp)
pinn = BurgersInformedNN(hp, logger, X_f, ub, lb)

# Defining the error function for the logger


def error():
    return 0.0
    U_pred, _ = pinn.predict(X_star, X, T)
    return np.linalg.norm(Exact_u-U_pred, 2)/np.linalg.norm(Exact_u, 2)


logger.set_error_fn(error)

# Training the PINN
pinn.fit(X_u_train, u_train)

# Getting the model predictions and calculating relative errror
U_pred, Sigma_pred = pinn.predict(X_star, X, T)
error_u = np.linalg.norm(Exact_u - U_pred, 2)/np.linalg.norm(Exact_u, 2)

# Compare the relative error between the prediciton and the reference solution
error_u = np.linalg.norm(Exact_u - U_pred, 2)/np.linalg.norm(Exact_u, 2)
print("Error on u: ", error_u)

# %% PLOTTING
plot_inf_cont_results(X_star, U_pred, Sigma_pred,
                      X_u_train, u_train, Exact_u, X, T, x, t,
                      save_path=eqnPath, save_hp=hp)
