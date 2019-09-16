import numpy as np
import tensorflow as tf
import sys
import json
import os
from scipy.interpolate import griddata

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(1234)
tf.random.set_seed(1234)

# %% LOCAL IMPORTS

eqnPath = "1d-burgers"
sys.path.append(eqnPath)
sys.path.append("utils")
from logger import Logger
from burgersutil import prep_data, plot_inf_cont_results
from advneuralnetwork import AdvNeuralNetwork

# %% HYPER PARAMETERS
if len(sys.argv) > 1:
    # if False:
    with open(sys.argv[1]) as hpFile:
        hp = json.load(hpFile)
else:
    hp = {}
    # Data size on u inside the domain
    hp["N_u"] = 200
    # Data size on u on each of the four boundaries
    hp["N_b"] = 100
    # Collocation points on the domain
    hp["N_f"] = 10000
    # Dimension of input, output and latent variable
    hp["X_dim"] = 1
    hp["Y_dim"] = 1
    hp["T_dim"] = 1
    hp["Z_dim"] = 2
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
    # DeepNN topo of the additional NN to infer K(u)
    hp["layers_P_k"] = [hp["Y_dim"],
                        50, 50, 50, 50,
                        hp["Y_dim"]]
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
    hp["batch_size_u"] = hp["N_i"] + hp["N_b"]
    hp["batch_size_f"] = hp["N_f"]
    # Domain size
    hp["L_1"] = 10.
    hp["L_2"] = 10.
    # Initial condition param
    hp["u_0"] = -10.
    # Other
    hp["q"] = 1
    hp["ksat"] = 10
    # Noise on initial data
    hp["noise"] = 0.0

# %% DEFINING THE MODEL


class BurgersInformedNN(AdvNeuralNetwork):
    def __init__(self, hp, logger, X_f, X_b,  ub, lb):
        super().__init__(hp, logger, ub, lb)

        # Normalizing (TODO: find out why we do this (lbb, ubb) thing)
        self.lbb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ubb = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        X_f = self.normalize(X_f)
        X_b = self.normalize_bnd(X_b)

        # Setting up tensors for colloc pts
        self.x1_f = X_f[:, 0:1]
        self.x2_f = X_f[:, 1:2]

        # Setting up boundaries pts
        self.x1_b1 = X_b[:, 0:1]
        self.x2_b1 = X_b[:, 1:2]
        self.x1_b2 = X_b[:, 2:3]
        self.x2_b2 = X_b[:, 3:4]
        self.x1_b3 = X_b[:, 4:5]
        self.x2_b3 = X_b[:, 5:6]
        self.x1_b4 = X_b[:, 6:7]
        self.x2_b4 = X_b[:, 7:8]

        # Specific hyperparameters
        self.u_0 = hp["u_0"]
        self.q = hp["q"]
        self.ksat = hp["ksat"]

        # Additional DNN model for u -> K(u)
        self.model_p_K = self.declare_model(hp["layers_P_k"])

    # Normalization functions adapted to our case
    def normalize(self, X):
        return (X - self.lb) - 0.5*(self.ub - self.lb)

    def normalize_bnd(self, X):
        return (X - self.lbb) - 0.5*(self.ubb - self.lbb)

    # Right-Hand Side
    def f(self, x):
        return tf.zeros_like(x)


    # Boundary helpers TODO: finish
    def get_b1(self, X1, X2, Z):   
        z_prior = Z       
        u = self.model_p(tf.concat([X1, X2, z_prior], axis=1))
        u_x1 = tf.gradients(u, X1)[0]
        k = self.net_P_k(u)
        temp = self.q + k * u_x1
        return temp

    def get_b2(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        u_x2 = tf.gradients(u, X2)[0]
        return u_x2

    def get_b3(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        temp = u - self.u_0
        return temp

    def get_b4(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        u_x2 = tf.gradients(u, X2)[0]
        return u_x2

    # TODO: the f thing
    def model_f(self, X1, X2, Z_u):
        u = self.net_P_u(X1, X2, Z_u)
        k = self.net_P_k(u)
        u_x1 = tf.gradients(u, X1)[0]
        u_x2 = tf.gradients(u, X2)[0]
        f_1 = tf.gradients(k*u_x1, X1)[0]
        f_2 = tf.gradients(k*u_x2, X2)[0]
        f = f_1 + f_2
        return f

    # TODO: remove this old (keeping as ex for model_f)
    def model_r(self, XZ_f):
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

    # TODO: override the losses (and maybe grads)

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

            Z_u = np.random.randn(self.batch_size_u, 1)
            Z_f = np.random.randn(self.batch_size_f, 1)

            # Dual-Optimization step
            for _ in range(self.k1):
                loss_T, grads = \
                    self.discriminator_grad(X_u_batch, u_batch, Z_u)
                self.optimizer_T.apply_gradients(
                    zip(grads, self.wrap_discriminator_variables()))
            for _ in range(self.k2):
                loss_G, loss_KL, loss_recon, loss_PDE, grads = \
                    self.generator_grad(X_u_batch, u_batch,
                                        X_f_batch, Z_u, Z_f)
                self.optimizer_KL.apply_gradients(
                    zip(grads, self.wrap_generator_variables()))

            loss_str = f"KL_loss: {loss_KL:.2e}," + \
                       f"Recon_loss: {loss_recon:.2e}," + \
                       f"PDE_loss: {loss_PDE:.2e}," \
                       f"T_loss: {loss_T:.2e}"
            self.logger.log_train_epoch(epoch, loss_G, custom=loss_str)

        self.logger.log_train_end(self.epochs)

    # TODO: update the predictions
    
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
            samples_mean[:, i:i+1] = self.generate_sample(X_star)

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
path = os.path.join(eqnPath, "data", "nonlinear2d_data.npz")
X_star, Exact_u, Exact_k, X_u_train, u_train, \
    X_f, ub, lb = prep_data(path, hp["N_u"], hp["N_b"], hp["N_f"],
                            hp["L_1"], hp["L_2"],
                            noise=hp["noise"])

# Creating the model
logger = Logger(frequency=100, hp=hp)
pinn = BurgersInformedNN(hp, logger, X_f, ub, lb)

# Defining the error function for the logger
def error():
    return 0.0
logger.set_error_fn(error)

# Training the PINN
pinn.fit(X_u_train, u_train)

N_samples = 500
kkk = np.zeros((X_star.shape[0], N_samples))
uuu = np.zeros((X_star.shape[0], N_samples))
fff = np.zeros((X_star.shape[0], N_samples))
for i in range(0, N_samples):
    kkk[:, i:i+1] = pinn.predict_k(X_star)
    uuu[:, i:i+1] = pinn.predict_u(X_star)
    fff[:, i:i+1] = pinn.predict_f(X_star)

# %% PLOTTING
plot_inf_cont_results(X_star, Exact_u.T, Exact_k.T, kkk, uuu, fff, ub, lb,
                      save_path=eqnPath, save_hp=hp)
