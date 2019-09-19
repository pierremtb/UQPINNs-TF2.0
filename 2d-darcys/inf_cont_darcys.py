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

eqnPath = "2d-darcys"
sys.path.append(eqnPath)
sys.path.append("utils")
from logger import Logger
from darcysutil import prep_data, plot_inf_cont_results
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
    hp["X_dim"] = 2
    hp["Y_dim"] = 1
    hp["T_dim"] = 0
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
    hp["layers_P_K"] = [hp["Y_dim"],
                        50, 50, 50, 50,
                        hp["Y_dim"]]
    # Setting up the TF SGD-based optimizer (set tf_epochs=0 to cancel it)
    hp["tf_epochs"] = 10000
    hp["tf_lr"] = 0.0001
    hp["tf_b1"] = 0.9
    hp["tf_eps"] = 1e-08
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
    hp["batch_size_u"] = hp["N_u"]
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
    # Frequency of training logs
    hp["log_frequency"] = 1

# %% DEFINING THE MODEL


class DarcysInformedNN(AdvNeuralNetwork):
    def __init__(self, hp, logger, X_f, X_b, ub, lb):
        super().__init__(hp, logger, ub, lb)

        # Normalizing (TODO: find out why we do this (lbb, ubb) thing)
        self.lbb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ubb = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        X_f = self.normalize(X_f)
        X_b = self.normalize_bnd(X_b)

        # Setting up tensors for colloc pts
        self.x1_f = X_f[:, 0:1]
        self.x2_f = X_f[:, 1:self.X_dim]

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

        # The default models are:model_p(X, z), model_q(X, u), model_t(X, u)
        # Additional DNN model for u -> K(u): model_p_K(u),
        # with a transf at the end
        self.model_p_K = self.declare_model(hp["layers_P_K"])
        # self.model_p_K.add(tf.keras.layers.Lambda(
        #     lambda Y: self.ksat * tf.exp(Y)))

    # Normalization functions adapted to our case
    def normalize(self, X):
        return (X - self.lb) - 0.5*(self.ub - self.lb)

    def normalize_bnd(self, X):
        return (X - self.lbb) - 0.5*(self.ubb - self.lbb)

    # Boundary helpers
    @tf.function
    def model_b1(self, XZ):
        x1 = XZ[:, 0:1]
        x2 = XZ[:, 1:self.X_dim]
        z = XZ[:, self.X_dim:self.X_dim+self.Z_dim]
        with tf.GradientTape() as tape:
            tape.watch(x1)
            XZtemp = tf.concat([x1, x2, z], axis=1)
            u = self.model_p(XZtemp)
        u_x1 = tape.gradient(u, x1)
        K = self.model_p_K_final(u)
        temp = self.q + K * u_x1
        return temp

    def model_p_K_final(self, u):
        K = self.model_p_K(u)
        return self.ksat * tf.exp(K)

    @tf.function
    def model_b2(self, XZ): 
        x1 = XZ[:, 0:1]
        x2 = XZ[:, 1:self.X_dim]
        z = XZ[:, self.X_dim:self.X_dim+self.Z_dim]
        with tf.GradientTape() as tape:
            tape.watch(x2)
            XZtemp = tf.concat([x1, x2, z], axis=1)
            u = self.model_p(XZtemp)
        u_x2 = tape.gradient(u, x2)
        return u_x2

    @tf.function
    def model_b3(self, XZ): 
        u = self.model_p(XZ)
        temp = u - self.u_0
        return temp

    @tf.function
    def model_b4(self, XZ):
        x1 = XZ[:, 0:1]
        x2 = XZ[:, 1:self.X_dim]
        z = XZ[:, self.X_dim:self.X_dim+self.Z_dim]
        with tf.GradientTape() as tape:
            tape.watch(x2)
            XZtemp = tf.concat([x1, x2, z], axis=1)
            u = self.model_p(XZtemp)
        u_x2 = tape.gradient(u, x2)
        return u_x2

    @tf.function
    def model_r(self, XZ):
        x1 = XZ[:, 0:1]
        x2 = XZ[:, 1:self.X_dim]
        z_prior = XZ[:, self.X_dim:self.X_dim+self.Z_dim]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x1)
            tape.watch(x2)
            u = self.model_p(tf.concat([x1, x2, z_prior], axis=1))
            u_x1 = tape.gradient(u, x1)
            u_x2 = tape.gradient(u, x2)
            K = self.model_p_K_final(u)
            Ku_x1 = K*u_x1
            Ku_x2 = K*u_x2
        f_1 = tape.gradient(Ku_x1, x1)
        f_2 = tape.gradient(Ku_x2, x2)
        del tape
        return f_1 + f_2

    @tf.function
    def physics_informed_loss(self, f_pred):
        b1_pred = self.model_b1(tf.concat([self.x1_b1, self.x2_b1, self.z_b1],
                                axis=1))
        b2_pred = self.model_b2(tf.concat([self.x1_b2, self.x2_b2, self.z_b2],
                                axis=1))
        b3_pred = self.model_b3(tf.concat([self.x1_b3, self.x2_b3, self.z_b3],
                                axis=1))
        b4_pred = self.model_b4(tf.concat([self.x1_b4, self.x2_b4, self.z_b4],
                                axis=1))
        return tf.reduce_mean(tf.square(f_pred)) + \
            tf.reduce_mean(tf.square(b1_pred)) + \
            tf.reduce_mean(tf.square(b2_pred)) + \
            tf.reduce_mean(tf.square(b3_pred)) + \
            tf.reduce_mean(tf.square(b4_pred))
    
    def wrap_generator_variables(self):
        var = super().wrap_generator_variables()
        var.extend(self.model_p_K.trainable_variables)
        return var

    def generate_latent_variables(self):
        self.z_b1 = np.random.randn(self.x1_b1.shape[0], self.Z_dim)
        self.z_b2 = np.random.randn(self.x1_b2.shape[0], self.Z_dim)
        self.z_b3 = np.random.randn(self.x1_b3.shape[0], self.Z_dim)
        self.z_b4 = np.random.randn(self.x1_b4.shape[0], self.Z_dim)
        return super().generate_latent_variables()

    # The training function
    def fit(self, X_u, u):
        self.logger.log_train_start(self)

        # Creating the tensors
        X_u = self.normalize(X_u)
        X_f = tf.concat([self.x1_f, self.x2_f], axis=1).numpy()

        self.logger.log_train_opt("Adam")
        for epoch in range(self.epochs):
            X_u_batch, u_batch, X_f_batch = self.fetch_minibatch(X_u, u, X_f)
            z_u, z_f = self.generate_latent_variables()
            loss_G, loss_KL, loss_recon, loss_PDE, loss_T = \
                self.optimization_step(X_u_batch, u_batch,
                                       X_f_batch, z_u, z_f)
            loss_str = f"{loss_G:.2e}, KL_loss: {loss_KL:.2e}," + \
                   f"Recon_loss: {loss_recon:.2e}," + \
                   f"PDE_loss: {loss_PDE:.2e}," \
                   f"T_loss: {loss_T:.2e}"
            self.logger.log_train_epoch(epoch, loss_G, custom=loss_str)

        self.logger.log_train_end(self.epochs)

    def predict_u(self, X_star):
        u_star = self.predict_sample(X_star)
        return u_star

    def predict_k(self, X_star):
        u_star = self.predict_u(X_star)
        k_star = self.model_p_K_final(u_star)
        return k_star / self.ksat

    def predict_f(self, X_star):
        X_star = self.normalize(X_star)
        Z = np.random.randn(X_star.shape[0], self.Z_dim)
        f_star = self.model_r(tf.concat([X_star, Z], axis=1))
        return f_star


# %% TRAINING THE MODEL

# Getting the data
path = os.path.join(eqnPath, "data", "nonlinear2d_data.npz")
X_star, Exact_u, Exact_k, X_u_train, u_train, \
    X_f, X_b, ub, lb = prep_data(path, hp["N_u"], hp["N_b"], hp["N_f"],
                                 hp["L_1"], hp["L_2"],
                                 noise=hp["noise"])

# Creating the model
logger = Logger(hp)
pinn = DarcysInformedNN(hp, logger, X_f, X_b, ub, lb)

# Defining the error function for the logger
def error():
    return 0.0
logger.set_error_fn(error)

# Training the PINN
pinn.fit(X_u_train, u_train)

# Predicting
print("Making the predictions…")
N_samples = 500
kkk = np.zeros((X_star.shape[0], N_samples))
uuu = np.zeros((X_star.shape[0], N_samples))
fff = np.zeros((X_star.shape[0], N_samples))
for i in range(0, N_samples):
    kkk[:, i:i+1] = pinn.predict_k(X_star)
    uuu[:, i:i+1] = pinn.predict_u(X_star)
    fff[:, i:i+1] = pinn.predict_f(X_star)

# %% PLOTTING
print("Plotting")
plot_inf_cont_results(X_star, Exact_u.T, Exact_k.T, kkk, uuu, fff, ub, lb,
                      save_path=eqnPath, save_hp=hp)
