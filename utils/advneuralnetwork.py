import numpy as np
import tensorflow as tf


class AdvNeuralNetwork(object):
    def __init__(self, hp, logger, ub, lb):

        self.ub = ub
        self.lb = lb

        # Setting up the optimizers with the previously defined hp
        self.epochs = hp["tf_epochs"]
        self.optimizer_KL = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])
        self.optimizer_T = tf.keras.optimizers.Adam(
            learning_rate=hp["tf_lr"],
            beta_1=hp["tf_b1"],
            epsilon=hp["tf_eps"])

        self.dtype = "float64"
        # Descriptive Keras models
        tf.keras.backend.set_floatx(self.dtype)
        self.model_p = self.declare_model(hp["layers_P"])
        self.model_q = self.declare_model(hp["layers_Q"])
        self.model_t = self.declare_model(hp["layers_T"])

        # Hp
        self.X_dim = hp["X_dim"]
        self.T_dim = hp["T_dim"]
        self.Y_dim = hp["Y_dim"]
        self.Z_dim = hp["Z_dim"]
        self.kl_lambda = hp["lambda"]
        self.kl_beta = hp["beta"]
        self.k1 = hp["k1"]
        self.k2 = hp["k2"]
        self.batch_size_u = hp["batch_size_u"]
        self.batch_size_f = hp["batch_size_f"]

        self.logger = logger
        #self.dtype = tf.float64

    def declare_model(self, layers):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        for width in layers[1:]:
            model.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer="glorot_normal"))
        return model

    def physics_informed_loss(self, f_pred):
        return tf.reduce_mean(tf.square(f_pred))

    # Mininizing the G-Loss
    @tf.function
    def generator_loss(self, X_u, u, u_pred, f_pred, Z_u):
        # Prior:
        z_u_prior = Z_u
        # Encoder: q(z|x,y)
        z_u_encoder = self.model_q(tf.concat([X_u, u_pred], axis=1))

        # Discriminator loss
        Y_pred = self.model_p(tf.concat([X_u, Z_u], axis=1))
        T_pred = self.model_t(tf.concat([X_u, Y_pred], axis=1))

        # KL-divergence between the data and model distributions
        KL = tf.reduce_mean(T_pred)

        # Entropic regularization
        log_q = -tf.reduce_mean(tf.square(z_u_prior - z_u_encoder))

        # Physics-informed loss
        loss_f = self.physics_informed_loss(f_pred)

        # Generator loss
        loss = KL + (1.0 - self.kl_lambda)*log_q + self.kl_beta * loss_f

        return loss, KL, (1.0 - self.kl_lambda)*log_q, self.kl_beta * loss_f

    # Minimizing the D-loss
    @tf.function
    def discriminator_loss(self, X_u, u, Z_u):
        # Prior: p(z)
        z_prior = Z_u
        # Decoder: p(y|x,z)
        u_pred = self.model_p(tf.concat([X_u, z_prior], axis=1))

        # Discriminator loss
        T_real = self.model_t(tf.concat([X_u, u], axis=1))
        T_fake = self.model_t(tf.concat([X_u, u_pred], axis=1))

        T_real = tf.sigmoid(T_real)
        T_fake = tf.sigmoid(T_fake)

        T_loss = -tf.reduce_mean(tf.math.log(1.0 - T_real + 1e-8) +
                                 tf.math.log(T_fake + 1e-8))

        return T_loss

    @tf.function
    def generator_grad(self, X_u, u, X_f, Z_u, Z_f):
        with tf.GradientTape(persistent=True) as tape:
            u_pred = self.model_p(tf.concat([X_u, Z_u], axis=1))
            f_pred = self.model_r(tf.concat([X_f, Z_f], axis=1))
            loss_G, KL, recon, loss_PDE = \
                self.generator_loss(X_u, u, u_pred, f_pred, Z_u)
        grads = tape.gradient(loss_G, self.wrap_generator_variables())
        del tape
        return loss_G, KL, recon, loss_PDE, grads

    @tf.function
    def discriminator_grad(self, X_u, u, Z_u):
        with tf.GradientTape(persistent=True) as tape:
            loss_T = self.discriminator_loss(X_u, u, Z_u)
        grads = tape.gradient(loss_T, self.wrap_discriminator_variables())
        del tape
        return loss_T, grads

    # right hand side terms of the PDE
    def f(self, X):
        raise NotImplementedError()

    def model_r(self, XZ_f):
        raise NotImplementedError()

    def wrap_generator_variables(self):
        var = []
        var.extend(self.model_p.trainable_variables)
        var.extend(self.model_q.trainable_variables)
        return var

    def wrap_discriminator_variables(self):
        var = []
        var.extend(self.model_t.trainable_variables)
        return var

    def generate_latent_variables(self):
        z_u = np.random.randn(self.batch_size_u, self.Z_dim)
        z_f = np.random.randn(self.batch_size_f, self.Z_dim)
        return z_u, z_f

    def summary(self):
        return self.model_p.summary()

    def normalize(self, X):
        raise NotImplementedError()

    def tensor(self, X):
        return tf.convert_to_tensor(X, dtype=self.dtype)

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

    @tf.function
    def optimization_step(self, X_u_batch, u_batch, X_f_batch, z_u, z_f):
        # Dual-Optimization step
        for _ in range(self.k1):
            loss_T, grads = \
                self.discriminator_grad(X_u_batch, u_batch, z_u)
            self.optimizer_T.apply_gradients(
                zip(grads, self.wrap_discriminator_variables()))
        for _ in range(self.k2):
            loss_G, loss_KL, loss_recon, loss_PDE, grads = \
                self.generator_grad(X_u_batch, u_batch,
                                    X_f_batch, z_u, z_f)
            self.optimizer_KL.apply_gradients(
                zip(grads, self.wrap_generator_variables()))
        return loss_G, loss_KL, loss_recon, loss_PDE, loss_T

    # Generate samples of y given x by sampling from the latent space z
    def sample_generator(self, X, Z):
        # Prior:
        z_prior = Z
        # Decoder: p(y|x,z)
        Y_pred = self.model_p(tf.concat([X, z_prior], axis=1))
        return Y_pred

    # Predict y given x
    def generate_sample(self, X_star):
        X_star = self.tensor(self.normalize(X_star))
        Z = np.random.randn(X_star.shape[0], 1)
        Y_star = self.sample_generator(X_star, Z)
        return Y_star
