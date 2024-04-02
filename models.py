import numpy as np
from scipy.special import logsumexp
import tensorflow as tf

# from gpflow.decors import params_as_tensors
import gpflow
from gpflow.models.svgp import SVGP
import gpflow.monitor as gpmon
from gpflow.monitor import MonitorTask


class DynamicCovarianceRegression(SVGP):
    """
    Effectively a helper wrapping call to SVGP.
    """
    def __init__(self, kern, likelihood, inducing_variable, minibatch_size=None, whiten=True):
        """

        :param X:
        :param Y:
        :param kern:
        :param likelihood:
        :param inducing_variable:
        :param minibatch_size:
        :param whiten:
        """
        cov_dim = likelihood.cov_dim
        nu = likelihood.nu

        super().__init__(kernel=kern, likelihood=likelihood,
                         mean_function=None,
                         num_latent_gps=cov_dim * nu,
                         q_diag=False,
                         whiten=whiten,
                         inducing_variable=inducing_variable,
                         )

        self.construct_predictive_density()

    def construct_predictive_density(self):
        # No more placeholders; use tf.Variable or function arguments instead
        D = self.likelihood.D
        self.X_new = tf.Variable([[0]], dtype=tf.float64, shape=[None, 1])  # Example, shape will be dynamic
        self.Y_new = tf.Variable([[0] * D], dtype=tf.float64, shape=[None, D])  # Example, shape will be dynamic
        self.n_samples = tf.Variable(0, dtype=tf.float64, shape=[])  # Example, initial value will be replaced
        # self.X_new = tf.zeros([0, 1], dtype=tf.float64)  # Use an empty tensor as a placeholder with the dynamic shape
        # self.Y_new = tf.zeros([0, D], dtype=tf.float64)  # Use an empty tensor as a placeholder with the dynamic shape
        # self.n_samples = tf.constant(10, dtype=tf.float64)  # Use a scalar tensor

        # demean
        
        # Assuming _build_predict is a method that can handle Tensors directly
        F_mean, F_var = self.predict_f(self.X_new)  # Use tf.Tensors directly
        N_new = tf.shape(F_mean)[0]
        cov_dim = self.likelihood.cov_dim
        self.F_mean_new = tf.reshape(F_mean, [N_new, cov_dim, -1])  # Using tf.reshape directly
        self.F_var_new = tf.reshape(F_var, [N_new, cov_dim, -1])

        nu = tf.shape(self.F_mean_new)[-1]
        F_samps = tf.random.normal([int(self.n_samples), N_new, cov_dim, nu], dtype=tf.float64) \
            * tf.sqrt(self.F_var_new) + self.F_mean_new
        log_det_cov, yt_inv_y = self.likelihood.make_gaussian_components(F_samps, self.Y_new)

        # compute the Gaussian metrics
        D_ = tf.cast(D, tf.float64)
        tf_pi = tf.constant(np.pi, dtype=tf.float64)
        self.logp_gauss_data = -0.5 * yt_inv_y
        self.logp_gauss = -0.5 * D_ * tf.math.log(2 * tf_pi) - 0.5 * log_det_cov + self.logp_gauss_data  # (S, N)
    
        if not self.likelihood.heavy_tail:
            self.logp_data = self.logp_gauss_data
            self.logp = self.logp_gauss
        else:
            dof = tf.cast(self.likelihood.dof, tf.float64)
            self.logp_data = -0.5 * (dof + D_) * tf.math.log(1.0 + yt_inv_y / dof)
            self.logp = tf.math.lgamma(0.5 * (dof + D_)) - tf.math.lgamma(0.5 * dof) - 0.5 * D_ * tf.math.log(np.pi * dof) \
                - 0.5 * log_det_cov + self.logp_data  # (S, N)

    def mcmc_predict_density(self, X_new, Y_new, n_samples=100):
        sess = self.enquire_session()
        outputs = sess.run([self.logp, self.logp_data, self.logp_gauss, self.logp_gauss_data],
                           feed_dict={self.X_new: X_new, self.Y_new: Y_new, self.n_samples: n_samples})
        log_S = np.log(n_samples)
        return tuple(map(lambda x: logsumexp(x, axis=0) - log_S, outputs))

class FullCovarianceRegression(DynamicCovarianceRegression):
    # @params_as_tensors
    def build_prior_KL(self):
        KL = super().build_prior_KL()
        if self.likelihood.approx_wishart:
            p_dist = tf.distributions.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
            q_dist = tf.distributions.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
            self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
            KL += self.KL_gamma
        return KL

    def mcmc_predict_matrix(self, X_new, n_samples):

        params = self.predict(X_new)
        mu, s2 = params['mu'], params['s2']
        scale_diag = params['scale_diag']

        N_new, D, nu = mu.shape
        F_samps = np.random.randn(n_samples, N_new, D, nu) * np.sqrt(s2) + mu  # (n_samples, N_new, D, nu)
        AF = scale_diag[:, None] * F_samps  # (n_samples, N_new, D, nu)
        affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_new, D, D)

        if self.likelihood.approx_wishart:
            sigma2inv_conc = params['sigma2inv_conc']
            sigma2inv_rate = params['sigma2inv_rate']
            sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate, size=[n_samples, D])  # (n_samples, D)

            if self.likelihood.model_inverse:
                lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps)  # (n_samples, D, D)
            else:
                lam = np.apply_along_axis(np.diag, axis=0, arr=sigma2inv_samps ** -1.0)
            affa = affa + lam[:, None, :, :]  # (n_samples, N_new, D, D)
        return affa

    def predict(self, X_new):
        mu, s2 = self.F_mean_new, self.F_var_new
        scale_diag = self.likelihood.scale_diag.numpy()  # (D,)
        params = dict(mu=mu, s2=s2, scale_diag=scale_diag)

        if self.likelihood.approx_wishart:
            sigma2inv_conc = self.likelihood.q_sigma2inv_conc.numpy()  # (D,)
            sigma2inv_rate = self.likelihood.q_sigma2inv_rate.numpy()
            params.update(dict(sigma2inv_conc=sigma2inv_conc, sigma2inv_rate=sigma2inv_rate))

        return params

class FactoredCovarianceRegression(DynamicCovarianceRegression):
    # @params_as_tensors
    def build_prior_KL(self):
        KL = super().build_prior_KL()
        p_dist = tf.distributions.Gamma(self.likelihood.p_sigma2inv_conc, rate=self.likelihood.p_sigma2inv_rate)
        q_dist = tf.distributions.Gamma(self.likelihood.q_sigma2inv_rate, rate=self.likelihood.q_sigma2inv_rate)
        self.KL_gamma = tf.reduce_sum(q_dist.kl_divergence(p_dist))
        KL += self.KL_gamma
        return KL

    def predict(self, X_new):
        """
        Compute the components needed for prediction: s2_diag, scale, F. It's more efficient to report this since scale
        is (D, K) and F is (T_test, K, K).

        In the Wishart process case, we construct the covariance matrix as:
            S = np.diag(s2_diag) + U * U^T
        where
            U = np.einsum('jk,ikl->ijl', scale, F)

        In the inverse Wishart process case, we construct the precision matrix as:
            S = np.diag(s2_diag ** -1.0) + U * U^T
        where
            U = np.einsum('jk,ikl->ijl', scale, F)

        :param X_new:
        :return:
        """

        sess = self.enquire_session()
        mu, s2 = sess.run([self.F_mean_new, self.F_var_new], feed_dict={self.X_new: X_new})  # (N_new, D, nu), (N_new, D, nu)
        scale = self.likelihood.scale.read_value(sess)  # (D, Kv)

        sigma2inv_conc = self.likelihood.q_sigma2inv_conc.read_value(sess)  # (D,)
        sigma2inv_rate = self.likelihood.q_sigma2inv_rate.read_value(sess)
        params = dict(mu=mu, s2=s2, scale=scale, sigma2inv_conc=sigma2inv_conc, sigma2inv_rate=sigma2inv_rate)
        return params


###########################################
#####  Loglikelihood helper function  #####
###########################################


def get_loglikel(model, Xt, Yt, minibatch_size=100):
    loglikel_ = 0.0
    loglikel_data_ = 0.0
    gauss_ll_ = 0.0
    gauss_ll_data_ = 0.0
    for mb in range(-(-len(Xt) // minibatch_size)):
        mb_start = mb * minibatch_size
        mb_finish = (mb + 1) * minibatch_size
        Xt_mb = Xt[mb_start:mb_finish, :]
        Yt_mb = Yt[mb_start:mb_finish, :]
        logp, logp_data, logp_gauss, logp_gauss_data = model.mcmc_predict_density(Xt_mb, Yt_mb)  # (N_new,), (N_new,)
        loglikel_ += np.sum(logp)  # simply summing over the log p(Y_n, X_n | F_n^)
        loglikel_data_ += np.sum(logp_data)
        gauss_ll_ += np.sum(logp_gauss)
        gauss_ll_data_ += np.sum(logp_gauss_data)
    return loglikel_, loglikel_data_, gauss_ll_, gauss_ll_data_


#################################################################
#####  custom GP Monitor tasks to track metrics and params  #####
#################################################################

class LoglikelTensorBoardTask(MonitorTask):
    def __init__(self, log_dir, model, Xt, Yt, summary_name):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.model = model
        self.Xt = Xt
        self.Yt = Yt
        self.summary_name = summary_name

    def run(self, step, compiled_update=None):
        with self.file_writer.as_default():
            # Calculate log likelihoods
            loglikel_, loglikel_data_, gauss_ll_, gauss_ll_data_ = get_loglikel(self.model, self.Xt, self.Yt)
            # Write summaries
            tf.summary.scalar(self.summary_name + '_full', loglikel_, step=step)
            tf.summary.scalar(self.summary_name + '_data', loglikel_data_, step=step)
            tf.summary.scalar(self.summary_name + '_gauss_full', gauss_ll_, step=step)
            tf.summary.scalar(self.summary_name + '_gauss_data', gauss_ll_data_, step=step)
            self.file_writer.flush()
