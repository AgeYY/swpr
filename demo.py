import os
import json

import numpy as np
import gpflow

from models import FullCovarianceRegression
# from models import FullCovarianceRegression, FactoredCovarianceRegression, LoglikelTensorBoardTask
from likelihoods import FullCovLikelihood


### Run this as a script

root_savedir = './savedir'
root_logdir = os.path.join(root_savedir, 'tf_logs')

if not os.path.exists(root_savedir):
    os.makedirs(root_savedir)

if not os.path.exists(root_logdir):
    os.makedirs(root_logdir)


#################################
#####  Make a fake dataset  #####
#################################

N = 500  # time points
D = 10
X = np.linspace(0, 5, N)[:, None]  # input time points
Y = np.random.randn(N, D) * 10
test_N = int(N * 0.1)

# holdout a test set
X_train, X_test = X[test_N:, :], X[:test_N, :]
Y_train, Y_test = Y[test_N:, :], Y[:test_N, :]

# holdout a validation set
holdout_ratio = 0.1
n_train = int(len(X_train) * (1.0 - holdout_ratio))
X_valid, X_train = X_train[n_train:, :], X_train[:n_train, :]
Y_valid, Y_train = Y_train[n_train:, :], Y_train[:n_train, :]


##########################################
#####  Build the GPflow model/graph  #####
##########################################


input_dim = 1  # dimension of the input
n_inducing = 100  # number of inducing points
n_samples = 20  # number of Monte Carlo samples
minibatch_size = 16  # minibatch size for training

factored = False  # whether or not to use a factored model
n_factors = None  # number of factors in a factored model (ignored if factored==False)
heavy_tail = False  # whether to use the heavy-tailed emission distribution
model_inverse = False  # if True, then use an inverse Wishart process; if False, use a Wishart process
approx_wishart = False  # if True, use the additive white noise model


# initilize the variational inducing points
x_min = X_train.min()
x_max = X_train.max()
# Z = np.linspace(x_min, x_max, self.n_inducing)[:, None]
Z = x_min + np.random.rand(n_inducing) * (x_max - x_min)
Z = Z[:, None]

# follow the gpflow monitor tutorial to log the optimization procedure
kern = gpflow.kernels.Matern32(1)

if not factored:
    likel = FullCovLikelihood(input_dim, D, n_samples,
                              heavy_tail=heavy_tail,
                              model_inverse=model_inverse,
                              approx_wishart=approx_wishart,
                              nu=None)

    model = FullCovarianceRegression(X_train, Y_train, kern, likel, Z, minibatch_size=minibatch_size)

    # else:
    #     likel = FactoredCovLikelihood(D, n_samples, n_factors,
    #                                   heavy_tail=heavy_tail,
    #                                   model_inverse=model_inverse,
    #                                   nu=None)

    #     model = FactoredCovarianceRegression(X_train, Y_train, kern, likel, Z, minibatch_size=minibatch_size)

# print(model.as_pandas_table())


####################################################
#####  GP Monitor tasks for tracking progress  #####
####################################################


# See GP Monitor's demo webpages for more information
monitor_lag = 10  # how often GP Monitor should display training statistics
save_lag = 100  # Don't make this too small. Saving is very I/O intensive

# create the global step parameter tracking the optimization, if using GP monitor's 'create_global_step'
# helper, this MUST be done before creating GP monitor tasks
# session = model.enquire_session()
# global_step = gpmon.create_global_step(session)

# # create the gpmonitor tasks
# print_task = gpmon.PrintTimingsTask().with_name('print') \
#     .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
#     .with_exit_condition(True)

# savedir = os.path.join(root_savedir, 'monitor-saves')
# saver_task = gpmon.CheckpointTask(savedir).with_name('saver') \
#     .with_condition(gpmon.PeriodicIterationCondition(save_lag)) \
#     .with_exit_condition(True)

# file_writer = gpmon.LogdirWriter(root_logdir, session.graph)

# model_tboard_task = gpmon.ModelToTensorBoardTask(file_writer, model).with_name('model_tboard') \
#     .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
#     .with_exit_condition(True)

# train_tboard_task = LoglikelTensorBoardTask(file_writer, model, X_train, Y_train,
#                                             summary_name='train_ll').with_name('train_tboard') \
#     .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
#     .with_exit_condition(True)

# # put the tasks together in a monitor
# monitor_tasks = [print_task, model_tboard_task, train_tboard_task, saver_task]

# # add one more if there is a validation set
# if X_valid is not None:
#     test_tboard_task = LoglikelTensorBoardTask(file_writer, model, X_valid, Y_valid,
#                                                summary_name='test_ll').with_name('test_tboard') \
#         .with_condition(gpmon.PeriodicIterationCondition(monitor_lag)) \
#         .with_exit_condition(True)
#     monitor_tasks.append(test_tboard_task)



##################################
#####  Run the optimization  #####
##################################

# Running the optimization
n_iterations = 1000
Y_mean = np.mean(Y_train, axis=0)
Yp_train = Y_train - Y_mean[None, :]

# TensorFlow's optimizer
# optimiser = tf.optimizers.Adam(learning_rate=0.01)
optimiser = gpflow.optimizers.Scipy()
# for _ in range(n_iterations):
optimiser.minimize(model.training_loss_closure((X_train, Yp_train)), model.trainable_variables)

# Prediction (assuming the model's predict method is compatible with TensorFlow 2 and GPflow 2)
preds = model.predict(X_test)

#############################
#####  Demo prediction  #####
#############################


# the format of the predictions depends on whether you're using a factored model; see the definitions in 'models'
if not factored:
    preds = model.predict(X_test)

    # look at the parameters in the dictionary 'preds':
    print("Parameters required for prediction:", preds.keys())

    # form Monte Carlo samples of the the predicted matrix like:
    mu, s2 = preds['mu'], preds['s2']  # (N_test, D, nu)
    N_test, D, nu = mu.shape
    # F_samps = np.random.randn(n_samples, N_test, D, nu) * (s2 ** 0.5) + mu  # (n_samples, N_test, D, nu)
    F_samps = mu[None, ...]
    AF = preds['scale_diag'][:, None] * F_samps  # (n_samples, N_test, D, nu)
    affa = np.matmul(AF, np.transpose(AF, [0, 1, 3, 2]))  # (n_samples, N_test, D, D)

    if not approx_wishart:
        additive_part = np.diag(np.ones(D) * 1e-5)[None, :, :]  # at least add some small jitter, like with normal GPs
    else:
        sigma2inv_conc = preds['sigma2inv_conc']  # (D,)
        sigma2inv_rate = preds['sigma2inv_rate']
        sigma2inv_samps = np.random.gamma(sigma2inv_conc, scale=1.0 / sigma2inv_rate, size=[n_samples, D])  # (n_samples, D)

        if model_inverse:
            # inverse Wishart process variants
            additive_part = np.apply_along_axis(np.diag, axis=1, arr=sigma2inv_samps)  # (n_samples, D, D)
        else:
            # Wishart process variants
            additive_part = np.apply_along_axis(np.diag, axis=1, arr=sigma2inv_samps ** -1.0)  # (n_samples, D, D)

    affa = affa + additive_part[:, None, :, :]  # shape: (n_samples, N_test, D, D)

else:
    preds = model.predict(X_test)

    # look at the parameters in the dictionary 'preds':
    print("Parameters required for prediction:", preds.keys())

    # In this case, you're probably using a factored model because you can't/shouldn't be representing the full
    # covariance matrix. You can form Monte Carlo samples of the required parameters/variables (to use in evaluating
    # a test loglikelihood, for example) as in the above example.
print(affa.shape)
print(affa[0, 0, :, :])


for key in preds.keys():
    if isinstance(preds[key], np.ndarray):
        preds[key] = preds[key].tolist()

# save predictions in json format
with open(os.path.join(root_savedir, 'preds.json'), 'w') as f:
    json.dump(preds, f)
