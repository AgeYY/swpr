import numpy as np
import tensorflow as tf
import gpflow
from gpflow import likelihoods

class MyWishart(likelihoods.Likelihood):
    def __init__(self, input_dim, D, R=1):
        super().__init__(input_dim=input_dim, latent_dim=D * D, observation_dim=D)
        self.D, self.R = D, R
        self.A_diag = gpflow.Parameter(np.ones(D), transform=gpflow.utilities.positive())

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data given a Gaussian distribution for the latent function values.
        We assume the data are conditionally independent given Fmu and Fvar.
        """
        # Fmu and Fvar are the mean and variance of the latent function values
        N, _ = Y.shape
