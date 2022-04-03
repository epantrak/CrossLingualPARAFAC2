import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np
from numpy.linalg import inv
from numpy.linalg import svd

from scipy import sparse
from scipy.sparse import eye
from scipy.sparse import diags
from scipy.sparse import csr_matrix

from Code.Utils.timer import Timer


def _parse_arguments(name, backends):
    parser = argparse.ArgumentParser(name)
    parser.add_argument("-b", "--backend", help="backend to run the test on",
                        choices=backends, default=backends[0])
    parser.add_argument("-q", "--quiet", action="store_true")
    return vars(parser.parse_args())


def __check_convergence(error_old, error, error_tol=1e-5):
    check = abs(error_old - error) < error_tol * error_old
    return not check


class Parafac2:
    n_rows = lambda self, x: x.shape[0]
    n_cols = lambda self, x: x.shape[1]
    error_fun = lambda self, x, x_hat: np.sum((x - x_hat) ** 2)
    fit_fun = lambda self, x_old, x_new, ord=2: np.sum(abs(x_old - x_new) ** ord) / np.sum(abs(x_old) ** ord)
    rec_fit_fun = lambda self, x_old, x_new, n_elements: np.sum(np.asarray(x_old - x_new).ravel() ** 2)  # / n_elements

    sum_norm_fun = lambda self, x: x / x.sum(axis=0, keepdims=1)

    def __init__(self, rank,
                 max_m_iter=100,
                 error_tol=1e-5,
                 approx_fit_error=None,
                 tb_writer=None):

        self.rank = rank
        self.max_m_iter = max_m_iter
        self.error_tol = error_tol
        self.approx_fit_error = approx_fit_error
        self.tb_writer = tb_writer  # SummaryWriter(log_folder)
        self.time = Timer()
        self._time = Timer()

        self._n_samples, self._n_languages, self._n_terms = 0, 0, []

        self.is_initialized = False

    def get_U(self):
        return self.U

    def get_H(self):
        return self.H

    def get_S(self):
        return self.S

    def get_W(self):
        return self.W

    def __initialize_decomposition(self):

        self.__init_U()  # projection matrices
        self.__init_H()  # factors
        self.__init_W()  # factors
        self.__init_S()  # factors

    def __init_U(self):

        self.U = [np.random.uniform(0, 1, size=(self._n_terms[k], self.rank)) for k in range(self._n_languages)]

    def __init_H(self):

        self.H = np.eye(self.rank, self.rank)

    def __init_W(self):

        self.W = np.random.uniform(0, 1, size=(self._n_samples, self.rank))

    def __init_S(self):

        self.S = [eye(self.rank) for i in range(self._n_languages)]

    def __partial_fit_U(self, X):

        for k in range(self._n_languages):
            csr_matrix.dot(self.H, self.S[k])
            u, s, vt = svd(csr_matrix.dot(csr_matrix.dot(self.H, self.S[k]).dot(self.W.T), X[k].T), full_matrices=False)
            self.U[k] = (u.dot(vt)).T

        return self

    def __partial_fit_H(self, X):

        lhs = np.sum(
            [csr_matrix.dot(csr_matrix.dot(self.U[k].T, X[k]).dot(self.W), self.S[k]) for k in
             range(self._n_languages)],
            axis=0)
        rhs = np.sum([self.S[k] * (self.W.T.dot(self.W)) * self.S[k] for k in range(self._n_languages)], axis=0)

        self.H = lhs.dot(inv(rhs))

        return self

    def __partial_fit_S(self, X):

        self.S = [diags(np.dot(inv(self.W.T.dot(self.W) * self.H.T.dot(self.H)),
                               np.diag(csr_matrix.dot(self.H.T.dot(self.U[k].T), X[k]).dot(self.W)))) for k in
                  range(self._n_languages)]
        return self

    def __partial_fit_W(self, X):

        lhs = np.sum([(csr_matrix.dot(X[k].T, self.U[k]).dot(self.H)) * self.S[k] for k in range(self._n_languages)],
                     axis=0)
        rhs = np.sum([self.S[k] * (self.H.T.dot(self.H)) * self.S[k] for k in range(self._n_languages)], axis=0)
        self.W = lhs.dot(inv(rhs))

        return self

    def __partial_fit_Gamma(self):

        raise ValueError('TODO: partial_fit_Gamma')

    def __rec_tensor(self, U, H, S, W):

        X_l_rec = [U[k].dot(H).dot(csr_matrix.dot(S[k], W.T)) for k in range(self._n_languages)]
        return X_l_rec

    def __fit_error(self, X):

        self._time.start()

        if self.approx_fit_error:
            if not float(self.approx_fit_error).is_integer():
                self.approx_fit_error = int(self.approx_fit_error * self._n_samples)
            sample_range = np.random.randint(1, size=self.approx_fit_error)
        else:
            sample_range = range(self._n_samples)

        errors = []
        for k in range(self._n_languages):
            UHS = self.U[k].dot(self.H * self.S[k])
            _error = 0
            for n in tqdm(sample_range, desc='lang {} fit_error'.format(k)):
                x_hat_n = UHS.dot(self.W[n].T)
                _error += np.linalg.norm(X[k][:, n] - x_hat_n[:, None], ord=2) ** 2
            errors.append(_error)

        mean_error = np.sum(errors)

        self._time.stop(tag='fit_error', verbose=True)

        return mean_error, errors

    def _check_init(self):

        if not self.is_initialized:
            self.__initialize_decomposition()
            self.is_initialized = True

    def partial_fit(self, X):

        self._n_languages, self._n_samples = len(X), X[0].shape[1]
        self._n_terms = [X_l.shape[0] for X_l in X]

        self._check_init()

        self.__partial_fit_U(X=X)
        self.__partial_fit_W(X=X)
        self.__partial_fit_H(X=X)
        self.__partial_fit_S(X=X)

        mean_loss, loss = self.__fit_error(X=X)

        return mean_loss, loss
