import numpy as np
from copy import deepcopy
from matplotlib import gridspec
import matplotlib.pyplot as plt

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor

from abc import ABCMeta, abstractmethod
from .smspline import SMSplineRegressor, SMSplineClassifier


__all__ = ["SimRegressor", "SimClassifier"]


class BaseSim(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, reg_lambda=0, reg_gamma=1e-5, knot_num=5, degree=3, random_state=0):

        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.degree = degree

        self.random_state = random_state

    def _first_order_thres(self, x, y):

        """calculate the projection indice using the first order stein's identity subject to hard thresholding

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        Returns
        -------
        np.array of shape (n_features, 1)
            the normalized projection inidce
        """

        self.mu = np.average(x, axis=0)
        self.cov = np.cov(x.T)
        self.inv_cov = np.linalg.pinv(self.cov, 1e-7)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0)
        zbar[np.abs(zbar * x.std(0)) < self.reg_lambda * np.max(np.abs(zbar * x.std(0)))] = 0
        if np.linalg.norm(zbar) > 0:
            beta = zbar / np.linalg.norm(zbar)
        else:
            beta = zbar
        return beta.reshape([-1, 1])

    def fit(self, x, y):

        """fit the Sim model

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        Returns
        -------
        object
            self : Estimator instance.
        """

        np.random.seed(self.random_state)
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        self.beta_ = self._first_order_thres(x, y)

        if len(self.beta_[np.abs(self.beta_) > 0]) > 0:
            if (self.beta_[np.argmax(np.abs(self.beta_))] < 0):
                self.beta_ = - self.beta_
        xb = np.dot(x, self.beta_)
        self._estimate_shape(xb, y, np.min(xb), np.max(xb))
        return self

    def decision_function(self, x):

        """output f(beta^T x) for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(beta^T x)
        """

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")
        xb = np.dot(x, self.beta_)
        pred = self.shape_fit_.decision_function(xb)
        return pred

    def visualize(self):

        """draw the fitted projection indices and ridge function
        """

        check_is_fitted(self, "beta_")
        check_is_fitted(self, "shape_fit_")

        xlim_min = - max(np.abs(self.beta_.min() - 0.1), np.abs(self.beta_.max() + 0.1))
        xlim_max = max(np.abs(self.beta_.min() - 0.1), np.abs(self.beta_.max() + 0.1))

        fig = plt.figure(figsize=(12, 4))
        outer = gridspec.GridSpec(1, 2, wspace=0.15)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = plt.Subplot(fig, inner[0])
        xgrid = np.linspace(self.shape_fit_.xmin, self.shape_fit_.xmax, 100).reshape([-1, 1])
        ygrid = self.shape_fit_.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid)
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1_main)

        ax1_density = plt.Subplot(fig, inner[1])
        xint = ((np.array(self.shape_fit_.bins_[1:]) + np.array(self.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, self.shape_fit_.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.autoscale()
        fig.add_subplot(ax1_density)

        ax2 = plt.Subplot(fig, outer[1])
        if len(self.beta_) <= 50:
            ax2.barh(np.arange(len(self.beta_)), [beta for beta in self.beta_.ravel()][::-1])
            ax2.set_yticks(np.arange(len(self.beta_)))
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(self.beta_.ravel()))][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(self.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
        else:
            right = np.round(np.linspace(0, np.round(len(self.beta_) * 0.45).astype(int), 5))
            left = len(self.beta_) - 1 - right
            input_ticks = np.unique(np.hstack([left, right])).astype(int)

            ax2.barh(np.arange(len(self.beta_)), [beta for beta in self.beta_.ravel()][::-1])
            ax2.set_yticks(input_ticks)
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(self.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
        ax2.set_title("Projection Indice", fontsize=12)
        fig.add_subplot(ax2)
        plt.show()


class SimRegressor(BaseSim, RegressorMixin):

    """
    Sim regression.

    Parameters
    ----------
    reg_lambda : float, optional. default=0.1
        Sparsity penalty strength

    reg_gamma : float or list of float, optional. default=0.1
        Roughness penalty strength of the spline algorithm

    degree : int, optional. default=3
        The order of the spline. Possible values include 1 and 3.

    knot_num : int, optional. default=10
        Number of knots

    random_state : int, optional. default=0
        Random seed
    """

    def __init__(self, reg_lambda=0, reg_gamma=1e-5, knot_num=5, degree=3, random_state=0):

        super(SimRegressor, self).__init__(reg_lambda=reg_lambda,
                                reg_gamma=reg_gamma,
                                knot_num=knot_num,
                                degree=degree,
                                random_state=random_state)

    def _validate_input(self, x, y):

        """method to validate data

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def _estimate_shape(self, x, y, xmin, xmax):

        """estimate the ridge function

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        xmin : float
            the minimum value of beta ^ x
        xmax : float
            the maximum value of beta ^ x
        """

        self.shape_fit_ = SMSplineRegressor(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                                xmin=xmin, xmax=xmax, degree=self.degree)
        self.shape_fit_.fit(x, y)

    def predict(self, x):

        """output f(beta^T x) for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(beta^T x)
        """
        pred = self.decision_function(x)
        return pred


class SimClassifier(BaseSim, ClassifierMixin):

    """
    Sim classification.

    Parameters
    ----------
    reg_lambda : float, optional. default=0.1
        Sparsity penalty strength

    reg_gamma : float or list of float, optional. default=0.1
        Roughness penalty strength of the spline algorithm

    degree : int, optional. default=3
        The order of the spline

    knot_num : int, optional. default=10
        Number of knots

    random_state : int, optional. default=0
        Random seed
    """

    def __init__(self, reg_lambda=0, reg_gamma=1e-5, knot_num=5, degree=3, random_state=0):

        super(SimClassifier, self).__init__(reg_lambda=reg_lambda,
                                reg_gamma=reg_gamma,
                                knot_num=knot_num,
                                degree=degree,
                                random_state=random_state)

    def _validate_input(self, x, y):

        """method to validate data

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()

    def _estimate_shape(self, x, y, xmin, xmax):

        """estimate the ridge function

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        xmin : float
            the minimum value of beta ^ x
        xmax : float
            the maximum value of beta ^ x
        """

        self.shape_fit_ = SMSplineClassifier(knot_num=self.knot_num, reg_gamma=self.reg_gamma,
                                xmin=xmin, xmax=xmax, degree=self.degree)
        self.shape_fit_.fit(x, y)

    def predict_proba(self, x):

        """output probability prediction for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples, 2)
            containing probability prediction
        """

        pred = self.decision_function(x)
        pred_proba = softmax(np.vstack([-pred, pred]).T / 2, copy=False)
        return pred_proba

    def predict(self, x):

        """output binary prediction for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing binary prediction
        """

        pred_proba = self.predict_proba(x)[:, 1]
        return self._label_binarizer.inverse_transform(pred_proba)
