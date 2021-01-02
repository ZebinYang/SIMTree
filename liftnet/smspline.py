import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor

import rpy2
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

try:
    bigsplines = importr("bigsplines")
except:
    utils = importr("utils")
    utils.install_packages("bigsplines")
    bigsplines = importr("bigsplines")

EPSILON = 1e-7

__all__ = ["SMSplineRegressor", "SMSplineClassifier"]


class BaseSMSpline(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, knot_num=5, degree=3, reg_gamma=1e-5, xmin=-1, xmax=1):

        self.knot_num = knot_num
        self.degree = degree
        self.reg_gamma = reg_gamma if isinstance(reg_gamma, list) else [reg_gamma]
        self.xmin = xmin
        self.xmax = xmax

    def _estimate_density(self, x):

        """method to estimate the density of input data

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        """

        self.density_, self.bins_ = np.histogram(x, bins=10, density=True)

    def diff(self, x, order=1):

        """method to calculate derivatives of the fitted adaptive spline to the input

        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        order : int
            order of derivative
        """
        if order > self.degree:
            raise Exception("order should not be greater than degree")
        if isinstance(self.sm_, (np.ndarray, np.int, int, np.floating, float)):
            derivative = np.zeros((x.shape[0], 1))
        elif "modelspec" in self.sm_.names:
            modelspec = self.sm_[int(np.where(self.sm_.names == "modelspec")[0][0])]
            knots = np.array(modelspec[0])
            coefs = np.array(modelspec[11]).reshape(-1, 1)
            basis = bigsplines.ssBasis((x - self.xmin) / (self.xmax - self.xmin), knots, m=1 if self.degree==1 else 2, d=order,
                               xmin=self.xmin, xmax=self.xmax, periodic=False, intercept=True)
            derivative = np.dot(basis[0], coefs).ravel()
        else:
            knots = np.array(self.sm_[12])
            coefs = np.array(self.sm_[15]).reshape(-1, 1)
            basis = bigsplines.ssBasis((x - self.xmin) / (self.xmax - self.xmin), knots, m=1 if self.degree==1 else 2, d=order,
                               xmin=0, xmax=1, periodic=False, intercept=True)
            derivative = np.dot(basis[0], coefs).ravel()
        return derivative

    def visualize(self):

        """draw the fitted shape function
        """

        check_is_fitted(self, "sm_")

        fig = plt.figure(figsize=(6, 4))
        inner = gridspec.GridSpec(2, 1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = plt.Subplot(fig, inner[0])
        xgrid = np.linspace(self.xmin, self.xmax, 100).reshape([-1, 1])
        ygrid = self.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid)
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Shape Function", fontsize=12)
        fig.add_subplot(ax1_main)

        ax1_density = plt.Subplot(fig, inner[1])
        xint = ((np.array(self.bins_[1:]) + np.array(self.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, self.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.autoscale()
        fig.add_subplot(ax1_density)
        plt.show()

    def decision_function(self, x):

        """output f(x) for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x)
        """

        check_is_fitted(self, "sm_")
        x = x.copy()
        x[x < self.xmin] = self.xmin
        x[x > self.xmax] = self.xmax
        if isinstance(self.sm_, (np.ndarray, np.int, int, np.floating, float)):
            pred = self.sm_ * np.ones(x.shape[0])
        else:
            if is_classifier(self):
                pred = bigsplines.predict_bigssg(self.sm_, ro.r("data.frame")(x=x))[1]
            if is_regressor(self):
                pred = bigsplines.predict_bigspline(self.sm_, ro.r("data.frame")(x=x))
        return pred


class SMSplineRegressor(BaseSMSpline, RegressorMixin):

    """Base class for Smoothing Spline regression.

    Details:
    1. This is an API for the well-known R package `bigsplines`, and we call the function bigssa through rpy2 interface.
    2. During prediction, the data which is outside of the given `xmin` and `xmax` will be clipped to the boundary.

    Parameters
    ----------
    knot_num : int, optional. default=5
           the number of knots

    degree : int, optional. default=3
          the order of the spline, possible values include 1 and 3

    reg_gamma : float or list of float, optional. default=0.1
            the roughness penalty strength of the spline algorithm, range from 0 to 1.

    xmin : float, optional. default=-1
        the min boundary of the input

    xmax : float, optional. default=1
        the max boundary of the input
    """

    def __init__(self, knot_num=5, degree=3, reg_gamma=1e-5, xmin=-1, xmax=1):

        super(SMSplineRegressor, self).__init__(knot_num=knot_num,
                                  degree=degree,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax)

    def _validate_input(self, x, y):

        """method to validate data

        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """

        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def get_loss(self, label, pred):

        """method to calculate the cross entropy loss

        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        Returns
        -------
        float
            the cross entropy loss
        """

        loss = np.average((label - pred) ** 2, axis=0)
        return loss

    def fit(self, x, y):

        """fit the smoothing spline

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

        x, y = self._validate_input(x, y)
        self._estimate_density(x)

        unique_num = len(np.unique(x.round(decimals=6)))
        if unique_num <= 1:
            self.sm_ = np.mean(y)
        else:
            kwargs = {"x": x.ravel(),
                   "y": y.ravel(),
                   "nknots": self.knot_num,
                   "type": "lin" if self.degree==1 else "cub",
                   "lambdas": ro.r("c")(np.array(self.reg_gamma)),
                   "rparm": 0.01}
            self.sm_ = bigsplines.bigspline(**kwargs)
        return self

    def predict(self, x):

        """output f(x) for given samples

        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        Returns
        -------
        np.array of shape (n_samples,)
            containing f(x)
        """

        pred = self.decision_function(x)
        return pred


class SMSplineClassifier(BaseSMSpline, ClassifierMixin):

    """Base class for Smoothing Spline classification.

    Details:
    1. This is an API for the well-known R package `bigsplines`, and we call the function bigssg through rpy2 interface.
    2. During prediction, the data which is outside of the given `xmin` and `xmax` will be clipped to the boundary.
    3. reg_gamma will be increased if the current value is too small

    Parameters
    ----------
    knot_num : int, optional. default=5
           the number of knots

    degree : int, optional. default=3
          the order of the spline, possible values include 1 and 3

    reg_gamma : float or list of float, optional. default=0.1
            the roughness penalty strength of the spline algorithm, range from 0 to 1.

    xmin : float, optional. default=-1
        the min boundary of the input

    xmax : float, optional. default=1
        the max boundary of the input
    """

    def __init__(self, knot_num=5, degree=3, reg_gamma=1e-5, xmin=-1, xmax=1):

        super(SMSplineClassifier, self).__init__(knot_num=knot_num,
                                  degree=degree,
                                  reg_gamma=reg_gamma,
                                  xmin=xmin,
                                  xmax=xmax)

    def get_loss(self, label, pred):

        """method to calculate the cross entropy loss

        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        Returns
        -------
        float
            the cross entropy loss
        """

        with np.errstate(divide="ignore", over="ignore"):
            pred = np.clip(pred, EPSILON, 1. - EPSILON)
            loss = - np.average(label * np.log(pred) + (1 - label) * np.log(1 - pred), axis=0)
        return loss

    def _validate_input(self, x, y):

        """method to validate data

        Parameters
        ---------
        x : array-like of shape (n_samples, 1)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing the output dataset
        """
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()

    def fit(self, x, y):

        """fit the smoothing spline

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

        x, y = self._validate_input(x, y)
        self._estimate_density(x)

        unique_num = len(np.unique(x.round(decimals=6)))
        if unique_num <= 1:
            p = np.clip(np.mean(y), EPSILON, 1. - EPSILON)
            self.sm_ = np.log(p / (1 - p))
        else:
            exit = False
            while not exit:
                try:
                    kwargs = {"formula": Formula('y ~ x'),
                           "family": "binomial",
                           "nknots": self.knot_num, 
                           "lambdas": ro.r("c")(np.array(self.reg_gamma)),
                           "rparm": 0.01,
                           "type": "lin" if self.degree==1 else "cub",
                           "data": pd.DataFrame({"x": x.ravel(), "y": y.ravel()})}
                    self.sm_ = bigsplines.bigssg(**kwargs)
                    exit = True
                except rpy2.rinterface_lib.embedded.RRuntimeError:
                    if isinstance(self.reg_gamma, list):
                        self.reg_gamma = [v * 10 for v in self.reg_gamma]
                    else:
                        self.reg_gamma = self.reg_gamma * 10
        return self

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
