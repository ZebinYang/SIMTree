import scipy
import numpy as np
from copy import deepcopy
from matplotlib import gridspec
import matplotlib.pyplot as plt

from sklearn.utils.extmath import softmax
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor

from abc import ABCMeta, abstractmethod

from .smspline import SMSplineRegressor, SMSplineClassifier


__all__ = ["SimRegressor", "SimClassifier"]


class BaseSim(BaseEstimator, metaclass=ABCMeta):

    @abstractmethod
    def __init__(self, nterms=5, reg_gamma=0.1, knot_num=10, knot_dist="quantile", degree=3, random_state=0):

        self.nterms = nterms
        self.reg_gamma = reg_gamma
        self.knot_num = knot_num
        self.knot_dist = knot_dist
        self.degree = degree
        
        self.random_state = random_state

    def _validate_hyperparameters(self):
        
        """method to validate model parameters
        """

        if isinstance(self.nterms, int):
            if self.nterms <= 0:
                raise ValueError("nterms must be >= 1, got %s." % self.nterms)
        else:
            raise ValueError("Invalid nterms.")

        if (isinstance(self.reg_gamma, float)) or (isinstance(self.reg_gamma, int)):
            if (self.reg_gamma < 0) or (self.reg_gamma > 1):
                raise ValueError("reg_gamma must be >= 0 and <=1, got %s." % self.reg_gamma)
        else:
            raise ValueError("Invalid reg_gamma.")

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)
        elif self.degree < 0:
            raise ValueError("degree must be >= 0, got %s." % self.degree)
        
        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
        elif self.knot_num <= 0:
            raise ValueError("knot_num must be > 0, got %s." % self.knot_num)

        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("method must be an element of [uniform, quantile], got %s." % self.knot_dist)

    def _validate_sample_weight(self, n_samples, sample_weight):
        
        """method to validate sample weight 
        
        Parameters
        ---------
        n_samples : int
            the number of samples
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        """

        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight.ravel() / np.sum(sample_weight)
        return sample_weight

    def _first_order_thres(self, x, y, sample_weight=None):

        """calculate the projection indice using the first order stein's identity subject to hard thresholding

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        np.array of shape (n_features, 1)
            the normalized projection inidce
        """
        
        self.mu = np.average(x, axis=0, weights=sample_weight) 
        self.cov = np.cov(x.T, aweights=sample_weight)
        self.inv_cov = np.linalg.pinv(self.cov)
        s1 = np.dot(self.inv_cov, (x - self.mu).T).T
        zbar = np.average(y.reshape(-1, 1) * s1, axis=0, weights=sample_weight)
        inactive_index = np.argsort(np.abs(zbar))[::-1][self.nterms:]
        zbar[inactive_index] = 0
        if np.linalg.norm(zbar) > 0:
            beta = zbar / np.linalg.norm(zbar)
        else:
            beta = zbar
        return beta.reshape([-1, 1])
    
    def fit(self, x, y, sample_weight=None):

        """fit the Sim model

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        object 
            self : Estimator instance.
        """

        np.random.seed(self.random_state)
        
        self._validate_hyperparameters()
        x, y = self._validate_input(x, y)
        n_samples, n_features = x.shape
        if sample_weight is None:
            sample_weight = np.ones(n_samples) / n_samples
        else:
            sample_weight = sample_weight / np.sum(sample_weight)
        
        self.beta_ = self._first_order_thres(x, y, sample_weight)
        
        if len(self.beta_[np.abs(self.beta_) > 0]) > 0:
            if (self.beta_[np.abs(self.beta_) > 0][0] < 0):
                self.beta_ = - self.beta_
        xb = np.dot(x, self.beta_)
        self._estimate_shape(xb, y, np.min(xb), np.max(xb), sample_weight)
        return self
    
    def fit_middle_update_adam(self, x, y, sample_weight=None, val_ratio=0.2, tol=0.0001,
                      max_middle_iter=3, n_middle_iter_no_change=3, max_inner_iter=100, n_inner_iter_no_change=5,
                      batch_size=100, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, stratify=True, verbose=False):

        """fine tune the fitted Sim model using middle update method (adam)

        Parameters
        ---------
        x : array-like of shape (n_samples, n_features)
            containing the input dataset
        y : array-like of shape (n_samples,)
            containing target values
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        val_ratio : float, optional, default=0.2
            the split ratio for validation set
        tol : float, optional, default=0.0001
            the tolerance for early stopping
        max_middle_iter : int, optional, default=3
            the maximal number of middle iteration
        n_middle_iter_no_change : int, optional, default=3
            the tolerance of non-improving middle iterations
        max_inner_iter : int, optional, default=100
            the maximal number of inner iteration (epoch) for "adam" optimizer
        n_inner_iter_no_change : int, optional, default=5
            the tolerance of non-improving inner iteration for adam optimizer
        batch_size : int, optional, default=100
            the batch_size for adam optimizer
        learning_rate : float, optional, default=1e-4
            the learning rate for adam optimizer
        beta_1 : float, optional, default=0.9
            the beta_1 parameter for adam optimizer
        beta_2 : float, optional, default=0.999
            the beta_1 parameter for adam optimizer
        stratify : bool, optional, default=True
            whether to stratify the target variable when splitting the validation set
        verbose : bool, optional, default=False
            whether to show the training history
        """
            
        x, y = self._validate_input(x, y)
        n_samples = x.shape[0]
        batch_size = min(batch_size, n_samples)
        sample_weight = self._validate_sample_weight(n_samples, sample_weight)

        if is_regressor(self):
            idx1, idx2 = train_test_split(np.arange(n_samples), test_size=val_ratio,
                                          random_state=self.random_state)
        elif is_classifier(self):
            if stratify:
                idx1, idx2 = train_test_split(np.arange(n_samples),test_size=val_ratio, stratify=y, random_state=self.random_state)
            else:
                idx1, idx2 = train_test_split(np.arange(n_samples),test_size=val_ratio, random_state=self.random_state)
            tr_x, tr_y, val_x, val_y = x[idx1], y[idx1], x[idx2], y[idx2]
        
        tr_x, tr_y, val_x, val_y = x[idx1], y[idx1], x[idx2], y[idx2]
        val_xb = np.dot(val_x, self.beta_)
        if is_regressor(self):
            val_pred = self.shape_fit_.predict(val_xb)
            val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
        elif is_classifier(self):
            val_pred = self.shape_fit_.predict_proba(val_xb)[:, 1]
            val_loss = self.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])

        self_copy = deepcopy(self)
        no_middle_iter_change = 0
        val_loss_middle_iter_best = val_loss
        for middle_iter in range(max_middle_iter):

            m_t = 0 # moving average of the gradient
            v_t = 0 # moving average of the gradient square
            num_updates = 0
            no_inner_iter_change = 0
            theta_0 = self_copy.beta_ 
            train_size = tr_x.shape[0]
            val_loss_inner_iter_best = np.inf
            for inner_iter in range(max_inner_iter):

                shuffle_index = np.arange(tr_x.shape[0])
                np.random.shuffle(shuffle_index)
                tr_x = tr_x[shuffle_index]
                tr_y = tr_y[shuffle_index]

                for iterations in range(train_size // batch_size):

                    num_updates += 1
                    offset = (iterations * batch_size) % train_size
                    batch_xx = tr_x[offset:(offset + batch_size), :]
                    batch_yy = tr_y[offset:(offset + batch_size)]
                    batch_sample_weight = sample_weight[idx1][offset:(offset + batch_size)]

                    xb = np.dot(batch_xx, theta_0)
                    if is_regressor(self_copy):
                        r = batch_yy - self_copy.shape_fit_.predict(xb)
                    elif is_classifier(self_copy):
                        r = batch_yy - self_copy.shape_fit_.predict_proba(xb)[:, 1]
                    
                    # gradient
                    dfxb = self_copy.shape_fit_.diff(xb, order=1)
                    g_t = np.average((- dfxb * r).reshape(-1, 1) * batch_xx, axis=0,
                                weights=batch_sample_weight).reshape(-1, 1)

                    # update the moving average 
                    m_t = beta_1 * m_t + (1 - beta_1) * g_t
                    v_t = beta_2 * v_t + (1 - beta_2) * (g_t * g_t)
                    # calculates the bias-corrected estimates
                    m_cap = m_t / (1 - (beta_1 ** (num_updates)))  
                    v_cap = v_t / (1 - (beta_2 ** (num_updates)))
                    # updates the parameters
                    theta_0 = theta_0 - (learning_rate * m_cap) / (np.sqrt(v_cap) + 1e-8)

                # validation loss
                val_xb = np.dot(val_x, theta_0)
                if is_regressor(self_copy):
                    val_pred = self_copy.shape_fit_.predict(val_xb)
                    val_loss = self_copy.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
                elif is_classifier(self_copy):
                    val_pred = self_copy.shape_fit_.predict_proba(val_xb)[:, 1]
                    val_loss = self_copy.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
                if verbose:
                    print("Middle iter:", middle_iter + 1, "Inner iter:", inner_iter + 1, "with validation loss:", np.round(val_loss, 5))
                # stop criterion
                if val_loss > val_loss_inner_iter_best - tol:
                    no_inner_iter_change += 1
                else:
                    no_inner_iter_change = 0
                if val_loss < val_loss_inner_iter_best:
                    val_loss_inner_iter_best = val_loss
                
                if no_inner_iter_change >= n_inner_iter_no_change:
                    break
  
            ## thresholding and normalization
            inactive_index = np.argsort(np.abs(theta_0))[::-1][self.nterms:]
            theta_0[inactive_index] = 0
            if np.linalg.norm(theta_0) > 0:
                theta_0 = theta_0 / np.linalg.norm(theta_0)
                if (theta_0[np.abs(theta_0) > 0][0] < 0):
                    theta_0 = - theta_0

            # ridge update
            self_copy.beta_ = theta_0
            tr_xb = np.dot(tr_x, self_copy.beta_)
            self_copy._estimate_shape(tr_xb, tr_y, np.min(tr_xb), np.max(tr_xb), sample_weight[idx1])
            
            val_xb = np.dot(val_x, self_copy.beta_)
            if is_regressor(self_copy):
                val_pred = self_copy.shape_fit_.predict(val_xb)
                val_loss = self_copy.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])
            elif is_classifier(self_copy):
                val_pred = self_copy.shape_fit_.predict_proba(val_xb)[:, 1]
                val_loss = self_copy.shape_fit_.get_loss(val_y, val_pred, sample_weight[idx2])

            if val_loss > val_loss_middle_iter_best - tol:
                no_middle_iter_change += 1
            else:
                no_middle_iter_change = 0
            if val_loss < val_loss_middle_iter_best:
                self.beta_ = self_copy.beta_
                self.shape_fit_ = self_copy.shape_fit_
                val_loss_middle_iter_best = val_loss
            if no_middle_iter_change >= n_middle_iter_no_change:
                break
                
        self = deepcopy(self_copy)

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
        if len(self.beta_) <= 20:
            rects = ax2.barh(np.arange(len(self.beta_)), [beta for beta in self.beta_.ravel()][::-1])
            ax2.set_yticks(np.arange(len(self.beta_)))
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(self.beta_.ravel()))][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(self.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
        else:
            right = np.round(np.linspace(0, np.round(len(self.beta_) * 0.45).astype(int), 5))
            left = len(self.beta_) - 1 - right
            input_ticks = np.unique(np.hstack([left, right])).astype(int)

            rects = ax2.barh(np.arange(len(self.beta_)), [beta for beta in self.beta_.ravel()][::-1])
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
    knot_dist : str, optional. default="quantile"
        Distribution of knots
      
        "uniform": uniformly over the domain

        "quantile": uniform quantiles of the given input data (not available when spline="p_spline" or "mono_p_spline")

    nterms : int, optional. default=5
        Maximum number of selected projection indices.

    reg_gamma : float, optional. default=0.1
        Roughness penalty strength of the spline algorithm
    
    degree : int, optional. default=3
        The order of the spline.
        
        For spline="smoothing_spline_bigsplines", possible values include 1 and 3.
    
        For spline="smoothing_spline_mgcv", possible values include 3, 4, ....
    
    knot_num : int, optional. default=10
        Number of knots
    
    random_state : int, optional. default=0
        Random seed
    """

    def __init__(self, nterms=5, reg_gamma=0.1, knot_num=10, knot_dist="quantile", degree=3, random_state=0):

        super(SimRegressor, self).__init__(nterms=nterms,
                                reg_gamma=reg_gamma,
                                knot_num=knot_num,
                                knot_dist=knot_dist,
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

    def _estimate_shape(self, x, y, xmin, xmax, sample_weight=None):
       
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
        sample_weight : array-like of shape (n_samples,), optional, default=None
            containing sample weights
        """

        self.shape_fit_ = SMSplineRegressor(knot_num=self.knot_num, knot_dist=self.knot_dist, reg_gamma=self.reg_gamma,
                                xmin=xmin, xmax=xmax, degree=self.degree)
        self.shape_fit_.fit(x, y, sample_weight)

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
    knot_dist : str, optional. default="quantile"
        Distribution of knots
      
        "uniform": uniformly over the domain

        "quantile": uniform quantiles of the given input data (not available when spline="p_spline" or "mono_p_spline")

    nterms : int, optional. default=5
        Maximum number of selected projection indices.

    reg_gamma : float, optional. default=0.1
        Roughness penalty strength of the spline algorithm
        
    degree : int, optional. default=3
        The order of the spline.

    knot_num : int, optional. default=10
        Number of knots

    random_state : int, optional. default=0
        Random seed
    """

    def __init__(self, nterms=5, reg_gamma=0.1, knot_num=10, knot_dist="quantile", degree=3, random_state=0):

        super(SimClassifier, self).__init__(nterms=nterms,
                                reg_gamma=reg_gamma,
                                knot_num=knot_num,
                                knot_dist=knot_dist,
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

    def _estimate_shape(self, x, y, xmin, xmax, sample_weight=None):

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
        sample_weight : array-like of shape (n_samples,), optional, default=None
            containing sample weights
        """

        self.shape_fit_ = SMSplineClassifier(knot_num=self.knot_num, knot_dist=self.knot_dist, reg_gamma=self.reg_gamma,
                                xmin=xmin, xmax=xmax, degree=self.degree)
        self.shape_fit_.fit(x, y, sample_weight)

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
