import os 
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .sim import SimRegressor, SimClassifier
from .mob import BaseMOB, BaseMOBRegressor, BaseMOBClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

EPSILON = 1e-7
__all__ = ["LIFTNetRegressor", "LIFTNetClassifier"]


class BaseLIFTNet(BaseMOB, metaclass=ABCMeta):
    """
        Base LIFTNet class for classification and regression.
     """

    @abstractmethod
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, n_split_grid=10, split_features=None,
                 knot_dist="quantile", degree=3, knot_num=5, nterms=5, reg_gamma=0.1,
                 sim_update=False, val_ratio=0.2, random_state=0):

        self.max_depth = max_depth
        self.n_split_grid = n_split_grid
        self.split_features = split_features
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        
        self.degree = degree
        self.knot_num = knot_num
        self.knot_dist = knot_dist
        self.nterms = nterms
        self.reg_gamma = reg_gamma
        self.sim_update = sim_update
        
        self.val_ratio = val_ratio
        self.random_state = random_state

    def _validate_hyperparameters(self):

        if not isinstance(self.max_depth, int):
            raise ValueError("degree must be an integer, got %s." % self.max_depth)

            if self.max_depth < 0:
                raise ValueError("degree must be >= 0, got" % self.max_depth)
   
        if self.split_features is not None:
            if not isinstance(self.split_features, list):
                raise ValueError("split_features must be an list or None, got %s." % 
                         self.split_features)
                
        if not isinstance(self.min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer, got %s." % self.min_samples_leaf)

            if self.min_samples_leaf < 0:
                raise ValueError("min_samples_leaf must be >= 0, got" % self.min_samples_leaf)

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be >= 0, got %s." % self.min_impurity_decrease)

        if self.val_ratio <= 0:
            raise ValueError("val_ratio must be > 0, got" % self.val_ratio)
        elif self.val_ratio >= 1:
            raise ValueError("val_ratio must be < 1, got %s." % self.val_ratio)

        if self.knot_dist not in ["uniform", "quantile"]:
            raise ValueError("method must be an element of [uniform, quantile], got %s." % self.knot_dist)

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)
            if self.degree < 0:
                raise ValueError("degree must be >= 0, got" % self.degree)

        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
            if self.knot_num <= 0:
                raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if isinstance(self.nterms, list):
            for val in self.nterms:
                if val < 0:
                    raise ValueError("all the elements in nterms must be >= 0, got %s." % self.nterms)
            self.nterms_list = self.nterms  
        elif (isinstance(self.nterms, float)) or (isinstance(self.nterms, int)):
            if (self.nterms < 0) or (self.nterms > 1):
                raise ValueError("nterms must be >= 0 and <=1, got %s." % self.nterms)
            self.nterms_list = [self.nterms]

        if isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if val < 0:
                    raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = self.reg_gamma  
        elif (isinstance(self.reg_gamma, float)) or (isinstance(self.reg_gamma, int)):
            if (self.reg_gamma < 0) or (self.reg_gamma > 1):
                raise ValueError("reg_gamma must be >= 0 and <=1, got %s." % self.reg_gamma)
            self.reg_gamma_list = [self.reg_gamma]

        if not isinstance(self.sim_update, bool):
            raise ValueError("sim_update must be boolean, got %s." % self.sim_update)
            
    def _first_order(self, x, y, sample_weight=None):

        """calculate the projection indice using the first order stein's identity

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
        if np.linalg.norm(zbar) > 0:
            beta = zbar / np.linalg.norm(zbar)
        else:
            beta = zbar
        return beta.reshape([-1, 1])
            
    def visualize_leaves(self, cols_per_row=3, folder="./results/", name="leaf_sim", save_png=False, save_eps=False):

        """draw the global interpretation of the fitted model
        
        Parameters
        ---------
        cols_per_row : int, optional, default=3,
            the number of sim models visualized on each row
        folder : str, optional, defalut="./results/"
            the folder of the file to be saved
        name : str, optional, default="global_plot"
            the name of the file to be saved
        save_png : bool, optional, default=False
            whether to save the figure in png form
        save_eps : bool, optional, default=False
            whether to save the figure in eps form
        """

        check_is_fitted(self, "tree")
        
        subfig_idx = 0
        max_ids = len(self.leaf_estimators_)
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        
        projection_indices = np.array([est.beta_.flatten() for est in self.leaf_estimators_]).T
        if projection_indices.shape[1] > 0:
            xlim_min = - max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
            xlim_max = max(np.abs(self.projection_indices_.min() - 0.1), np.abs(self.projection_indices_.max() + 0.1))
        
        for node_id, est in self.leaf_estimators_.items():
            inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.15, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            xgrid = np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = est.shape_fit_.decision_function(xgrid)
            ax1_main.plot(xgrid, ygrid, color="red")
            ax1_main.set_xticklabels([])
            ax1_main.set_title("Node " + str(node_id), fontsize=16)
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(est.shape_fit_.bins_[1:]) + np.array(est.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, est.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            fig.add_subplot(ax1_density)

            ax2 = fig.add_subplot(inner[:, 1])
            if len(est.beta_) <= 20:
                rects = ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
                ax2.set_yticks(np.arange(len(est.beta_)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(est.beta_.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(est.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                right = np.round(np.linspace(0, np.round(len(est.beta_) * 0.45).astype(int), 5))
                left = len(est.beta_) - 1 - right
                input_ticks = np.unique(np.hstack([left, right])).astype(int)

                rects = ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(est.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            fig.add_subplot(ax2)
            subfig_idx += 1

        plt.show()
        if max_ids > 0:
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                save_path = folder + name
                fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)


class LIFTNetRegressor(BaseLIFTNet, BaseMOBRegressor, RegressorMixin):
    
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, n_split_grid=10, split_features=None,
                 knot_dist="quantile", degree=3, knot_num=5, nterms=5, reg_gamma=0.1,
                 sim_update=False, val_ratio=0.2, random_state=0):

        super(LIFTNetRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 n_split_grid=n_split_grid,
                                 split_features=split_features,
                                 degree=degree,
                                 knot_num=knot_num,
                                 knot_dist=knot_dist,
                                 nterms=nterms,
                                 reg_gamma=reg_gamma,
                                 sim_update=sim_update,
                                 val_ratio=val_ratio,
                                 random_state=random_state)

    def build_root(self):
        
        root_clf = SimRegressor(nterms=0, reg_gamma=0, degree=self.degree,
                        knot_dist=self.knot_dist, knot_num=self.knot_num,
                        random_state=self.random_state)
        root_clf.fit(self.x, self.y)
        root_impurity = self.get_loss(self.y, root_clf.predict(self.x))
        return root_impurity

    def build_leaf(self, sample_indice):
        
        best_estimator = None
        n_samples = len(sample_indice)
        best_impurity = np.inf
        idx1, idx2 = train_test_split(sample_indice, test_size=self.val_ratio, random_state=self.random_state)
        for nterms in self.nterms_list:
            for reg_gamma in self.reg_gamma_list:
                estimator = SimRegressor(nterms=nterms, reg_gamma=reg_gamma, degree=self.degree,
                                 knot_num=self.knot_num, random_state=self.random_state)
                estimator.fit(self.x[idx1], self.y[idx1])
                current_impurity = self.get_loss(self.y[idx2], estimator.predict(self.x[idx2]))
                if current_impurity < best_impurity:
                    best_estimator = estimator
                    best_impurity = current_impurity
        best_estimator.fit(self.x[sample_indice], self.y[sample_indice])
        if self.sim_update:
            best_estimator.fit_middle_update_adam(self.x[sample_indice], self.y[sample_indice],
                  max_inner_iter=10, n_inner_iter_no_change=1,
                  batch_size=min(100, int(0.2 * n_samples)), val_ratio=self.val_ratio, stratify=False, verbose=False)
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity
    
    def node_split(self, sample_indice):
        
        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        max_deviation = 0
        best_feature = None
        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        beta_parent = self._first_order(node_x, node_y)
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < EPSILON:
                continue

            split_point = 0
            for i, _ in enumerate(sortted_indice):

                if i == (n_samples - 1):
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue
                
                if sortted_feature[i + 1] <= sortted_feature[i] + EPSILON:
                    continue

                if (i + 1 - self.min_samples_leaf) < 1 / self.n_split_grid * (split_point + 1) * (n_samples - 2 * self.min_samples_leaf):
                    continue

                split_point += 1
                left_indice = sortted_indice[:(i + 1)]
                right_indice = sortted_indice[(i + 1):]
                beta_left = self._first_order(node_x[left_indice], node_y[left_indice])
                beta_right = self._first_order(node_x[right_indice], node_y[right_indice])
                deviation = len(left_indice) * np.linalg.norm(beta_parent - beta_left) + \
                        len(right_indice) * np.linalg.norm(beta_parent - beta_right)
                if deviation > max_deviation:
                    best_position = i + 1
                    max_deviation = deviation
                    best_feature = feature_indice
                    best_threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

        if best_position is not None:
            sortted_indice = np.argsort(node_x[:, best_feature])
            best_left_indice = sample_indice[sortted_indice[:best_position]]
            best_right_indice = sample_indice[sortted_indice[best_position:]]

            left_clf = SimRegressor(nterms=0, reg_gamma=0, degree=self.degree,
                             knot_dist=self.knot_dist, knot_num=self.knot_num,
                             random_state=self.random_state)
            left_clf.fit(self.x[best_left_indice], self.y[best_left_indice])

            right_clf = SimRegressor(nterms=0, reg_gamma=0, degree=self.degree,
                             knot_dist=self.knot_dist, knot_num=self.knot_num,
                             random_state=self.random_state)
            right_clf.fit(self.x[best_right_indice], self.y[best_right_indice])

            best_left_impurity = self.get_loss(self.y[best_left_indice].ravel(), left_clf.predict(self.x[best_left_indice]))
            best_right_impurity = self.get_loss(self.y[best_right_indice].ravel(), right_clf.predict(self.x[best_right_indice]))
            best_impurity = (len(left_indice) * best_left_impurity + len(right_indice) * best_right_impurity) / n_samples
            
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
    
    
class LIFTNetClassifier(BaseLIFTNet, BaseMOBClassifier, ClassifierMixin):
    
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, n_split_grid=10, split_features=None,
                 knot_dist="quantile", degree=3, knot_num=5, nterms=5, reg_gamma=0.1,
                 sim_update=False, val_ratio=0.2, random_state=0):

        super(LIFTNetClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 n_split_grid=n_split_grid,
                                 split_features=split_features,
                                 degree=degree,
                                 knot_num=knot_num,
                                 knot_dist=knot_dist,
                                 nterms=nterms,
                                 reg_gamma=reg_gamma,
                                 sim_update=sim_update,
                                 val_ratio=val_ratio,
                                 random_state=random_state)

    def build_root(self):
        
        root_clf = SimClassifier(nterms=0, reg_gamma=0, degree=self.degree,
                         knot_dist=self.knot_dist, knot_num=self.knot_num,
                         random_state=self.random_state)
        root_clf.fit(self.x, self.y)
        root_impurity = self.get_loss(self.y, root_clf.predict_proba(self.x)[:, 1])
        return root_impurity

    def build_leaf(self, sample_indice):
        
        best_estimator = None
        n_samples = len(sample_indice)
        idx1, idx2 = train_test_split(sample_indice, test_size=self.val_ratio, random_state=self.random_state)
        if (self.y[sample_indice].std() == 0) | (self.y[idx1].std() == 0)| (self.y[idx2].std() == 0):
            best_impurity = 0
            predict_func = lambda x: np.mean(self.y[sample_indice])
        else:
            best_impurity = np.inf
            for nterms in self.nterms_list:
                for reg_gamma in self.reg_gamma_list:
                    estimator = SimClassifier(degree=self.degree,
                             nterms=nterms, reg_gamma=reg_gamma, knot_dist=self.knot_dist, knot_num=self.knot_num,
                             random_state=self.random_state)
                    estimator.fit(self.x[idx1], self.y[idx1])
                    current_impurity = self.get_loss(self.y[idx2], estimator.predict_proba(self.x[idx2])[:, 1])
                    if current_impurity < best_impurity:
                        best_estimator = estimator
                        best_impurity = current_impurity
            best_estimator.fit(self.x[sample_indice], self.y[sample_indice])
            if self.sim_update:
                best_estimator.fit_middle_update_adam(self.x[sample_indice], self.y[sample_indice],
                      max_inner_iter=10, n_inner_iter_no_change=1,
                      batch_size=min(100, int(0.2 * n_samples)), val_ratio=self.val_ratio, stratify=False, verbose=False)
            predict_func = lambda x: best_estimator.predict_proba(x)[:, 1]
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity
    
    def node_split(self, sample_indice):
        
        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        max_deviation = 0
        best_feature = None
        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        beta_parent = self._first_order(node_x, node_y)
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < EPSILON:
                continue

            split_point = 0
            for i, _ in enumerate(sortted_indice):

                if i == (n_samples - 1):
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue
                
                if sortted_feature[i + 1] <= sortted_feature[i] + EPSILON:
                    continue

                if (i + 1 - self.min_samples_leaf) < 1 / self.n_split_grid * (split_point + 1) * (n_samples - 2 * self.min_samples_leaf):
                    continue

                split_point += 1
                left_indice = sortted_indice[:(i + 1)]
                right_indice = sortted_indice[(i + 1):]
                beta_left = self._first_order(node_x[left_indice], node_y[left_indice])
                beta_right = self._first_order(node_x[right_indice], node_y[right_indice])
                deviation = len(left_indice) * np.linalg.norm(beta_parent - beta_left) + \
                        len(right_indice) * np.linalg.norm(beta_parent - beta_right)
                if deviation > max_deviation:
                    best_position = i + 1
                    max_deviation = deviation
                    best_feature = feature_indice
                    best_threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

        if best_position is not None:
            sortted_indice = np.argsort(node_x[:, best_feature])
            best_left_indice = sample_indice[sortted_indice[:best_position]]
            best_right_indice = sample_indice[sortted_indice[best_position:]]

            left_clf = SimClassifier(nterms=0, reg_gamma=0, degree=self.degree,
                             knot_dist=self.knot_dist, knot_num=self.knot_num,
                             random_state=self.random_state)
            left_clf.fit(self.x[best_left_indice], self.y[best_left_indice])

            right_clf = SimClassifier(nterms=0, reg_gamma=0, degree=self.degree,
                             knot_dist=self.knot_dist, knot_num=self.knot_num,
                             random_state=self.random_state)
            right_clf.fit(self.x[best_right_indice], self.y[best_right_indice])
            
            best_left_impurity = self.get_loss(self.y[best_left_indice].ravel(), left_clf.predict_proba(self.x[best_left_indice])[:, 1])
            best_right_impurity = self.get_loss(self.y[best_right_indice].ravel(), right_clf.predict_proba(self.x[best_right_indice])[:, 1])
            best_impurity = (len(left_indice) * best_left_impurity + len(right_indice) * best_right_impurity) / n_samples
            
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
