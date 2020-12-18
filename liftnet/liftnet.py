import os
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from abc import ABCMeta, abstractmethod

from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.base import RegressorMixin, ClassifierMixin, is_regressor, is_classifier

from .sim import SimRegressor, SimClassifier
from .mobtree import BaseMoBTree, BaseMoBTreeRegressor, BaseMoBTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

__all__ = ["LIFTNetRegressor", "LIFTNetClassifier"]


class BaseLIFTNet(BaseMoBTree, metaclass=ABCMeta):
    """
        Base LIFTNet class for classification and regression.
     """

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 n_split_grid=10, split_features=None, n_feature_search=5,
                 degree=3, knot_num=5, reg_lambda=0.001, reg_gamma=0.000001, random_state=0):

        super(BaseLIFTNet, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 random_state=random_state)
        self.degree = degree
        self.knot_num = knot_num
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.n_split_grid = n_split_grid
        self.n_feature_search = n_feature_search

    def _validate_hyperparameters(self):

        if not isinstance(self.max_depth, int):
            raise ValueError("degree must be an integer, got %s." % self.max_depth)
            if self.max_depth < 0:
                raise ValueError("degree must be >= 0, got %s." % self.max_depth)

        if self.split_features is not None:
            if not isinstance(self.split_features, list):
                raise ValueError("split_features must be an list or None, got %s." % self.split_features)

        if not isinstance(self.n_feature_search, int):
            raise ValueError("n_feature_search must be an integer, got %s." % self.n_feature_search)
            if self.n_feature_search <= 0:
                raise ValueError("n_feature_search must be > 0, got %s." % self.n_feature_search)
                
        if not isinstance(self.min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer, got %s." % self.min_samples_leaf)
            if self.min_samples_leaf < 0:
                raise ValueError("min_samples_leaf must be >= 0, got %s." % self.min_samples_leaf)

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be >= 0, got %s." % self.min_impurity_decrease)

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)
            if self.degree < 0:
                raise ValueError("degree must be >= 0, got %s." % self.degree)

        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
            if self.knot_num <= 0:
                raise ValueError("knot_num must be > 0, got %s." % self.knot_num)

        if isinstance(self.reg_lambda, list):
            for val in self.reg_lambda:
                if val < 0:
                    raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_lambda)
            self.reg_lambda_list = self.reg_lambda
        elif (isinstance(self.reg_lambda, float)) or (isinstance(self.reg_lambda, int)):
            if (self.reg_lambda < 0) or (self.reg_lambda > 1):
                raise ValueError("reg_lambda must be >= 0 and <=1, got %s." % self.reg_lambda)
            self.reg_lambda_list = [self.reg_lambda]
        else:
            raise ValueError("Invalid reg_lambda")

        if isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if val < 0:
                    raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = self.reg_gamma
        elif (isinstance(self.reg_gamma, float)) or (isinstance(self.reg_gamma, int)):
            if (self.reg_gamma < 0) or (self.reg_gamma > 1):
                raise ValueError("reg_gamma must be >= 0 and <=1, got %s." % self.reg_gamma)
            self.reg_gamma_list = [self.reg_gamma]
        else:
            raise ValueError("Invalid reg_gamma")

    def _first_order(self, x, y):

        """calculate the projection indice using the first order stein's identity

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
        # for high dimensional data,
        # we speed up the computation by ignoring the correlation among variables.
        if x.shape[1] <= 100:
            mu = np.average(x, axis=0)
            cov = np.cov(x.T)
            inv_cov = np.linalg.pinv(cov, 1e-7)
            s1 = np.dot(inv_cov, (x - mu).T).T
        else:
            mu = np.average(x, axis=0)
            var = np.std(x, 0) ** 2
            var[var == 0] = np.inf
            inv_cov = np.diag(1 / var)
            s1 = np.diag(inv_cov) * (x - mu)
        beta = np.average(y.reshape(-1, 1) * s1, axis=0)
        return beta.reshape([-1, 1])
    
    def select_features(self, sample_indice):
            
        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        if self.n_feature_search > len(self.split_features):
            return self.split_features

        feature_impurity = []
        beta_parent = self._first_order(node_x, node_y)
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < self.EPSILON:
                continue

            split_point = 0
            max_deviation = 0
            for i, _ in enumerate(sortted_indice):

                if i == (n_samples - 1):
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue

                if sortted_feature[i + 1] <= sortted_feature[i] + self.EPSILON:
                    continue

                if self.min_samples_leaf < n_samples / (self.n_split_grid - 1):
                    if (i + 1) / n_samples < (split_point + 1) / (self.n_split_grid + 1):
                        continue
                elif n_samples > 2 * self.min_samples_leaf:
                    if (i + 1 - self.min_samples_leaf) / (n_samples - 2 * self.min_samples_leaf) < split_point / (self.n_split_grid - 1):
                        continue
                elif (i + 1) != self.min_samples_leaf:
                    continue

                split_point += 1
                left_indice = sortted_indice[:(i + 1)]
                right_indice = sortted_indice[(i + 1):]
                beta_left = self._first_order(node_x[left_indice], node_y[left_indice])
                beta_right = self._first_order(node_x[right_indice], node_y[right_indice])
                deviation = (len(left_indice) * np.linalg.norm(beta_parent - beta_left) +
                        len(right_indice) * np.linalg.norm(beta_parent - beta_right)) / n_samples
                if deviation > max_deviation:
                    pos = i + 1
                    max_deviation = deviation
                    threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

            if max_deviation > 0:
                left_indice = sample_indice[sortted_indice[:pos]]
                right_indice = sample_indice[sortted_indice[pos:]]
                if is_regressor(self):
                    left_clf = SimRegressor(reg_lambda=0, reg_gamma=1e-9, degree=self.degree,
                                    knot_num=self.knot_num, random_state=self.random_state)
                    left_clf.fit(self.x[left_indice], self.y[left_indice])

                    right_clf = SimRegressor(reg_lambda=0, reg_gamma=1e-9, degree=self.degree,
                                     knot_num=self.knot_num, random_state=self.random_state)
                    right_clf.fit(self.x[right_indice], self.y[right_indice])

                    left_impurity = self.get_loss(self.y[left_indice].ravel(), left_clf.predict(self.x[left_indice]))
                    right_impurity = self.get_loss(self.y[right_indice].ravel(), right_clf.predict(self.x[right_indice]))
                if is_classifier(self):
                    left_clf = SimClassifier(reg_lambda=0, reg_gamma=1e-9, degree=self.degree,
                                    knot_num=self.knot_num, random_state=self.random_state)
                    left_clf.fit(self.x[left_indice], self.y[left_indice])

                    right_clf = SimClassifier(reg_lambda=0, reg_gamma=1e-9, degree=self.degree,
                                     knot_num=self.knot_num, random_state=self.random_state)
                    right_clf.fit(self.x[right_indice], self.y[right_indice])

                    left_impurity = self.get_loss(self.y[left_indice].ravel(), left_clf.predict_proba(self.x[left_indice])[:, 1])
                    right_impurity = self.get_loss(self.y[right_indice].ravel(), right_clf.predict_proba(self.x[right_indice])[:, 1])
                current_impurity = (len(left_indice) * left_impurity + len(right_indice) * right_impurity) / n_samples
                feature_impurity.append(current_impurity)
            else:
                feature_impurity.append(np.inf)
        split_feature_indices = np.argsort(feature_impurity)[:self.n_feature_search]
        important_split_features = np.array(self.split_features)[split_feature_indices]
        return important_split_features

    def visualize_one_leaf(self, node_id, folder="./results/", name="leaf_sim", save_png=False, save_eps=False):

        """draw one of the leaf node.

        Parameters
        ---------
        node_id : int
            the id of leaf node
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
        if node_id not in self.leaf_estimators_.keys():
            print("Invalid leaf node id.")
            return

        if self.leaf_estimators_[node_id] is None:
            print("This is a constant node, and SIM is not available.")
            return

        projection_indices = np.array([est.beta_.flatten() for nodeid, est in self.leaf_estimators_.items() if est is not None]).T
        if projection_indices.shape[1] > 0:
            xlim_min = - max(np.abs(projection_indices.min() - 0.1), np.abs(projection_indices.max() + 0.1))
            xlim_max = max(np.abs(projection_indices.min() - 0.1), np.abs(projection_indices.max() + 0.1))

        fig = plt.figure(figsize=(10, 4))
        est = self.leaf_estimators_[node_id]
        outer = gridspec.GridSpec(1, 2, wspace=0.15)
        inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], wspace=0.1, hspace=0.1, height_ratios=[6, 1])
        ax1_main = fig.add_subplot(inner[0])
        xgrid = np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 100).reshape([-1, 1])
        ygrid = est.shape_fit_.decision_function(xgrid)
        ax1_main.plot(xgrid, ygrid, color="red")
        ax1_main.set_xticklabels([])
        ax1_main.set_title("Node " + str(node_id), fontsize=16)
        ax1_main.set_xticks(np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 5))
        fig.add_subplot(ax1_main)

        ax1_density = fig.add_subplot(inner[1])  
        xint = ((np.array(est.shape_fit_.bins_[1:]) + np.array(est.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, est.shape_fit_.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        ax1_density.set_xticks(np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 5))
        fig.add_subplot(ax1_density)

        ax2 = fig.add_subplot(outer[1])
        if len(est.beta_) <= 50:
            ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
            ax2.set_yticks(np.arange(len(est.beta_)))
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(est.beta_.ravel()))][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(est.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
        else:
            right = np.round(np.linspace(0, np.round(len(est.beta_) * 0.45).astype(int), 5))
            left = len(est.beta_) - 1 - right
            input_ticks = np.unique(np.hstack([left, right])).astype(int)

            ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
            ax2.set_yticks(input_ticks)
            ax2.set_yticklabels(["X" + str(idx + 1) for idx in input_ticks][::-1])
            ax2.set_xlim(xlim_min, xlim_max)
            ax2.set_ylim(-1, len(est.beta_))
            ax2.axvline(0, linestyle="dotted", color="black")
            
        ax2title = ""
        sortind = np.argsort(np.abs(est.beta_).ravel())[::-1]
        for i in range(est.beta_.shape[0]):
            if i == 0:
                ax2title += str(round(np.abs(est.beta_[sortind[i], 0]), 3)) + "X" + str(sortind[i] + 1)
                continue
            elif (i > 0) & (i < 3):
                if np.abs(est.beta_[sortind[i], 0]) > 0.001:
                    if est.beta_[sortind[i], 0] > 0:
                        ax2title += " + "
                    else:
                        ax2title += " - "
                    ax2title += str(round(np.abs(est.beta_[sortind[i], 0]), 3)) + "X" + str(sortind[i] + 1)
                else:
                    break
            elif i == 3:
                if np.abs(est.beta_[sortind[3], 0]) > 0.001:
                    ax2title += "+..."
            else:
                break
        ax2.set_title(ax2title)
        fig.add_subplot(ax2)
        plt.show()
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
        projection_indices = np.array([est.beta_.flatten() for nodeid, est in self.leaf_estimators_.items() if est is not None]).T
        max_ids = projection_indices.shape[1]
        fig = plt.figure(figsize=(8 * cols_per_row, 4.6 * int(np.ceil(max_ids / cols_per_row))))
        outer = gridspec.GridSpec(int(np.ceil(max_ids / cols_per_row)), cols_per_row, wspace=0.15, hspace=0.25)
        if max_ids > 0:
            xlim_min = - max(np.abs(projection_indices.min() - 0.1), np.abs(projection_indices.max() + 0.1))
            xlim_max = max(np.abs(projection_indices.min() - 0.1), np.abs(projection_indices.max() + 0.1))

        for node_id, est in self.leaf_estimators_.items():

            if est is None:
                continue

            inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.15, height_ratios=[6, 1], width_ratios=[3, 1])
            ax1_main = fig.add_subplot(inner[0, 0])
            xgrid = np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 100).reshape([-1, 1])
            ygrid = est.shape_fit_.decision_function(xgrid)
            ax1_main.plot(xgrid, ygrid, color="red")
            ax1_main.set_xticklabels([])
            ax1_main.set_title("Node " + str(node_id), fontsize=16)
            ax1_main.set_xticks(np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 5))
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(est.shape_fit_.bins_[1:]) + np.array(est.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, est.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
            ax1_density.set_xticks(np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, 5))
            fig.add_subplot(ax1_density)

            ax2 = fig.add_subplot(inner[:, 1])
            if len(est.beta_) <= 50:
                ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
                ax2.set_yticks(np.arange(len(est.beta_)))
                ax2.set_yticklabels(["X" + str(idx + 1) for idx in range(len(est.beta_.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(est.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                right = np.round(np.linspace(0, np.round(len(est.beta_) * 0.45).astype(int), 5))
                left = len(est.beta_) - 1 - right
                input_ticks = np.unique(np.hstack([left, right])).astype(int)

                ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
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


class LIFTNetRegressor(BaseLIFTNet, BaseMoBTreeRegressor, RegressorMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 n_split_grid=10, split_features=None, n_feature_search=5,
                 degree=3, knot_num=5, reg_lambda=0.001, reg_gamma=0.000001, random_state=0):

        super(LIFTNetRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 n_split_grid=n_split_grid,
                                 split_features=split_features,
                                 n_feature_search=n_feature_search,
                                 degree=degree,
                                 knot_num=knot_num,
                                 reg_lambda=reg_lambda,
                                 reg_gamma=reg_gamma,
                                 random_state=random_state)

    def build_root(self):

        root_clf = SimRegressor(reg_lambda=0, reg_gamma=1e-6, degree=self.degree,
                        knot_num=self.knot_num, random_state=self.random_state)
        root_clf.fit(self.x, self.y)
        root_impurity = self.get_loss(self.y, root_clf.predict(self.x))
        return root_impurity

    def build_leaf(self, sample_indice):

        base = SimRegressor(degree=self.degree, knot_num=self.knot_num, random_state=self.random_state)
        grid = GridSearchCV(base, param_grid={"reg_lambda": self.reg_lambda_list,
                                  "reg_gamma": self.reg_gamma_list},
                      scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
                      cv=5, refit="mse", n_jobs=1, error_score=np.nan)
        grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
        best_estimator = grid.best_estimator_
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity

    def node_split(self, sample_indice):

        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        best_feature = None
        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        important_split_features = self.select_features(sample_indice)
        for feature_indice in important_split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < self.EPSILON:
                continue

            split_point = 0
            for i, _ in enumerate(sortted_indice):

                if i == (n_samples - 1):
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue
                
                if sortted_feature[i + 1] <= sortted_feature[i] + self.EPSILON:
                    continue

                if self.min_samples_leaf < n_samples / (self.n_split_grid - 1):
                    if (i + 1) / n_samples < (split_point + 1) / (self.n_split_grid + 1):
                        continue
                elif n_samples > 2 * self.min_samples_leaf:
                    if (i + 1 - self.min_samples_leaf) / (n_samples - 2 * self.min_samples_leaf) < split_point / (self.n_split_grid - 1):
                        continue
                elif (i + 1) != self.min_samples_leaf:
                    continue

                split_point += 1
                left_indice = sortted_indice[:(i + 1)]
                estimator = SimRegressor(reg_lambda=0, reg_gamma=1e-6, degree=self.degree,
                                 knot_num=self.knot_num,
                                 random_state=self.random_state)
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = self.get_loss(node_y[left_indice].ravel(), estimator.predict(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                estimator = SimRegressor(reg_lambda=0, reg_gamma=1e-6, degree=self.degree,
                                 knot_num=self.knot_num,
                                 random_state=self.random_state)
                estimator.fit(node_x[right_indice], node_y[right_indice])
                right_impurity = self.get_loss(node_y[right_indice].ravel(), estimator.predict(node_x[right_indice]))

                current_impurity = (len(left_indice) * left_impurity + len(right_indice) * right_impurity) / n_samples
                if current_impurity < best_impurity:
                    best_position = i + 1
                    best_feature = feature_indice
                    best_impurity = current_impurity
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity
                    best_threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

        if best_position is not None:
            sortted_indice = np.argsort(node_x[:, best_feature])
            best_left_indice = sample_indice[sortted_indice[:best_position]]
            best_right_indice = sample_indice[sortted_indice[best_position:]]
            
        node = {"feature": best_feature, "threshold": best_threshold, "left": best_left_indice, "right": best_right_indice,
              "impurity": best_impurity, "left_impurity": best_left_impurity, "right_impurity": best_right_impurity}
        return node


class LIFTNetClassifier(BaseLIFTNet, BaseMoBTreeClassifier, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 n_split_grid=10, split_features=None, n_feature_search=5,
                 degree=3, knot_num=5, reg_lambda=0.001, reg_gamma=0.000001, random_state=0):

        super(LIFTNetClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 n_split_grid=n_split_grid,
                                 split_features=split_features,
                                 n_feature_search=n_feature_search,
                                 degree=degree,
                                 knot_num=knot_num,
                                 reg_lambda=reg_lambda,
                                 reg_gamma=reg_gamma,
                                 random_state=random_state)

    def build_root(self):

        root_clf = SimClassifier(reg_lambda=0, reg_gamma=1e-6, degree=self.degree,
                         knot_num=self.knot_num, random_state=self.random_state)
        root_clf.fit(self.x, self.y)
        root_impurity = self.get_loss(self.y, root_clf.predict_proba(self.x)[:, 1])
        return root_impurity

    def build_leaf(self, sample_indice):

        if (self.y[sample_indice].std() == 0) | (self.y[sample_indice].sum() < 5) | ((1 - self.y[sample_indice]).sum() < 5):
            best_impurity = 0
            best_estimator = None
            predict_func = lambda x: np.mean(self.y[sample_indice])
        else:
            base = SimClassifier(degree=self.degree, knot_num=self.knot_num)
            grid = GridSearchCV(base, param_grid={"reg_lambda": self.reg_lambda_list,
                                      "reg_gamma": self.reg_gamma_list},
                          scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)},
                          cv=5, refit="auc", n_jobs=1, error_score=np.nan)
            grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
            best_estimator = grid.best_estimator_
            predict_func = lambda x: best_estimator.predict_proba(x)[:, 1]
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity

    def node_split(self, sample_indice):

        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        best_feature = None
        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        important_split_features = self.select_features(sample_indice)
        for feature_indice in important_split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < self.EPSILON:
                continue

            split_point = 0
            for i, _ in enumerate(sortted_indice):

                if i == (n_samples - 1):
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue
                
                if sortted_feature[i + 1] <= sortted_feature[i] + self.EPSILON:
                    continue

                if self.min_samples_leaf < n_samples / (self.n_split_grid - 1):
                    if (i + 1) / n_samples < (split_point + 1) / (self.n_split_grid + 1):
                        continue
                elif n_samples > 2 * self.min_samples_leaf:
                    if (i + 1 - self.min_samples_leaf) / (n_samples - 2 * self.min_samples_leaf) < split_point / (self.n_split_grid - 1):
                        continue
                elif (i + 1) != self.min_samples_leaf:
                    continue

                split_point += 1
                left_indice = sortted_indice[:(i + 1)]
                estimator = SimClassifier(reg_lambda=0, reg_gamma=1e-6, degree=self.degree,
                                 knot_num=self.knot_num,
                                 random_state=self.random_state)
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = self.get_loss(node_y[left_indice].ravel(), estimator.predict_proba(node_x[left_indice])[:, 1])

                right_indice = sortted_indice[(i + 1):]
                estimator = SimClassifier(reg_lambda=0, reg_gamma=1e-6, degree=self.degree,
                                 knot_num=self.knot_num,
                                 random_state=self.random_state)
                estimator.fit(node_x[right_indice], node_y[right_indice])
                right_impurity = self.get_loss(node_y[right_indice].ravel(), estimator.predict_proba(node_x[right_indice])[:, 1])

                current_impurity = (len(left_indice) * left_impurity + len(right_indice) * right_impurity) / n_samples
                if current_impurity < best_impurity:
                    best_position = i + 1
                    best_feature = feature_indice
                    best_impurity = current_impurity
                    best_left_impurity = left_impurity
                    best_right_impurity = right_impurity
                    best_threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

        if best_position is not None:
            sortted_indice = np.argsort(node_x[:, best_feature])
            best_left_indice = sample_indice[sortted_indice[:best_position]]
            best_right_indice = sample_indice[sortted_indice[best_position:]]

        node = {"feature": best_feature, "threshold": best_threshold, "left": best_left_indice, "right": best_right_indice,
              "impurity": best_impurity, "left_impurity": best_left_impurity, "right_impurity": best_right_impurity}
        return node
