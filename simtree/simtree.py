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
from .mobtree import MoBTree, MoBTreeRegressor, MoBTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

__all__ = ["SIMTreeRegressor", "SIMTreeClassifier"]


class SIMTree(metaclass=ABCMeta):
    """
        Base SIMTree class for classification and regression.
     """

    def __init__(self, max_depth=3, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=1, n_feature_search=10, n_split_grid=20,
                 degree=3, knot_num=20, reg_lambda=0, reg_gamma=1e-5, random_state=0):

        super(SIMTree, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.degree = degree
        self.knot_num = knot_num
        self.reg_gamma = reg_gamma
        self.reg_lambda = reg_lambda

    def _validate_hyperparameters(self):

        super(SIMTree, self)._validate_hyperparameters()

        if not isinstance(self.degree, (np.int, int)):
            raise ValueError("degree must be an integer, got %s." % self.degree)
            if self.degree < 0:
                raise ValueError("degree must be >= 0, got %s." % self.degree)

        if not isinstance(self.knot_num, (np.int, int)):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
            if self.knot_num <= 0:
                raise ValueError("knot_num must be > 0, got %s." % self.knot_num)

        if isinstance(self.reg_lambda, list):
            for val in self.reg_lambda:
                if isinstance(val, (np.ndarray, np.int, int, np.floating, float)):
                    if val < 0:
                        raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_lambda)
                else:
                    raise ValueError("Invalid reg_lambda")
        elif isinstance(self.reg_lambda, (np.ndarray, np.int, int, np.floating, float)):
            if self.reg_lambda < 0:
                raise ValueError("reg_lambda must be >= 0, got %s." % self.reg_lambda)
            self.reg_lambda = [self.reg_lambda]
        else:
            raise ValueError("Invalid reg_lambda")

        if isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if isinstance(val, (np.ndarray, np.int, int, np.floating, float)):
                    if (val < 0) or (val > 1):
                        raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
                else:
                    raise ValueError("Invalid reg_gamma")
        elif isinstance(self.reg_gamma, (np.ndarray, np.int, int, np.floating, float)):
            if (self.reg_gamma < 0) or (self.reg_gamma > 1):
                raise ValueError("reg_gamma must be >= 0 and <=1, got %s." % self.reg_gamma)
            self.reg_gamma = [self.reg_gamma]
        else:
            raise ValueError("Invalid reg_gamma")
    
    def get_projection_index(self, node_id):
        
        """return the projection index of one leaf node.

        Parameters
        ---------
        node_id : int
            the id of leaf node
        """

        check_is_fitted(self, "tree")
        if node_id not in self.leaf_estimators_.keys():
            print("Invalid leaf node id.")
            return

        if self.leaf_estimators_[node_id] is None:
            print("This is a constant node, and SIM is not available.")
            return

        return self.leaf_estimators_[node_id].beta_.flatten()

    def get_feature_importance(self, node_id):
        
        """return the feature_importance of one leaf node.

        Parameters
        ---------
        node_id : int
            the id of leaf node
        """

        check_is_fitted(self, "tree")
        if node_id not in self.leaf_estimators_.keys():
            print("Invalid leaf node id.")
            return

        if self.leaf_estimators_[node_id] is None:
            print("This is a constant node, and SIM is not available.")
            return

        importance = (self.x[self.decision_path_indice(self.x, node_id)] * self.leaf_estimators_[node_id].beta_.ravel()).std(0, ddof=1)
        return importance

    def get_projection_equation(self, node_id, precision=3):
        
        """return the projection equation of one leaf node in string format.

        Parameters
        ---------
        node_id : int
            the id of leaf node
        precision : int
            the precision of coefficients
        """

        check_is_fitted(self, "tree")
        if node_id not in self.leaf_estimators_.keys():
            print("Invalid leaf node id.")
            return

        if self.leaf_estimators_[node_id] is None:
            print("This is a constant node, and SIM is not available.")
            return

        equation = ""
        est = self.leaf_estimators_[node_id]
        importance = self.get_feature_importance(node_id)
        sortind = np.argsort(importance)[::-1]
        for i in range(est.beta_.shape[0]):
            if i == 0:
                if est.beta_[sortind[i], 0] < 0:
                    equation += "- "
                equation += str(round(np.abs(est.beta_[sortind[i], 0]), precision)) + self.feature_names[sortind[i]]
                continue
            else:
                if np.abs(est.beta_[sortind[i], 0]) > 0:
                    if est.beta_[sortind[i], 0] > 0:
                        equation += " + "
                    else:
                        equation += " - "
                    equation += str(round(np.abs(est.beta_[sortind[i], 0]), precision)) + self.feature_names[sortind[i]]
        return equation
    
    def get_sparsity(self, node_id):
                
        """return the sparsity of the projection index in one leaf node, i.e., the percentage of zero coefficients.

        Parameters
        ---------
        node_id : int
            the id of leaf node
        """

        check_is_fitted(self, "tree")
        if node_id not in self.leaf_estimators_.keys():
            print("Invalid leaf node id.")
            return

        if self.leaf_estimators_[node_id] is None:
            print("This is a constant node, and SIM is not available.")
            return

        est = self.leaf_estimators_[node_id]
        sparsity = np.mean(est.beta_ == 0)
        return sparsity

    def get_roughness(self, node_id, grid_size=100):
                
        """return the roughness of the ridge function in one leaf node, i.e., the root-mean-square second derivative of the ridge function.
           for a fair comparison, the roughness is adjusted by mapping the x to be within [0, 1]

        Parameters
        ---------
        node_id : int
            the id of leaf node
        grid_size : int
            the number of grid points for approximation
        """

        check_is_fitted(self, "tree")
        if node_id not in self.leaf_estimators_.keys():
            print("Invalid leaf node id.")
            return

        if self.leaf_estimators_[node_id] is None:
            print("This is a constant node, and SIM is not available.")
            return

        est = self.leaf_estimators_[node_id]
        adj = (est.shape_fit_.xmax - est.shape_fit_.xmin) ** 2
        xgrid = np.linspace(est.shape_fit_.xmin, est.shape_fit_.xmax, grid_size + 2)[1:-1]
        roughness = adj * np.sqrt(np.mean([(est.shape_fit_.diff(x, order=2)) ** 2 for x in xgrid]))
        return roughness

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
        outer = gridspec.GridSpec(1, 2, wspace=0.25, width_ratios=[1.2, 1])
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

        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.2, hspace=0.1, width_ratios=[1, 1])
        ax2_coef = fig.add_subplot(inner[0])
        if len(est.beta_) <= 50:
            ax2_coef.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
            ax2_coef.set_yticks(np.arange(len(est.beta_)))
            ax2_coef.set_yticklabels([self.feature_names[idx] for idx in range(len(est.beta_.ravel()))][::-1])
            ax2_coef.set_xlim(xlim_min, xlim_max)
            ax2_coef.set_ylim(-1, len(est.beta_))
            ax2_coef.axvline(0, linestyle="dotted", color="black")
        else:
            right = np.round(np.linspace(0, np.round(len(est.beta_) * 0.45).astype(int), 5))
            left = len(est.beta_) - 1 - right
            input_ticks = np.unique(np.hstack([left, right])).astype(int)

            ax2_coef.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
            ax2_coef.set_yticks(input_ticks)
            ax2_coef.set_yticklabels([self.feature_names[idx] for idx in input_ticks][::-1])
            ax2_coef.set_xlim(xlim_min, xlim_max)
            ax2_coef.set_ylim(-1, len(est.beta_))
            ax2_coef.axvline(0, linestyle="dotted", color="black")

        ax2_coef.set_title("Projection Index")
        fig.add_subplot(ax2_coef)

        ax2_importance = fig.add_subplot(inner[1])  
        ax2_coef.get_shared_y_axes().join(ax2_coef, ax2_importance)
        ax2_importance.set_yticklabels([])
        ax2_importance.barh(self.feature_names, self.get_feature_importance(node_id)[::-1])
        ax2_importance.set_title("Importance")
        fig.add_subplot(ax2_importance)
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

            inner = outer[subfig_idx].subgridspec(2, 2, wspace=0.25, height_ratios=[6, 1], width_ratios=[3, 1])
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
                ax2.set_yticklabels([self.feature_names[idx][:8] for idx in range(len(est.beta_.ravel()))][::-1])
                ax2.set_xlim(xlim_min, xlim_max)
                ax2.set_ylim(-1, len(est.beta_))
                ax2.axvline(0, linestyle="dotted", color="black")
            else:
                right = np.round(np.linspace(0, np.round(len(est.beta_) * 0.45).astype(int), 5))
                left = len(est.beta_) - 1 - right
                input_ticks = np.unique(np.hstack([left, right])).astype(int)

                ax2.barh(np.arange(len(est.beta_)), [beta for beta in est.beta_.ravel()][::-1])
                ax2.set_yticks(input_ticks)
                ax2.set_yticklabels([self.feature_names[idx][:8] for idx in input_ticks][::-1])
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


class SIMTreeRegressor(SIMTree, MoBTreeRegressor, RegressorMixin):

    def __init__(self, max_depth=3, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=1, n_feature_search=10, n_split_grid=20,
                 degree=3, knot_num=20, reg_lambda=0, reg_gamma=1e-5, random_state=0):

        super(SIMTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 degree=degree,
                                 knot_num=knot_num,
                                 reg_lambda=reg_lambda,
                                 reg_gamma=reg_gamma,
                                 random_state=random_state)

        self.base_estimator = SimRegressor(reg_lambda=0, reg_gamma=1e-5, degree=self.degree,
                                 knot_num=self.knot_num, random_state=self.random_state)

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        base = SimRegressor(reg_gamma=self.reg_gamma, degree=self.degree,
                      knot_num=self.knot_num, random_state=self.random_state)
        grid = GridSearchCV(base, param_grid={"reg_lambda": self.reg_lambda},
                      scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
                      cv=5, refit="mse", n_jobs=1, error_score=np.nan)
        grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
        best_estimator = grid.best_estimator_
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity


class SIMTreeClassifier(SIMTree, MoBTreeClassifier, ClassifierMixin):

    def __init__(self, max_depth=3, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=1, n_feature_search=10, n_split_grid=20,
                 degree=3, knot_num=20, reg_lambda=0, reg_gamma=1e-5, random_state=0):

        super(SIMTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 degree=degree,
                                 knot_num=knot_num,
                                 reg_lambda=reg_lambda,
                                 reg_gamma=reg_gamma,
                                 random_state=random_state)

        self.base_estimator = SimClassifier(reg_lambda=0, reg_gamma=1e-5, degree=self.degree,
                                 knot_num=self.knot_num, random_state=self.random_state)

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        if (self.y[sample_indice].std() == 0) | (self.y[sample_indice].sum() < 5) | ((1 - self.y[sample_indice]).sum() < 5):
            best_estimator = None
            predict_func = lambda x: np.ones(x.shape[0]) * self.y[sample_indice].mean()
            best_impurity = self.get_loss(self.y[sample_indice], predict_func(self.x[sample_indice]))
        else:
            base = SimClassifier(reg_gamma=self.reg_gamma, degree=self.degree,
                          knot_num=self.knot_num, random_state=self.random_state)
            grid = GridSearchCV(base, param_grid={"reg_lambda": self.reg_lambda},
                          scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)},
                          cv=5, refit="auc", n_jobs=1, error_score=np.nan)
            grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
            best_estimator = grid.best_estimator_
            predict_func = lambda x: best_estimator.predict_proba(x)[:, 1]
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity
