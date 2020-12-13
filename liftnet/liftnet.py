import os
import numpy as np
from matplotlib import gridspec
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
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0.0001, n_split_grid=10, split_features=None,
                 degree=3, knot_num=5, nterms=5, reg_gamma=0.1, val_ratio=0.2, random_state=0):

        self.max_depth = max_depth
        self.n_split_grid = n_split_grid
        self.split_features = split_features
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

        self.degree = degree
        self.knot_num = knot_num
        self.nterms = nterms
        self.reg_gamma = reg_gamma

        self.val_ratio = val_ratio
        self.random_state = random_state

    def _validate_hyperparameters(self):

        if not isinstance(self.max_depth, int):
            raise ValueError("degree must be an integer, got %s." % self.max_depth)

            if self.max_depth < 0:
                raise ValueError("degree must be >= 0, got %s." % self.max_depth)

        if self.split_features is not None:
            if not isinstance(self.split_features, list):
                raise ValueError("split_features must be an list or None, got %s." % self.split_features)

        if not isinstance(self.min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer, got %s." % self.min_samples_leaf)

            if self.min_samples_leaf < 0:
                raise ValueError("min_samples_leaf must be >= 0, got %s." % self.min_samples_leaf)

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be >= 0, got %s." % self.min_impurity_decrease)

        if self.val_ratio <= 0:
            raise ValueError("val_ratio must be > 0, got %s." % self.val_ratio)
        elif self.val_ratio >= 1:
            raise ValueError("val_ratio must be < 1, got %s." % self.val_ratio)

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)
            if self.degree < 0:
                raise ValueError("degree must be >= 0, got %s." % self.degree)

        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)
            if self.knot_num <= 0:
                raise ValueError("knot_num must be > 0, got %s." % self.knot_num)

        if isinstance(self.nterms, list):
            for val in self.nterms:
                if val <= 0:
                    raise ValueError("all the elements in nterms must be >= 1, got %s." % self.nterms)
            self.nterms_list = self.nterms
        elif (isinstance(self.nterms, float)) or (isinstance(self.nterms, int)):
            if self.nterms <= 0:
                raise ValueError("nterms must be >= 1, got %s." % self.nterms)
            self.nterms_list = [self.nterms]
        else:
            raise ValueError("Invalid nterms")

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

        mu = np.average(x, axis=0)
        cov = np.cov(x.T)
        inv_cov = np.linalg.pinv(cov, 1e-5)
        s1 = np.dot(inv_cov, (x - mu).T).T
        s1 = np.diag(inv_cov) * (x - mu)
        beta = np.average(y.reshape(-1, 1) * s1, axis=0)
        return beta.reshape([-1, 1])

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
            raise("Invalid leaf node id.")

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
        fig.add_subplot(ax1_main)

        ax1_density = fig.add_subplot(inner[1])  
        xint = ((np.array(est.shape_fit_.bins_[1:]) + np.array(est.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
        ax1_density.bar(xint, est.shape_fit_.density_, width=xint[1] - xint[0])
        ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
        ax1_density.set_yticklabels([])
        fig.add_subplot(ax1_density)

        ax2 = fig.add_subplot(outer[1])
        if len(est.beta_) <= 20:
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
        for i in range(3):
            if i > 0:
                if est.beta_[sortind[i], 0] > 0:
                    ax2title += " + "
                else:
                    ax2title += " - "
            if np.abs(est.beta_[sortind[i], 0]) > 0.001:
                ax2title += str(round(np.abs(est.beta_[sortind[i], 0]), 3)) + "X" + str(sortind[i] + 1)
        ax2title += "+..."
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
            fig.add_subplot(ax1_main)

            ax1_density = fig.add_subplot(inner[1, 0])  
            xint = ((np.array(est.shape_fit_.bins_[1:]) + np.array(est.shape_fit_.bins_[:-1])) / 2).reshape([-1, 1]).reshape([-1])
            ax1_density.bar(xint, est.shape_fit_.density_, width=xint[1] - xint[0])
            ax1_main.get_shared_x_axes().join(ax1_main, ax1_density)
            ax1_density.set_yticklabels([])
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


class LIFTNetRegressor(BaseLIFTNet, BaseMOBRegressor, RegressorMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, n_split_grid=10, split_features=None,
                 degree=3, knot_num=5, nterms=5, reg_gamma=0.1,
                 val_ratio=0.2, random_state=0):

        super(LIFTNetRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 n_split_grid=n_split_grid,
                                 split_features=split_features,
                                 degree=degree,
                                 knot_num=knot_num,
                                 nterms=nterms,
                                 reg_gamma=reg_gamma,
                                 val_ratio=val_ratio,
                                 random_state=random_state)

    def build_root(self):

        root_clf = SimRegressor(nterms=None, reg_gamma=1e-9, degree=self.degree,
                        knot_num=self.knot_num, random_state=self.random_state)
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
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity

    def node_split(self, sample_indice):

        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        feature_impurity = []
        beta_parent = self._first_order(node_x, node_y)
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < EPSILON:
                continue

            max_deviation = 0
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
                deviation = (len(left_indice) * np.linalg.norm(beta_parent - beta_left) +
                        len(right_indice) * np.linalg.norm(beta_parent - beta_right)) / n_samples
                if deviation > max_deviation:
                    pos = i + 1
                    max_deviation = deviation
                    threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

            if max_deviation > 0:
                left_indice = sample_indice[sortted_indice[:pos]]
                right_indice = sample_indice[sortted_indice[pos:]]
                left_clf = SimRegressor(nterms=None, reg_gamma=1e-9, degree=self.degree,
                                knot_num=self.knot_num, random_state=self.random_state)
                left_clf.fit(self.x[left_indice], self.y[left_indice])

                right_clf = SimRegressor(nterms=None, reg_gamma=1e-9, degree=self.degree,
                                 knot_num=self.knot_num, random_state=self.random_state)
                right_clf.fit(self.x[right_indice], self.y[right_indice])

                left_impurity = self.get_loss(self.y[left_indice].ravel(), left_clf.predict(self.x[left_indice]))
                right_impurity = self.get_loss(self.y[right_indice].ravel(), right_clf.predict(self.x[right_indice]))
                current_impurity = (len(left_indice) * left_impurity + len(right_indice) * right_impurity) / n_samples
                feature_impurity.append(current_impurity)
            else:
                feature_impurity.append(np.inf)

        best_feature = None
        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        split_features = np.argsort(feature_impurity)[:1]
        for feature_indice in split_features:

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
                estimator = SimRegressor(nterms=None, reg_gamma=1e-9, degree=self.degree,
                                 knot_num=self.knot_num,
                                 random_state=self.random_state)
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = self.get_loss(node_y[left_indice].ravel(), estimator.predict(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                estimator = SimRegressor(nterms=None, reg_gamma=1e-9, degree=self.degree,
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


class LIFTNetClassifier(BaseLIFTNet, BaseMOBClassifier, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, n_split_grid=10, split_features=None,
                 degree=3, knot_num=5, nterms=5, reg_gamma=0.1, val_ratio=0.2, random_state=0):

        super(LIFTNetClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 n_split_grid=n_split_grid,
                                 split_features=split_features,
                                 degree=degree,
                                 knot_num=knot_num,
                                 nterms=nterms,
                                 reg_gamma=reg_gamma,
                                 val_ratio=val_ratio,
                                 random_state=random_state)

    def build_root(self):

        root_clf = SimClassifier(nterms=None, reg_gamma=1e-9, degree=self.degree,
                         knot_num=self.knot_num, random_state=self.random_state)
        root_clf.fit(self.x, self.y)
        root_impurity = self.get_loss(self.y, root_clf.predict_proba(self.x)[:, 1])
        return root_impurity

    def build_leaf(self, sample_indice):

        best_estimator = None
        n_samples = len(sample_indice)
        idx1, idx2 = train_test_split(sample_indice, test_size=self.val_ratio, random_state=self.random_state)
        if (self.y[sample_indice].std() == 0) | (self.y[idx1].std() == 0) | (self.y[idx2].std() == 0):
            best_impurity = 0
            predict_func = lambda x: np.mean(self.y[sample_indice])
        else:
            best_impurity = np.inf
            for nterms in self.nterms_list:
                for reg_gamma in self.reg_gamma_list:
                    estimator = SimClassifier(degree=self.degree,
                             nterms=nterms, reg_gamma=reg_gamma, knot_num=self.knot_num,
                             random_state=self.random_state)
                    estimator.fit(self.x[idx1], self.y[idx1])
                    current_impurity = self.get_loss(self.y[idx2], estimator.predict_proba(self.x[idx2])[:, 1])
                    if current_impurity < best_impurity:
                        best_estimator = estimator
                        best_impurity = current_impurity
            best_estimator.fit(self.x[sample_indice], self.y[sample_indice])
            predict_func = lambda x: best_estimator.predict_proba(x)[:, 1]
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity

    def node_split(self, sample_indice):

        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        feature_impurity = []
        beta_parent = self._first_order(node_x, node_y)
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < EPSILON:
                continue

            max_deviation = 0
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
                deviation = (len(left_indice) * np.linalg.norm(beta_parent - beta_left) +
                        len(right_indice) * np.linalg.norm(beta_parent - beta_right)) / n_samples
                if deviation > max_deviation:
                    pos = i + 1
                    max_deviation = deviation
                    threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

            if max_deviation > 0:
                left_indice = sample_indice[sortted_indice[:pos]]
                right_indice = sample_indice[sortted_indice[pos:]]
                left_clf = SimClassifier(nterms=None, reg_gamma=1e-9, degree=self.degree,
                                knot_num=self.knot_num, random_state=self.random_state)
                left_clf.fit(self.x[left_indice], self.y[left_indice])

                right_clf = SimClassifier(nterms=None, reg_gamma=1e-9, degree=self.degree,
                                 knot_num=self.knot_num, random_state=self.random_state)
                right_clf.fit(self.x[right_indice], self.y[right_indice])

                left_impurity = self.get_loss(self.y[left_indice].ravel(), left_clf.predict_proba(self.x[left_indice])[:, 1])
                right_impurity = self.get_loss(self.y[right_indice].ravel(), right_clf.predict_proba(self.x[right_indice])[:, 1])
                current_impurity = (len(left_indice) * left_impurity + len(right_indice) * right_impurity) / n_samples
                feature_impurity.append(current_impurity)
            else:
                feature_impurity.append(np.inf)

        best_feature = None
        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        split_features = np.argsort(feature_impurity)[:1]
        for feature_indice in split_features:

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
                estimator = SimClassifier(nterms=None, reg_gamma=1e-9, degree=self.degree,
                                 knot_num=self.knot_num,
                                 random_state=self.random_state)
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = self.get_loss(node_y[left_indice].ravel(), estimator.predict_proba(node_x[left_indice])[:, 1])

                right_indice = sortted_indice[(i + 1):]
                estimator = SimClassifier(nterms=None, reg_gamma=1e-9, degree=self.degree,
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
