import os 
import copy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import PredefinedSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, mean_squared_error, log_loss

from sklearn.linear_model import Ridge, LinearRegression, LogisticRegression

from pysim import SimRegressor, SimClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

EPSILON = 1e-7

class BaseDTSim(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for sim classification and regression.
     """

    @abstractmethod
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, split_method="constant", base_method="constant",
                 n_split_grid=10, degree=2, knot_num=10, reg_lambda=0.1, reg_gamma=10, random_state=0):

        self.max_depth = max_depth
        self.base_method = base_method
        self.split_method = split_method
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        
        self.degree = degree
        self.knot_num = knot_num
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.n_split_grid = n_split_grid

        self.random_state = random_state

        if self.split_method == "constant":
            self.node_split = self.node_split_constant
        elif self.split_method == "sim":
            self.node_split = self.node_split_sim
        elif self.split_method == "glm":
            self.node_split = self.node_split_glm

    def _validate_hyperparameters(self):

        if not isinstance(self.max_depth, int):
            raise ValueError("degree must be an integer, got %s." % self.max_depth)

            if self.max_depth < 0:
                raise ValueError("degree must be >= 0, got" % self.max_depth)
   
        if self.base_method not in ["sim", "glm", "constant"]:
            raise ValueError("method must be an element of [sim, glm, constant], got %s." % 
                         self.base_method)

        if self.split_method not in ["sim", "glm", "constant"]:
            raise ValueError("method must be an element of [sim, glm, constant], got %s." % 
                         self.split_method)
        
        if not isinstance(self.min_samples_leaf, int):
            raise ValueError("degree must be an integer, got %s." % self.min_samples_leaf)

            if self.min_samples_leaf < 0:
                raise ValueError("degree must be >= 0, got" % self.min_samples_leaf)

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be >= 0, got %s." % self.min_impurity_decrease)

        if not isinstance(self.degree, int):
            raise ValueError("degree must be an integer, got %s." % self.degree)

            if self.degree < 0:
                raise ValueError("degree must be >= 0, got" % self.degree)

        if not isinstance(self.knot_num, int):
            raise ValueError("knot_num must be an integer, got %s." % self.knot_num)

            if self.knot_num <= 0:
                raise ValueError("knot_num must be > 0, got" % self.knot_num)

        if isinstance(self.reg_lambda, list):
            for val in self.reg_lambda:
                if val < 0:
                    raise ValueError("all the elements in reg_lambda must be >= 0, got %s." % self.reg_lambda)
            self.reg_lambda_list = self.reg_lambda  
        elif (isinstance(self.reg_lambda, float)) or (isinstance(self.reg_lambda, int)):
            if (self.reg_lambda < 0) or (self.reg_lambda > 1):
                raise ValueError("reg_lambda must be >= 0 and <=1, got %s." % self.reg_lambda)
            self.reg_lambda_list = [self.reg_lambda]

        if isinstance(self.reg_gamma, list):
            for val in self.reg_gamma:
                if val < 0:
                    raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = self.reg_gamma  
        elif (isinstance(self.reg_gamma, float)) or (isinstance(self.reg_gamma, int)):
            if self.reg_gamma < 0:
                raise ValueError("all the elements in reg_gamma must be >= 0, got %s." % self.reg_gamma)
            self.reg_gamma_list = [self.reg_gamma]

    @abstractmethod
    def node_split_constant(self):
        pass

    @abstractmethod
    def node_split_glm(self):
        pass

    @abstractmethod
    def node_split_sim(self):
        pass
    
    def add_node(self, parent_id, is_left, is_leaf, depth, feature, threshold, impurity, sample_indice):

        self.node_count += 1
        if parent_id is not None:
            if is_left:
                self.tree[parent_id].update({"left_child_id":self.node_count})
            else:
                self.tree[parent_id].update({"right_child_id":self.node_count})

        node_id = self.node_count
        n_samples = len(sample_indice)
        if is_leaf:
            predict_func, estimator = self.build_leaf(sample_indice)
            node = {"node_id":node_id, "parent_id":parent_id, "depth":depth, "feature":feature, "impurity":impurity,
                  "n_samples": n_samples, "is_left":is_left, "is_leaf":is_leaf, "value":np.mean(self.y[sample_indice]),
                  "predict_func":predict_func, "estimator":estimator}
        else:
            node = {"node_id":node_id, "parent_id":parent_id, "depth":depth,"feature":feature, "impurity":impurity,
                  "n_samples": n_samples, "is_left":is_left, "is_leaf":is_leaf, "value":np.mean(self.y[sample_indice]),
                  "left_child_id":None, "right_child_id":None, "threshold":threshold}            
        self.tree.update({node_id:node})
        return node_id
    
    def fit(self, x, y):

        self.tree = {}
        self.node_count = 0
        self._validate_hyperparameters()
        self.x, self.y = self._validate_input(x, y)
        n_samples, n_features = self.x.shape
        sample_indice = np.arange(n_samples)

        np.random.seed(self.random_state)
        root_impurity = self.build_root()
        root_node = {"sample_indice": sample_indice,
                 "parent_id":None,
                 "depth": 0,
                 "impurity":root_impurity,
                 "is_left":False}
        pending_node_list = [root_node]
        while len(pending_node_list) > 0:
            stack_record = pending_node_list.pop()
            sample_indice = stack_record["sample_indice"]
            parent_id = stack_record["parent_id"]
            depth = stack_record["depth"]
            impurity = stack_record["impurity"]
            is_left = stack_record["is_left"]

            if sample_indice is None:
                is_leaf = True
            else:
                n_samples = len(sample_indice)
                is_leaf = (depth >= self.max_depth or
                       n_samples < 2 * self.min_samples_leaf)
                
            if not is_leaf:
                split = self.node_split(sample_indice)
                impurity_improvement = impurity - split["impurity"]
                is_leaf = (is_leaf or (impurity_improvement < self.min_impurity_decrease) or
                        (split["left"] is None) or (split["right"] is None))
              
            if is_leaf:
                node_id = self.add_node(parent_id, is_left, is_leaf, depth, 
                                None, None, impurity, sample_indice)
            else:
                node_id = self.add_node(parent_id, is_left, is_leaf, depth, 
                                split["feature"], split["threshold"], impurity, sample_indice)

                pending_node_list.append({"sample_indice":split["left"],
                                 "parent_id":node_id,
                                 "depth":depth + 1,
                                 "impurity":split["left_impurity"],
                                 "is_left":True})
                pending_node_list.append({"sample_indice":split["right"],
                                 "parent_id":node_id,
                                 "depth": depth + 1,
                                 "impurity":split["right_impurity"],
                                 "is_left":False})
        return self

    def decision_path(self, x):

        n_samples = x.shape[0]
        path_all = np.zeros((n_samples, self.node_count))
        for idx, row in enumerate(x):
            path = []
            node = self.tree[1]
            while not node['is_leaf']:
                path.append(node["node_id"] - 1)
                if row[node['feature']] < node['threshold']:
                    node = self.tree[node['left_child_id']]
                else:
                    node = self.tree[node['right_child_id']]
            path.append(node["node_id"] - 1)
            path_all[idx][path] = 1
        return path_all

    def decision_function(self, x):
        
        check_is_fitted(self, "tree")
            
        leaf_idx = []
        for row in x:
            node = self.tree[1]
            while not node['is_leaf']:
                if row[node['feature']] < node['threshold']:
                    node = self.tree[node['left_child_id']]
                else:
                    node = self.tree[node['right_child_id']]
            leaf_idx.append(node['node_id'])
        
        n_samples = x.shape[0]
        pred = np.zeros((n_samples))
        for node_id in np.unique(leaf_idx):
            sample_indice = np.array(leaf_idx) == node_id
            pred[sample_indice] = self.tree[node_id]['predict_func'](x[sample_indice, :]).ravel()
        return pred
                
    def plot_tree(self, folder="./results/", name="demo", save_png=False, save_eps=False):

        draw_tree = copy.deepcopy(self.tree)
        pending_node_list = [draw_tree[1]]
        max_depth = np.max([item["depth"] for key, item in self.tree.items()])
        while len(pending_node_list) > 0:

            item = pending_node_list.pop()
            if item["parent_id"] is None:
                xy = (0.5, 0)
                parent_xy = None
            else:
                parent_xy = draw_tree[item["parent_id"]]["xy"]
                if item["is_left"]:
                    xy = (parent_xy[0] - 1 / 2 ** (item["depth"] + 1), 3 * item["depth"] / (3 * max_depth + 2))
                else:
                    xy = (parent_xy[0] + 1 / 2 ** (item["depth"] + 1), 3 * item["depth"] / (3 * max_depth + 2))

            if item["is_leaf"]:
                draw_tree[item["node_id"]].update({"xy": xy, 
                                      "parent_xy": parent_xy,
                                      "estimator":item["estimator"],
                                      "label": "impurity = " + str(np.round(item["impurity"], 3)) 
                                             + "\nsamples = " + str(int(item["n_samples"]))
                                             + "\nvalue = " + str(np.round(item["value"], 3))})
            else:
                draw_tree[item["node_id"]].update({"xy": xy,
                                       "parent_xy": parent_xy,
                                       "label": "X[" + str(item["feature"]) + "] <=" + str(np.round(item["threshold"], 3)) 
                                            + "\nimpurity = " + str(np.round(item["impurity"], 3)) 
                                            + "\nsamples = " + str(int(item["n_samples"])) 
                                            + "\nvalue = " + str(np.round(item["value"], 3))})
                pending_node_list.append(self.tree[item["left_child_id"]])
                pending_node_list.append(self.tree[item["right_child_id"]])


        fig = plt.figure(figsize=(2 ** (max_depth + 1), max_depth * 2))
        tree = fig.add_axes([0.0, 0.0, 1, 1])
        ax_width = tree.get_window_extent().width
        ax_height = tree.get_window_extent().height

        color_list = [229, 129, 57]
        values = np.array([item["value"] for key, item in self.tree.items()])
        min_value, max_value = values.min(), values.max()

        for key, item in draw_tree.items():

            alpha = (item["value"] - min_value) / (max_value - min_value)
            color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color_list]
            kwargs = dict(bbox={"fc": '#%2x%2x%2x' % tuple(color), "boxstyle": "round"}, arrowprops={"arrowstyle":"<-"}, 
                          ha='center', va='center', zorder=100 - 10 * item["depth"], xycoords='axes pixels', fontsize=14)

            if item["parent_id"] is None:
                tree.annotate(item["label"], (item["xy"][0] * ax_width, (1 - item["xy"][1]) * ax_height), **kwargs)
            else:
                if item["is_left"]:
                    tree.annotate(item["label"], ((item["parent_xy"][0] - 0.01 / 2 ** (item["depth"] + 1)) * ax_width,
                                         (1 - item["parent_xy"][1] - 0.1 / max_depth) * ax_height),
                                        (item["xy"][0] * ax_width, (1 - item["xy"][1]) * ax_height), **kwargs)
                else:
                    tree.annotate(item["label"], ((item["parent_xy"][0] + 0.01 / 2 ** (item["depth"] + 1)) * ax_width,
                                         (1 - item["parent_xy"][1] - 0.1 / max_depth) * ax_height),
                                        (item["xy"][0] * ax_width, (1 - item["xy"][1]) * ax_height), **kwargs)

            if item["is_leaf"] and (self.base_method == "sim"):
                sim = item["estimator"]
                
                if sim is None:
                    leaf_constant = fig.add_axes([item["xy"][0] - 0.45 / (2 ** max_depth),
                                       1 - 3 * max_depth / (3 * max_depth + 2) - 1 / max_depth,
                                       0.85 / (2 ** max_depth), 0.6 / max_depth])
                    leaf_constant.axhline(item["value"])
                    leaf_constant.set_ylim(item["value"] - 0.5, item["value"] + 0.5)
                    leaf_constant.axes.get_xaxis().set_ticks([])
                    leaf_constant.axes.get_yaxis().set_ticks([])
                else:
                    leaf_ridge = fig.add_axes([item["xy"][0] - 0.45 / (2 ** max_depth),
                                       1 - 3 * max_depth / (3 * max_depth + 2) - 1 / max_depth,
                                       0.85 / (2 ** max_depth), 0.6 / max_depth])
                    xgrid = np.linspace(sim.shape_fit_.xmin, sim.shape_fit_.xmax, 100).reshape([-1, 1])
                    ygrid = sim.shape_fit_.decision_function(xgrid)
                    leaf_ridge.plot(xgrid, ygrid)
                    leaf_ridge.axes.get_xaxis().set_ticks([])
                    leaf_ridge.axes.get_yaxis().set_ticks([])

                    betas = np.hstack([item["estimator"].beta_ for key, item in draw_tree.items() 
                                       if item["is_leaf"] and (item["estimator"] is not None)])
                    xlim_min = - max(np.abs(betas.min() - 0.1), np.abs(betas.max() + 0.1))
                    xlim_max = max(np.abs(betas.min() - 0.1), np.abs(betas.max() + 0.1))

                    leaf_proj = fig.add_axes([item["xy"][0] - 0.45 / (2 ** max_depth),
                                      1 - 3 * max_depth / (3 * max_depth + 2) - 1.5 / max_depth,
                                     0.85 / (2 ** max_depth), 0.4 / max_depth])
                    leaf_proj.bar(np.arange(len(sim.beta_)), sim.beta_.ravel())
                    leaf_proj.bar(np.arange(len(sim.beta_)), sim.beta_.ravel())
                    leaf_proj.set_xticks(np.arange(len(sim.beta_)))
                    leaf_proj.set_xticklabels(["X" + str(idx + 1) for idx in range(len(sim.beta_.ravel()))])
                    leaf_proj.set_ylim(xlim_min, xlim_max)
                    leaf_proj.axes.get_yaxis().set_ticks([])
        tree.set_axis_off()
        plt.show()
        if max_depth > 0:
            save_path = folder + name
            if save_eps:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig("%s.eps" % save_path, bbox_inches="tight", dpi=100)
            if save_png:
                if not os.path.exists(folder):
                    os.makedirs(folder)
                fig.savefig("%s.png" % save_path, bbox_inches="tight", dpi=100)


class DTSimRegressor(BaseDTSim, ClassifierMixin):
    
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, split_method="constant", base_method="constant", n_split_grid=10,
             degree=2, knot_num=10, reg_lambda=0.1, reg_gamma=10, random_state=0):

        super(DTSimRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 base_method=base_method,
                                 n_split_grid=n_split_grid,
                                 split_method=split_method,
                                 degree=degree,
                                 knot_num=knot_num,
                                 reg_lambda=reg_lambda,
                                 reg_gamma=reg_gamma,
                                 random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()
    
    def build_root(self):
        
        if self.split_method == "constant":
            root_impurity = self.y.var()
        elif self.split_method == "sim":
            root_clf = SimRegressor(method='first_order_thres', degree=self.degree, reg_lambda=0, reg_gamma=0,
                                     knot_num=self.knot_num, random_state=self.random_state)
            root_clf.fit(self.x, self.y)
            root_impurity = mean_squared_error(self.y, root_clf.predict(self.x))
        elif self.split_method == "glm":
            root_clf = LinearRegression()
            root_clf.fit(self.x, self.y)
            root_impurity = mean_squared_error(self.y, root_clf.predict(self.x))

        return root_impurity

    def build_leaf(self, sample_indice):
        
        estimator = None
        n_samples = len(sample_indice)
        if self.base_method == "constant":
            predict_func = lambda x: np.mean(self.y[sample_indice])
        elif self.base_method == "sim":
            best_impurity = np.inf
            for reg_lambda in self.reg_lambda_list:
                for reg_gamma in self.reg_gamma_list:
                    estimator = SimRegressor(method='first_order_thres', degree=self.degree,
                             reg_lambda=reg_lambda, reg_gamma=reg_gamma,
                             knot_num=self.knot_num, random_state=self.random_state)
                    estimator.fit(self.x[sample_indice], self.y[sample_indice])
                    current_impurity = mean_squared_error(self.y[sample_indice], estimator.predict(self.x[sample_indice]))
                    if current_impurity < best_impurity:
                        best_estimator = estimator
                        best_impurity = current_impurity
            predict_func = lambda x: best_estimator.predict(x)
        elif self.base_method == "glm":
            best_impurity = np.inf
            for alpha in (0.1, 1.0, 10.0):
                estimator = Ridge(alpha=alpha)
                estimator.fit(self.x[sample_indice], self.y[sample_indice])
                current_impurity = mean_squared_error(self.y[sample_indice], estimator.predict(self.x[sample_indice]))
                if current_impurity < best_impurity:
                    best_estimator = estimator
                    best_impurity = current_impurity
            predict_func = lambda x: best_estimator.predict(x)
        return predict_func, estimator
    
    def node_split_constant(self, sample_indice):
        
        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        best_impurity = np.inf
        best_feature = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        for feature_indice in range(n_features):

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < EPSILON:
                continue

            sum_left = 0
            sum_total = np.sum(node_y)
            sq_sum_total = np.sum(node_y ** 2)
            for i, _ in enumerate(sortted_indice):

                n_left = i + 1
                n_right = n_samples - i - 1
                sum_left += node_y[sortted_indice[i]]
                if i == (n_samples - 1):
                    continue

                if sortted_feature[i + 1] <= sortted_feature[i] + EPSILON:
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue

                current_impurity = (sq_sum_total / n_samples - (sum_left / n_left) ** 2 * n_left / n_samples -
                             ((sum_total - sum_left) / n_right) ** 2 * n_right / n_samples)

                if current_impurity < best_impurity:
                    best_position = i + 1
                    best_feature = feature_indice
                    best_impurity = current_impurity
                    best_threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

        sortted_indice = np.argsort(node_x[:, best_feature])
        best_left_indice = sample_indice[sortted_indice[:best_position]]
        best_right_indice = sample_indice[sortted_indice[best_position:]]
        best_left_impurity = node_y[sortted_indice[:best_position]].var()
        best_right_impurity = node_y[sortted_indice[best_position:]].var()
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
    
        
    def node_split_glm(self, sample_indice):
        
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
        for feature_indice in range(n_features):

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
                estimator = LinearRegression()
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = mean_squared_error(node_y[left_indice], estimator.predict(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                estimator = LinearRegression()
                estimator.fit(node_x[right_indice], node_y[right_indice])
                right_impurity = mean_squared_error(node_y[right_indice], estimator.predict(node_x[right_indice]))

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
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node

    
    def node_split_sim(self, sample_indice):
        
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
        for feature_indice in range(n_features):

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
                estimator = SimRegressor(method='first_order_thres', degree=self.degree,
                                 reg_lambda=0, reg_gamma=0,
                                 knot_num=self.knot_num, random_state=self.random_state)
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = mean_squared_error(node_y[left_indice], estimator.predict(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                estimator = SimRegressor(method='first_order_thres', degree=self.degree,
                                 reg_lambda=0, reg_gamma=0,
                                 knot_num=self.knot_num, random_state=self.random_state)
                estimator.fit(node_x[right_indice], node_y[right_indice])
                right_impurity = mean_squared_error(node_y[right_indice], estimator.predict(node_x[right_indice]))

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
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
    
        
    def predict(self, x):
        return self.decision_function(x)

    
class DTSimClassifier(BaseDTSim, ClassifierMixin):
    
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, split_method="constant", base_method="constant", n_split_grid=10,
                 degree=2, knot_num=10, reg_lambda=0.1, reg_gamma=10, random_state=0):

        super(DTSimClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 base_method=base_method,
                                 n_split_grid=n_split_grid,
                                 split_method=split_method,
                                 degree=degree,
                                 knot_num=knot_num,
                                 reg_lambda=reg_lambda,
                                 reg_gamma=reg_gamma,
                                 random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)

        self._label_binarizer = LabelBinarizer()
        self._label_binarizer.fit(y)
        self.classes_ = self._label_binarizer.classes_

        y = self._label_binarizer.transform(y) * 1.0
        return x, y.ravel()
    
    def build_root(self):
        
        if self.split_method == "constant":
            p = self.y.mean()
            root_impurity = - p * np.log2(p) - (1 - p) * np.log2((1 - p)) if (p > 0) and (p < 1) else 0 
        elif self.split_method == "sim":
            root_clf = SimClassifier(method='first_order_thres', degree=self.degree, reg_lambda=0, reg_gamma=0,
                                     knot_num=self.knot_num, random_state=self.random_state)
            root_clf.fit(self.x, self.y)
            root_impurity = log_loss(self.y, root_clf.predict_proba(self.x))
        elif self.split_method == "glm":
            root_clf = LogisticRegression(penalty='none', random_state=self.random_state)
            root_clf.fit(self.x, self.y.ravel())
            root_impurity = log_loss(self.y, root_clf.predict_proba(self.x)[:, 1])
        return root_impurity

    def build_leaf(self, sample_indice):
        
        estimator = None
        n_samples = len(sample_indice)
        if self.base_method == "constant":
            predict_func = lambda x: np.mean(self.y[sample_indice])
        elif self.base_method == "sim":
            if self.y[sample_indice].std() == 0:
                predict_func = lambda x: np.mean(self.y[sample_indice])
            else:
                best_impurity = np.inf
                for reg_lambda in self.reg_lambda_list:
                    for reg_gamma in self.reg_gamma_list:
                        estimator = SimClassifier(method='first_order_thres', degree=self.degree,
                                 reg_lambda=reg_lambda, reg_gamma=reg_gamma,
                                 knot_num=self.knot_num, random_state=self.random_state)
                        estimator.fit(self.x[sample_indice], self.y[sample_indice])
                        current_impurity = log_loss(self.y[sample_indice], estimator.predict_proba(self.x[sample_indice]))
                        if current_impurity < best_impurity:
                            best_estimator = estimator
                            best_impurity = current_impurity
                predict_func = lambda x: best_estimator.predict_proba(x)
        elif self.base_method == "glm":
            if self.y[sample_indice].std() == 0:
                predict_func = lambda x: np.mean(self.y[sample_indice])
            else:
                best_impurity = np.inf
                for alpha in (0.1, 1.0, 10.0):
                    estimator = LogisticRegression(C=alpha)
                    estimator.fit(self.x[sample_indice], self.y[sample_indice])
                    current_impurity = log_loss(self.y[sample_indice], estimator.predict(self.x[sample_indice]))
                    if current_impurity < best_impurity:
                        best_estimator = estimator
                        best_impurity = current_impurity
                predict_func = lambda x: best_estimator.predict_proba(x)[:, 1]
        return predict_func, estimator
    
    def node_split_constant(self, sample_indice):
        
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
        for feature_indice in range(n_features):

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < EPSILON:
                continue

            sum_left = 0
            sum_total = np.sum(node_y)
            for i, _ in enumerate(sortted_indice):

                n_left = i + 1
                n_right = n_samples - i - 1
                sum_left += node_y[sortted_indice[i]]
                if i == (n_samples - 1):
                    continue

                if sortted_feature[i + 1] <= sortted_feature[i] + EPSILON:
                    continue

                if ((i + 1) < self.min_samples_leaf) or ((n_samples - i - 1) < self.min_samples_leaf):
                    continue

                left_impurity = 0 
                right_impurity = 0
                pleft = sum_left / n_left
                pright = (sum_total - sum_left) / n_right
                if (pleft > 0) and (pleft < 1):
                    left_impurity = (- pleft * np.log2(pleft) - (1 - pleft) * np.log2((1 - pleft)))
                if (pright > 0) and (pright < 1):
                    right_impurity = (- pright * np.log2(pright) - (1 - pright) * np.log2((1 - pright)))
                current_impurity = (n_left / n_samples * left_impurity + n_right / n_samples * right_impurity)

                if current_impurity < best_impurity:
                    best_position = i + 1
                    best_feature = feature_indice
                    best_impurity = current_impurity
                    best_threshold = (sortted_feature[i] + sortted_feature[i + 1]) / 2

        if best_position is not None:
            sortted_indice = np.argsort(node_x[:, best_feature])
            best_left_indice = sample_indice[sortted_indice[:best_position]]
            best_right_indice = sample_indice[sortted_indice[best_position:]]
            
            pleft = node_y[sortted_indice[:best_position]].mean()
            pright = node_y[sortted_indice[best_position:]].mean()
            best_left_impurity = - pleft * np.log2(pleft) - (1 - pleft) * np.log2((1 - pleft)) if (pleft > 0) and (pleft < 1) else 0 
            best_right_impurity = - pright * np.log2(pright) - (1 - pright) * np.log2((1 - pright)) if (pright > 0) and (pright < 1) else 0 
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
             "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
    
    def node_split_glm(self, sample_indice):
        
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
        for feature_indice in range(n_features):

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
                if node_y[left_indice].std() == 0:
                    left_impurity = 0
                else:
                    left_clf = LogisticRegression(penalty='none', random_state=self.random_state)
                    left_clf.fit(node_x[left_indice], node_y[left_indice].ravel())
                    left_impurity = log_loss(node_y[left_indice], left_clf.predict_proba(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                if node_y[right_indice].std() == 0:
                    right_impurity = 0
                else:
                    right_clf = LogisticRegression(penalty='none', random_state=self.random_state)
                    right_clf.fit(node_x[right_indice], node_y[right_indice].ravel())
                    right_impurity = log_loss(node_y[right_indice], right_clf.predict_proba(node_x[right_indice]))
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
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node

    def node_split_sim(self, sample_indice):
        
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
        for feature_indice in range(n_features):

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

                if (i - self.min_samples_leaf) < 1 / self.n_split_grid * (split_point + 1) * (n_samples - 2 * self.min_samples_leaf):
                    continue

                split_point += 1
                left_indice = sortted_indice[:(i + 1)]
                if node_y[left_indice].std() == 0:
                    left_impurity = 0
                else:
                    left_clf = SimClassifier(method='first_order_thres', degree=self.degree, reg_lambda=0, reg_gamma=0,
                                     knot_num=self.knot_num, random_state=self.random_state)
                    left_clf.fit(node_x[left_indice], node_y[left_indice])
                    left_impurity = log_loss(node_y[left_indice], left_clf.predict_proba(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                if node_y[right_indice].std() == 0:
                    right_impurity = 0
                else:
                    right_clf = SimClassifier(method='first_order_thres', degree=self.degree, reg_lambda=0, reg_gamma=0,
                                      knot_num=self.knot_num, random_state=self.random_state)
                    right_clf.fit(node_x[right_indice], node_y[right_indice])
                    right_impurity = log_loss(node_y[right_indice], right_clf.predict_proba(node_x[right_indice]))
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
        node = {"feature":best_feature, "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
    
    def predict_proba(self, x):
        return self.decision_function(x)
    
    def predict(self, x):
        pred_proba = self.predict_proba(x)
        return self._label_binarizer.inverse_transform(pred_proba)