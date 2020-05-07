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

from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


try:
    fps = importr("fps")
except:
    devtools = importr("devtools")
    devtools.install_github("https://github.com/vqv/fps")
    fps = importr("fps")
    
numpy2ri.activate()


from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

EPSILON = 1e-7


def first_order(x, y):

    reg_lambda = 0.0
    mu = np.average(x, axis=0) 
    cov = np.cov(x.T)
    inv_cov = np.linalg.pinv(cov)
    s1 = np.dot(inv_cov, (x - mu).T).T
    zbar = np.average(y.reshape(-1, 1) * s1, axis=0)
    sigmat = np.dot(zbar.reshape([-1, 1]), zbar.reshape([-1, 1]).T)

    reg_lambda_max = np.max(np.abs(sigmat) - np.abs(sigmat) * np.eye(sigmat.shape[0]), axis=0).max()
    spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(reg_lambda * reg_lambda_max))
    beta = np.array(fps.coef_fps(spca_solver, reg_lambda * reg_lambda_max))
    return beta

def second_order(x, y):

    n_samples, n_features = x.shape
    mu = np.average(x, axis=0) 
    cov = np.cov(x.T)
    inv_cov = np.linalg.pinv(cov)
    s1 = np.dot(inv_cov, (x - self.mu).T).T
    sigmat = np.tensordot(s1 * y.reshape([-1, 1]), s1, axes=([0], [0]))
    sigmat -= np.average(y, axis=0) * inv_cov

    reg_lambda_max = np.max(np.abs(sigmat) - np.abs(sigmat) * np.eye(sigmat.shape[0]), axis=0).max()
    spca_solver = fps.fps(sigmat, 1, 1, -1, -1, ro.r.c(reg_lambda * reg_lambda_max))
    beta = np.array(fps.coef_fps(spca_solver, reg_lambda * reg_lambda_max))
    return beta


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
        if self.split_features is None:
            self.split_features = np.arange(n_features).tolist()
        
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
                if np.dot(row, node['feature']) < node['threshold']:
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
                if np.dot(row, node['feature']) < node['threshold']:
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
        
        if self.split_method == "sim":
            root_clf = SimRegressor(method='first_order', degree=self.degree, reg_lambda=0, reg_gamma=0,
                                     knot_num=self.knot_num, random_state=self.random_state)
            root_clf.fit(self.x, self.y)
            root_impurity = mean_squared_error(self.y, root_clf.predict(self.x))

        return root_impurity

    def build_leaf(self, sample_indice):
        
        estimator = None
        n_samples = len(sample_indice)
        if self.base_method == "sim":
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
        return predict_func, estimator
    
    def node_split_sim(self, sample_indice):
        
        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        best_position = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        best_impurity = np.inf
        best_left_impurity = np.inf
        best_right_impurity = np.inf
        
        beta = first_order(node_x, node_y)
        current_feature = np.dot(node_x, beta[:, 1])
        sortted_indice = np.argsort(current_feature)
        sortted_feature = current_feature[sortted_indice]
        feature_range = sortted_feature[-1] - sortted_feature[0]

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
            estimator = SimRegressor(method='first_order', degree=self.degree,
                             reg_lambda=0, reg_gamma=0,
                             knot_num=self.knot_num, random_state=self.random_state)
            estimator.fit(node_x[left_indice], node_y[left_indice])
            left_impurity = mean_squared_error(node_y[left_indice], estimator.predict(node_x[left_indice]))

            right_indice = sortted_indice[(i + 1):]
            estimator = SimRegressor(method='first_order', degree=self.degree,
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
        node = {"feature":beta[:, 1], "threshold":best_threshold, "left":best_left_indice, "right":best_right_indice,
              "impurity":best_impurity, "left_impurity":best_left_impurity, "right_impurity":best_right_impurity}
        return node
    
        
    def predict(self, x):
        return self.decision_function(x)
