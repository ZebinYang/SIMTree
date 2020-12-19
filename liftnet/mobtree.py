import os
import copy
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_X_y, column_or_1d
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, is_classifier, is_regressor


__all__ = ["BaseMoBTreeRegressor", "BaseMoBTreeClassifier"]


class BaseMoBTree(BaseEstimator, metaclass=ABCMeta):
    """
        Base class for classification and regression.
     """

    @abstractmethod
    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 split_features=None, feature_names=None, random_state=0):

        self.max_depth = max_depth
        self.split_features = split_features
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.feature_names = feature_names

        self.EPSILON = 1e-7
        self.random_state = random_state

    def _validate_hyperparameters(self):

        if not isinstance(self.max_depth, int):
            raise ValueError("degree must be an integer, got %s." % self.max_depth)
            if self.max_depth < 0:
                raise ValueError("degree must be >= 0, got %s." % self.max_depth)

        if self.split_features is not None:
            if not isinstance(self.split_features, list):
                raise ValueError("split_features must be an list or None, got %s." %
                         self.split_features)

        if not isinstance(self.min_samples_leaf, int):
            raise ValueError("min_samples_leaf must be an integer, got %s." % self.min_samples_leaf)

            if self.min_samples_leaf < 0:
                raise ValueError("min_samples_leaf must be >= 0, got %s." % self.min_samples_leaf)

        if self.min_impurity_decrease < 0.:
            raise ValueError("min_impurity_decrease must be >= 0, got %s." % self.min_impurity_decrease)
        
        if self.feature_names is not None:
            self.feature_names = list(self.feature_names)
            if len(self.feature_names) != self.n_features:
                raise ValueError("feature_names must have the same length as n_features, got %s." % self.feature_names)
        else:
            self.feature_names = ["x" + str(i + 1) for i in range(self.n_features)]

    @abstractmethod
    def build_root(self):
        pass

    @abstractmethod
    def build_leaf(self, sample_indice):
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
            predict_func, estimator, best_impurity = self.build_leaf(sample_indice)
            node = {"node_id": node_id, "parent_id": parent_id, "depth": depth, "feature": feature, "impurity": best_impurity,
                  "n_samples": n_samples, "is_left": is_left, "is_leaf": is_leaf, "value": np.mean(self.y[sample_indice]),
                  "predict_func": predict_func, "estimator":estimator}
            self.leaf_estimators_.update({node_id: estimator})
        else:
            node = {"node_id": node_id, "parent_id": parent_id, "depth": depth,"feature": feature, "impurity": impurity,
                  "n_samples": n_samples, "is_left": is_left, "is_leaf": is_leaf, "value": np.mean(self.y[sample_indice]),
                  "left_child_id":None, "right_child_id": None, "threshold": threshold}            
        self.tree.update({node_id: node})
        return node_id

    def fit(self, x, y):

        self.tree = {}
        self.node_count = 0
        self.leaf_estimators_ = {}
        self._validate_hyperparameters()
        self.x, self.y = self._validate_input(x, y)
        n_samples, n_features = self.x.shape
        sample_indice = np.arange(n_samples)
        if self.split_features is None:
            self.split_features = np.arange(n_features).tolist()

        np.random.seed(self.random_state)
        root_impurity = self.build_root()
        root_node = {"sample_indice": sample_indice,
                 "parent_id": None,
                 "depth": 0,
                 "impurity": root_impurity,
                 "is_left": False}
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

                pending_node_list.append({"sample_indice": split["left"],
                                 "parent_id": node_id,
                                 "depth": depth + 1,
                                 "impurity": split["left_impurity"],
                                 "is_left": True})
                pending_node_list.append({"sample_indice": split["right"],
                                 "parent_id": node_id,
                                 "depth": depth + 1,
                                 "impurity": split["right_impurity"],
                                 "is_left": False})
        return self

    def plot_tree(self, folder="./results/", name="demo", save_png=False, save_eps=False):

        draw_tree = copy.deepcopy(self.tree)
        pending_node_list = [draw_tree[1]]
        max_depth = 1 + np.max([item["depth"] for key, item in self.tree.items()])
        while len(pending_node_list) > 0:

            item = pending_node_list.pop()
            if item["parent_id"] is None:
                xy = (0.5, 0)
                parent_xy = None
            else:
                parent_xy = draw_tree[item["parent_id"]]["xy"]
                if item["is_left"]:
                    xy = (parent_xy[0] - 1 / 2 ** (item["depth"] + 1), 3 * item["depth"] / (3 * max_depth - 2))
                else:
                    xy = (parent_xy[0] + 1 / 2 ** (item["depth"] + 1), 3 * item["depth"] / (3 * max_depth - 2))

            if item["is_leaf"]:
                if is_regressor(self):
                    draw_tree[item["node_id"]].update({"xy": xy,
                                          "parent_xy": parent_xy,
                                          "estimator": item["estimator"],
                                          "label": "____Node " + str(item["node_id"]) + "____" +
                                                "\nMSE: " + str(np.round(item["impurity"], 3))
                                                 + "\nSize: " + str(int(item["n_samples"]))
                                                 + "\nMean: " + str(np.round(item["value"], 3))})
                elif is_classifier(self):
                    draw_tree[item["node_id"]].update({"xy": xy,
                                          "parent_xy": parent_xy,
                                          "estimator": item["estimator"],
                                          "label": "____Node " + str(item["node_id"]) + "____" +
                                                "\nCEntropy: " + str(np.round(item["impurity"], 3))
                                                 + "\nSize: " + str(int(item["n_samples"]))
                                                 + "\nMean: " + str(np.round(item["value"], 3))})
            else:
                if is_regressor(self):
                    
                    
                    draw_tree[item["node_id"]].update({"xy": xy,
                                           "parent_xy": parent_xy,
                                           "label": "____Node " + str(item["node_id"]) + "____"
                                        + "\n" + self.feature_names[item["feature"]] + " <=" + str(np.round(item["threshold"], 3))
                                        + "\nMSE: " + str(np.round(item["impurity"], 3))
                                        + "\nSize: " + str(int(item["n_samples"]))
                                        + "\nMean: " + str(np.round(item["value"], 3))})
                elif is_classifier(self):
                    draw_tree[item["node_id"]].update({"xy": xy,
                                           "parent_xy": parent_xy,
                                           "label": "____Node " + str(item["node_id"]) + "____" +
                                        + "\n" + self.feature_names[item["feature"]] + " <=" + str(np.round(item["threshold"], 3))
                                        + "\nCEntropy: " + str(np.round(item["impurity"], 3))
                                        + "\nSize: " + str(int(item["n_samples"]))
                                        + "\nMean: " + str(np.round(item["value"], 3))})

                pending_node_list.append(self.tree[item["left_child_id"]])
                pending_node_list.append(self.tree[item["right_child_id"]])

        fig = plt.figure(figsize=(2 ** max_depth, (max_depth - 0.8) * 2))
        tree = fig.add_axes([0.0, 0.0, 1, 1])
        ax_width = tree.get_window_extent().width
        ax_height = tree.get_window_extent().height

        color_list = [229, 129, 57]
        color_leaf_list = [57, 157, 229]
        values = np.array([item["value"] for key, item in self.tree.items()])
        min_value, max_value = values.min(), values.max()

        for key, item in draw_tree.items():

            if max_value == min_value:
                if item["is_leaf"]:
                    color = color_leaf_list
                else:
                    color = color_list
            else:
                alpha = (item["value"] - min_value) / (max_value - min_value)
                alpha = np.clip(alpha, 0.1, 0.9)
                if item["is_leaf"]:
                    color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color_leaf_list]
                else:
                    color = [int(round(alpha * c + (1 - alpha) * 255, 0)) for c in color_list]

            kwargs = dict(bbox={"fc": '#%2x%2x%2x' % tuple(color), "boxstyle": "round"}, arrowprops={"arrowstyle": "<-"},
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
                fig.savefig("%s.png" % save_path, bbox_inches="tight")

    def decision_path(self, x):

        n_samples = x.shape[0]
        path_all = np.zeros((n_samples, self.node_count))
        for idx, row in enumerate(x):
            path = []
            node = self.tree[1]
            while not node['is_leaf']:
                path.append(node["node_id"] - 1)
                if row[node['feature']] <= node['threshold']:
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
                if row[node['feature']] <= node['threshold']:
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


class BaseMoBTreeRegressor(BaseMoBTree, RegressorMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 split_features=None, feature_names=None, random_state=0):

        super(BaseMoBTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 feature_names=feature_names,
                                 random_state=random_state)

    def _validate_input(self, x, y):
        x, y = check_X_y(x, y, accept_sparse=["csr", "csc", "coo"],
                         multi_output=True, y_numeric=True)
        return x, y.ravel()

    def get_loss(self, label, pred):

        """method to calculate the MSE loss

        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        Returns
        -------
        float
            the MSE loss
        """
        loss = np.average((label - pred) ** 2, axis=0)
        return loss

    def predict(self, x):
        return self.decision_function(x)

    
class BaseMoBTreeClassifier(BaseMoBTree, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 split_features=None, feature_names=None, random_state=0):

        super(BaseMoBTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 feature_names=feature_names,
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

    def get_loss(self, label, pred):

        """method to calculate the cross entropy loss

        Parameters
        ---------
        label : array-like of shape (n_samples,)
            containing the input dataset
        pred : array-like of shape (n_samples,)
            containing the output dataset
        sample_weight : array-like of shape (n_samples,), optional
            containing sample weights
        Returns
        -------
        float
            the cross entropy loss
        """

        with np.errstate(divide="ignore", over="ignore"):
            pred = np.clip(pred, self.EPSILON, 1. - self.EPSILON)
            loss = - np.average(label * np.log(pred) + (1 - label) * np.log(1 - pred), axis=0)
        return loss

    def predict_proba(self, x):
        proba = self.decision_function(x).reshape(-1, 1)
        return np.hstack([1 - proba, proba])

    def predict(self, x):
        pred_proba = self.decision_function(x)
        return self._label_binarizer.inverse_transform(pred_proba)
