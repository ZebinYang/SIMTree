import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from .mobtree import MoBTreeRegressor, MoBTreeClassifier


__all__ = ["CARTRegressor", "CARTClassifier"]


class CARTRegressor(MoBTreeRegressor, RegressorMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 split_features=None, feature_names=None, random_state=0):

        super(CARTRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 feature_names=feature_names,
                                 random_state=random_state)

    def build_root(self):

        root_impurity = self.y.var()
        return root_impurity

    def build_leaf(self, sample_indice):

        best_estimator = None
        predict_func = lambda x: np.mean(self.y[sample_indice])
        best_impurity = self.y[sample_indice].var()
        return predict_func, best_estimator, best_impurity

    def node_split(self, sample_indice):

        node_x = self.x[sample_indice]
        node_y = self.y[sample_indice]
        n_samples, n_features = node_x.shape

        best_impurity = np.inf
        best_feature = None
        best_threshold = None
        best_left_indice = None
        best_right_indice = None
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < self.EPSILON:
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
        node = {"feature": best_feature, "threshold": best_threshold, "left": best_left_indice, "right": best_right_indice,
              "impurity": best_impurity, "left_impurity": best_left_impurity, "right_impurity": best_right_impurity}
        return node


class CARTClassifier(MoBTreeClassifier, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 split_features=None, feature_names=None, random_state=0):

        super(CARTClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 feature_names=feature_names,
                                 random_state=random_state)

    def build_root(self):

        p = self.y.mean()
        root_impurity = - p * np.log2(p) - (1 - p) * np.log2((1 - p)) if (p > 0) and (p < 1) else 0
        return root_impurity

    def build_leaf(self, sample_indice):

        best_estimator = None
        predict_func = lambda x: np.ones(len(sample_indice)) * self.y[sample_indice].mean()
        best_impurity = self.get_loss(self.y[sample_indice], predict_func(self.x[sample_indice]))
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
        for feature_indice in self.split_features:

            current_feature = node_x[:, feature_indice]
            sortted_indice = np.argsort(current_feature)
            sortted_feature = current_feature[sortted_indice]
            feature_range = sortted_feature[-1] - sortted_feature[0]
            if feature_range < self.EPSILON:
                continue

            sum_left = 0
            sum_total = np.sum(node_y)
            for i, _ in enumerate(sortted_indice):

                n_left = i + 1
                n_right = n_samples - i - 1
                sum_left += node_y[sortted_indice[i]]
                if i == (n_samples - 1):
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
        node = {"feature": best_feature, "threshold": best_threshold, "left": best_left_indice, "right": best_right_indice,
             "impurity": best_impurity, "left_impurity": best_left_impurity, "right_impurity": best_right_impurity}
        return node
