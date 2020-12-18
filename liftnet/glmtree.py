import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.base import RegressorMixin, ClassifierMixin

from .mobtree import BaseMoBTreeRegressor, BaseMoBTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


__all__ = ["GLMTreeRegressor", "GLMTreeClassifier"]


class GLMTreeRegressor(BaseMoBTreeRegressor, RegressorMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 n_split_grid=10, split_features=None, random_state=0):

        super(GLMTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 random_state=random_state)
        self.n_split_grid = n_split_grid

    def build_root(self):

        root_clf = LinearRegression()
        root_clf.fit(self.x, self.y)
        root_impurity = self.get_loss(self.y, root_clf.predict(self.x))
        return root_impurity

    def build_leaf(self, sample_indice):

        best_estimator = LassoCV(n_alphas=10, cv=5)
        best_estimator.fit(self.x[sample_indice], self.y[sample_indice])
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
        for feature_indice in self.split_features:

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
                estimator = LinearRegression()
                estimator.fit(node_x[left_indice], node_y[left_indice])
                left_impurity = self.get_loss(node_y[left_indice].ravel(), estimator.predict(node_x[left_indice]))

                right_indice = sortted_indice[(i + 1):]
                estimator = LinearRegression()
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


class GLMTreeClassifier(BaseMoBTreeClassifier, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0,
                 n_split_grid=10, split_features=None, random_state=0):

        super(GLMTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 split_features=split_features,
                                 random_state=random_state)
        self.n_split_grid = n_split_grid

    def build_root(self):

        root_clf = LogisticRegression(penalty='none', random_state=self.random_state)
        root_clf.fit(self.x, self.y.ravel())
        root_impurity = self.get_loss(self.y, root_clf.predict_proba(self.x)[:, 1])
        return root_impurity

    def build_leaf(self, sample_indice):

        if (self.y[sample_indice].std() == 0) | (self.y[sample_indice].sum() < 5) | ((1 - self.y[sample_indice]).sum() < 5):
            best_impurity = 0
            best_estimator = None
            predict_func = lambda x: np.mean(self.y[sample_indice])
        else:
            best_estimator = LogisticRegressionCV(Cs=10, penalty="l1", solver="liblinear", cv=5, max_iter=1000)
            best_estimator.fit(self.x[sample_indice], self.y[sample_indice])
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
        for feature_indice in self.split_features:

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
                if node_y[left_indice].std() == 0:
                    left_impurity = 0
                else:
                    left_clf = LogisticRegression(penalty='none', random_state=self.random_state)
                    left_clf.fit(node_x[left_indice], node_y[left_indice].ravel())
                    left_impurity = self.get_loss(node_y[left_indice].ravel(), left_clf.predict_proba(node_x[left_indice])[:, 1])

                right_indice = sortted_indice[(i + 1):]
                if node_y[right_indice].std() == 0:
                    right_impurity = 0
                else:
                    right_clf = LogisticRegression(penalty='none', random_state=self.random_state)
                    right_clf.fit(node_x[right_indice], node_y[right_indice].ravel())
                    right_impurity = self.get_loss(node_y[right_indice].ravel(), right_clf.predict_proba(node_x[right_indice])[:, 1])
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
