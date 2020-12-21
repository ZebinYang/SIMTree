import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.base import RegressorMixin, ClassifierMixin

from .mobtree import MoBTreeRegressor, MoBTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


__all__ = ["GLMTreeRegressor", "GLMTreeClassifier"]


class GLMTreeRegressor(MoBTreeRegressor, RegressorMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=5, n_feature_search=5, n_split_grid=20, random_state=0):

        super(GLMTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.base_estimator = LinearRegression()

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        best_estimator = LassoCV(n_alphas=10, cv=5, normalize=True)
        best_estimator.fit(self.x[sample_indice], self.y[sample_indice])
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity


class GLMTreeClassifier(MoBTreeClassifier, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=5, n_feature_search=5, n_split_grid=20, random_state=0):

        super(GLMTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.base_estimator = LogisticRegression(penalty='none', random_state=self.random_state)

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
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
