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

    def __init__(self, max_depth=2, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=5, n_feature_search=10, n_split_grid=20, reg_lambda=0, random_state=0):

        super(GLMTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.reg_lambda = reg_lambda
        self.base_estimator = LinearRegression()

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        mx = self.x[sample_indice].mean(0)
        sx = self.x[sample_indice].std(0) + self.EPSILON
        nx = (self.x[sample_indice] - mx) / sx

        best_estimator = LassoCV(alphas=self.reg_lambda, cv=5, n_jobs=1, random_state=self.random_state)
        best_estimator.fit(nx, self.y[sample_indice])
        best_estimator.coef_ = best_estimator.coef_ / sx
        best_estimator.intercept_ = best_estimator.intercept_ - np.dot(mx, best_estimator.coef_.T)
        xmin = np.min(np.dot(self.x[sample_indice], best_estimator.coef_) + best_estimator.intercept_)
        xmax = np.max(np.dot(self.x[sample_indice], best_estimator.coef_) + best_estimator.intercept_)
        predict_func = lambda x: np.clip(best_estimator.predict(x), xmin, xmax)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity


class GLMTreeClassifier(MoBTreeClassifier, ClassifierMixin):

    def __init__(self, max_depth=2, min_samples_leaf=50, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=5, n_feature_search=10, n_split_grid=20, reg_lambda=0, random_state=0):

        super(GLMTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.reg_lambda = reg_lambda
        self.base_estimator = LogisticRegression(penalty='none', random_state=self.random_state)

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
            best_estimator = LogisticRegressionCV(Cs=self.reg_lambda, penalty="l1", solver="liblinear", scoring="roc_auc",
                                      cv=5, n_jobs=1, random_state=self.random_state)
            mx = self.x[sample_indice].mean(0)
            sx = self.x[sample_indice].std(0) + self.EPSILON
            nx = (self.x[sample_indice] - mx) / sx
            best_estimator.fit(nx, self.y[sample_indice])
            best_estimator.coef_ = best_estimator.coef_ / sx
            best_estimator.intercept_ = best_estimator.intercept_ - np.dot(mx, best_estimator.coef_.T)
            xmin = np.min(np.dot(self.x[sample_indice], best_estimator.coef_.ravel()))
            xmax = np.max(np.dot(self.x[sample_indice], best_estimator.coef_.ravel()))
            predict_func = lambda x: 1 / (1 + np.exp(- np.clip(np.dot(x, best_estimator.coef_.ravel()), xmin, xmax) - best_estimator.intercept_))
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity
