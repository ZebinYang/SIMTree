import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error
from sklearn.base import RegressorMixin, ClassifierMixin, is_regressor, is_classifier

from .mobtree import MoBTreeRegressor, MoBTreeClassifier

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

__all__ = ["CustomMobTreeRegressor", "CustomMobTreeClassifier"]


class CustomMobTreeRegressor(MoBTreeRegressor, RegressorMixin):

    def __init__(self, base_estimator, param_dict={}, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=5, n_feature_search=5, n_split_grid=20, random_state=0, **kargs):

        super(CustomMobTreeRegressor, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.param_dict = param_dict
        self.base_estimator = base_estimator
        if "random_state" in self.base_estimator.get_params().keys()
            self.base_estimator.set_params(**{"random_state": self.random_state})
        self.base_estimator.set_params(**kargs)

    def build_root(self):

        self.base_estimator.fit(self.x, self.y)
        root_impurity = self.evaluate_estimator(self.base_estimator, self.x, self.y.ravel())
        return root_impurity

    def build_leaf(self, sample_indice):

        grid = GridSearchCV(self.base_estimator, param_grid=self.param_dict,
                      scoring={"mse": make_scorer(mean_squared_error, greater_is_better=False)},
                      cv=5, refit="mse", n_jobs=1, error_score=np.nan)
        grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
        best_estimator = grid.best_estimator_
        predict_func = lambda x: best_estimator.predict(x)
        best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict(self.x[sample_indice]))
        return predict_func, best_estimator, best_impurity


class CustomMobTreeClassifier(MoBTreeClassifier, RegressorMixin):

    def __init__(self, base_estimator, param_dict={}, max_depth=2, min_samples_leaf=10, min_impurity_decrease=0, feature_names=None,
                 split_features=None, n_screen_grid=5, n_feature_search=5, n_split_grid=20, random_state=0, **kargs):

        super(CustomMobTreeClassifier, self).__init__(max_depth=max_depth,
                                 min_samples_leaf=min_samples_leaf,
                                 min_impurity_decrease=min_impurity_decrease,
                                 feature_names=feature_names,
                                 split_features=split_features,
                                 n_screen_grid=n_screen_grid,
                                 n_feature_search=n_feature_search,
                                 n_split_grid=n_split_grid,
                                 random_state=random_state)
        self.param_dict = param_dict
        self.base_estimator = base_estimator
        if "random_state" in self.base_estimator.get_params().keys()
            self.base_estimator.set_params(**{"random_state": self.random_state})
        self.base_estimator.set_params(**kargs)

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
            grid = GridSearchCV(self.base_estimator, param_grid=self.param_dict,
                          scoring={"auc": make_scorer(roc_auc_score, needs_proba=True)},
                          cv=5, refit="auc", n_jobs=1, error_score=np.nan)
            grid.fit(self.x[sample_indice], self.y[sample_indice].ravel())
            best_estimator = grid.best_estimator_
            predict_func = lambda x: best_estimator.predict_proba(x)[:, 1]
            best_impurity = self.get_loss(self.y[sample_indice], best_estimator.predict_proba(self.x[sample_indice])[:, 1])
        return predict_func, best_estimator, best_impurity
