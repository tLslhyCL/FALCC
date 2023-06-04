import math
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#from bayes_opt import BayesianOptimization
from algorithm.codes import DiversityMeasures
from sklearn.metrics import make_scorer


class HyperOptimizedLearner:
    """A class for hyperparameter optimization of ensemble learning algorithms.

    This class takes a specified ensemble learning algorithm, performs hyperparameter optimization
    using a chosen search method, and returns the best estimator based on ensemble entropy.

    Parameters
    ----------
    learner : str, {'RandomForest', 'AdaBoost'}
        The ensemble learning algorithm to optimize.
    search_method : str, {'random', 'full', 'bayes_opt'}
        The search method for hyperparameter optimization.
    input_file : str
        The name of the dataset used for training and evaluation.
    sbt : bool, optional, default: False
        Indicates whether 'sbt' was used and includes it in the output file name.
    n_iter : int, optional, default: 25
        Number of iterations for random search or Bayesian optimization.
    cv : int, optional, default: 5
        Number of cross-validation folds.
    ensemble_entropy : float in [0, 1], str {'mean', 'middle', 'quarter', 'three_quarters'}, optional, default: 1
        The desired ensemble entropy. If a float, returns the estimator with entropy closest to the specified value.
        If 'mean', 'middle', 'quarter', or 'three_quarters', returns the estimator with the corresponding percentile of entropy.

    Attributes
    ----------
    estimators_ : list of estimator objects
        The final list of estimators after the optimization process.
    """

    def __init__(
        self,
        learner,
        search_method,
        input_file,
        sbt=False,
        n_iter=25,
        cv=5,
        ensemble_entropy=1,
    ):
        self.learner = learner
        self.search_method = search_method
        self.input_file = input_file
        self.sbt = sbt
        self.estimators_ = None
        self.n_iter = n_iter
        self.cv = cv
        self.ensemble_entropy = ensemble_entropy
        self.scoring = {
            "accuracy": "accuracy",
            "entropy": DiversityMeasures().entropy_score,
        }

    def fit(self, X, y):
        """Performs hyperparameter optimization and fits the ensemble learning algorithm to the data.

        Parameters
        ----------
        X : array-like or pd.DataFrame of shape (n_samples, n_features)
            The training input samples.
        y : array-like or pd.Series of shape (n_samples,)
            The target values.

        Returns
        -------
        None
        """
        rf_param_distributions = {
            "n_estimators": range(3, 11),
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 5),
            "max_features": ["sqrt", "log2"],
        }
        ab_param_distributions = {
            "n_estimators": range(5, 13),
            "base_estimator__criterion": ["entropy"],
            "base_estimator__splitter": ["best", "random"],
            "base_estimator__max_depth": range(1, 7),
            "base_estimator__max_features": [None],
        }
        if self.search_method == "random" and self.learner == "RandomForest":
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                param_distributions=rf_param_distributions,
                scoring=self.scoring,
                refit="entropy",
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=-1,
            )

        elif self.search_method == "random" and self.learner == "AdaBoost":
            search = RandomizedSearchCV(
                AdaBoostClassifier(base_estimator=DecisionTreeClassifier()),
                param_distributions=ab_param_distributions,
                scoring=self.scoring,
                refit="entropy",
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=-1,
            )

        elif self.search_method == "full" and self.learner == "RandomForest":
            search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=rf_param_distributions,
                scoring=self.scoring,
                refit="entropy",
                cv=self.cv,
                n_jobs=-1,
            )

        elif self.search_method == "full" and self.learner == "AdaBoost":
            search = GridSearchCV(
                AdaBoostClassifier(
                    base_estimator=DecisionTreeClassifier(random_state=42)
                ),
                error_score="raise",
                param_grid=ab_param_distributions,
                scoring=self.scoring,
                refit="entropy",
                cv=self.cv,
                n_jobs=-1,
            )

        elif self.search_method == "bayes_opt" and self.learner == "RandomForest":

            def rf_crossval(n_estimators, criterion, max_depth, max_features):
                return rf_cv(
                    n_estimators=round(n_estimators),
                    criterion=["gini", "entropy"][round(criterion)],
                    max_depth=round(max_depth),
                    max_features=["sqrt", "log2"][round(max_features)],
                    X=X,
                    y=y,
                    score="entropy",
                    cv=self.cv,
                )

            search = BayesianOptimization(
                rf_crossval,
                {
                    "n_estimators": (3, 11),
                    "criterion": (-0.5, 1.49),
                    "max_depth": (0.51, 5.49),
                    "max_features": (-0.5, 1.49),
                },
                random_state=42,
            )

            search.maximize(init_points=5, n_iter=self.n_iter)
            return

        elif self.search_method == "bayes_opt" and self.learner == "AdaBoost":

            def ada_crossval(
                n_estimators,
                base_estimator__criterion,
                base_estimator__splitter,
                base_estimator__max_depth,
                base_estimator__max_features,
            ):
                return ada_cv(
                    algorithm="SAMME.R",
                    n_estimators=round(n_estimators),
                    base_estimator__criterion=["gini", "entropy"][
                        round(base_estimator__criterion)
                    ],
                    base_estimator__splitter=["best", "random"][
                        round(base_estimator__splitter)
                    ],
                    base_estimator__max_depth=round(base_estimator__max_depth),
                    base_estimator__min_samples_split=2,
                    base_estimator__min_samples_leaf=1,
                    base_estimator__max_features=[None, "sqrt", "log2"][
                        round(base_estimator__max_features)
                    ],
                    X=X,
                    y=y,
                    score="entropy",
                    cv=self.cv,
                )

            search = BayesianOptimization(
                ada_crossval,
                {
                    "n_estimators": (15, 20),
                    "base_estimator__criterion": (-0.5, 1.49),
                    "base_estimator__splitter": (-0.5, 1.49),
                    "base_estimator__max_depth": (1, 7),
                    "base_estimator__max_features": (-0.5, 2.49),
                },
                random_state=42,
            )

            search.maximize(init_points=5, n_iter=self.n_iter)
            return

        search.fit(X, y)
        results = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_entropy", axis=0, ascending=True, ignore_index=True
        )

        results.to_csv(
            "Results/"
            + str(self.learner)
            + "_"
            + str(self.search_method)
            + "_"
            + str(self.input_file)
            + "_"
            + "sbt"
            + str(self.sbt)
            + "_results.csv",
            index=False,
        )

        # returns a trained estimator with the desired entropy
        entropies = search.cv_results_["mean_test_entropy"]
        params = search.cv_results_["params"]
        params_base_estimator = dict({})

        if self.ensemble_entropy == "mean":
            if self.learner == "RandomForest":
                final_ensemble = RandomForestClassifier(
                    **params[np.abs(entropies - entropies.mean()).argmin()]
                ).fit(X, y)
            elif self.learner == "AdaBoost":
                params_ensemble = params[
                    np.abs(entropies - entropies.mean()).argmin()
                ].copy()
                keys = list(params_ensemble.keys())
                for key in keys:
                    if "base_estimator__" in key:
                        params_base_estimator[key[16:]] = params_ensemble.pop(key)
                final_ensemble = AdaBoostClassifier(
                    **params_ensemble,
                    base_estimator=DecisionTreeClassifier(**params_base_estimator),
                ).fit(X, y)

        elif self.ensemble_entropy == "middle":
            middle = (max(entropies) - min(entropies)) / 2 + min(entropies)

            if self.learner == "RandomForest":
                final_ensemble = RandomForestClassifier(
                    **params[np.abs(entropies - middle).argmin()]
                ).fit(X, y)
            elif self.learner == "AdaBoost":
                params_ensemble = params[np.abs(entropies - middle).argmin()].copy()
                keys = list(params_ensemble.keys())
                for key in keys:
                    if "base_estimator__" in key:
                        params_base_estimator[key[16:]] = params_ensemble.pop(key)
                final_ensemble = AdaBoostClassifier(
                    **params_ensemble,
                    base_estimator=DecisionTreeClassifier(**params_base_estimator),
                ).fit(X, y)

        elif self.ensemble_entropy == "quarter":
            quarter = (max(entropies) - min(entropies)) / 4 + min(entropies)
            if self.learner == "RandomForest":
                final_ensemble = RandomForestClassifier(
                    **params[np.abs(entropies - quarter).argmin()]
                ).fit(X, y)
            elif self.learner == "AdaBoost":
                params_ensemble = params[np.abs(entropies - quarter).argmin()].copy()
                keys = list(params_ensemble.keys())
                for key in keys:
                    if "base_estimator__" in key:
                        params_base_estimator[key[16:]] = params_ensemble.pop(key)
                final_ensemble = AdaBoostClassifier(
                    **params_ensemble,
                    base_estimator=DecisionTreeClassifier(**params_base_estimator),
                ).fit(X, y)

        elif self.ensemble_entropy == "three_quarters":
            three_quarters = max(entropies) - (max(entropies) - min(entropies)) / 2
            if self.learner == "RandomForest":
                final_ensemble = RandomForestClassifier(
                    **params[np.abs(entropies - three_quarters).argmin()]
                ).fit(X, y)
            elif self.learner == "AdaBoost":
                params_ensemble = params[
                    np.abs(entropies - three_quarters).argmin()
                ].copy()
                keys = list(params_ensemble.keys())
                for key in keys:
                    if "base_estimator__" in key:
                        params_base_estimator[key[16:]] = params_ensemble.pop(key)
                final_ensemble = AdaBoostClassifier(
                    **params_ensemble,
                    base_estimator=DecisionTreeClassifier(**params_base_estimator),
                ).fit(X, y)

        else:
            if self.learner == "RandomForest":
                final_ensemble = RandomForestClassifier(
                    **params[np.abs(entropies - self.ensemble_entropy).argmin()]
                ).fit(X, y)
            elif self.learner == "AdaBoost":
                params_ensemble = params[
                    np.abs(entropies - self.ensemble_entropy).argmin()
                ].copy()
                keys = list(params_ensemble.keys())
                for key in keys:
                    if "base_estimator__" in key:
                        params_base_estimator[key[16:]] = params_ensemble.pop(key)
                final_ensemble = AdaBoostClassifier(
                    **params_ensemble,
                    base_estimator=DecisionTreeClassifier(**params_base_estimator),
                ).fit(X, y)

        self.estimators_ = final_ensemble.estimators_
