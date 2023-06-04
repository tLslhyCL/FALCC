"""
In this python file multiple classification models are trained.
"""
import copy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset
from algorithm.codes import AdaBoostClassifierMult
from algorithm.codes import HyperOptimizedLearner
#from .FaX_AI import FaX_methods
#from .Fair_SMOTE.SMOTE import smote
#from .Fair_SMOTE.Generate_Samples import generate_samples



class Models():
    """Multiple different model learners are part of this class.

    Parameters
    ----------
    X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
        Training data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_train: array-like, shape (n_samples)
        Label vector relative to the training data X_train.

    X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
        Test data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_test: array-like, shape (n_samples)
        Label vector relative to the test data X_test.

    sens_attrs: list of strings
        List of the column names of sensitive attributes in the dataset.

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.
    """
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, favored, ignore_sens=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.ignore_sens = ignore_sens


    def decision_tree(self, sample_weight):
        """Fit the decision tree model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        dt_pred: list of predicted label for our testdata X_test

        "dectree": string of the used model name
        """
        clf = DecisionTreeClassifier()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            dt_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            dt_pred = clf.predict(self.X_test)

        return clf, dt_pred, "dectree"


    def linear_svm(self, sample_weight):
        """Fit the linear support vector machine model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        svm_pred: list of predicted label for our testdata X_test

        "linsvm": string of the used model name
        """
        clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train,
                    **{'linear_svc__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            svm_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, **{'linear_svc__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train, self.y_train)
            svm_pred = clf.predict(self.X_test)

        return clf, svm_pred, "linsvm"


    def nonlinear_svm(self, sample_weight):
        """Fit the nonlinear support vector machine model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        svm_pred: list of predicted label for our testdata X_test

        "nonlinsvm": string of the used model name
        """
        clf = Pipeline([("poly_features", PolynomialFeatures(degree=2)), \
            ("scaler", StandardScaler()), ("svm_clf", LinearSVC(C=10, loss="hinge"))])
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train,
                    **{'svm_clf__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            svm_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, **{'svm_clf__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train, self.y_train)
            svm_pred = clf.predict(self.X_test)

        return clf, svm_pred, "nonlinsvm"


    def log_regr(self, sample_weight):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = LogisticRegression(solver='lbfgs')
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            reg_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            reg_pred = clf.predict(self.X_test)

        return clf, reg_pred, "logregr"


    def softmax_regr(self, sample_weight):
        """Fit the Softmax regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "softmaxregr": string of the used model name
        """
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            reg_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            reg_pred = clf.predict(self.X_test)

        return clf, reg_pred, "softmaxregr"


    def adaboost(self, modelsize):
        """Fit the AdaBoost models according to the given training data via the adapted
        AdaBoostClassifier.

        Parameters
        ----------
        modelsize: integer
            Amount of models that should be trained.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "adaboost": string of the used model name
        """
        abc = AdaBoostClassifierMult([LogisticRegression(),
            Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss="hinge"))]),
            DecisionTreeClassifier()], modelsize)
        if self.ignore_sens:
            classifier_list = abc.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            classifier_list = abc.fit(self.X_train, self.y_train)
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "adaboost"

    def opt_learner(self, ensemble_strat, input_file, sbt):
        if ensemble_strat=="RandomForest":
            estimator = HyperOptimizedLearner(learner="RandomForest", search_method="full", input_file=input_file, sbt=sbt, cv=5, ensemble_entropy=1)
        elif ensemble_strat=="AdaBoost":
            estimator = HyperOptimizedLearner(learner="AdaBoost", search_method="full", input_file=input_file, sbt=sbt, cv=5, ensemble_entropy=1)
        if self.ignore_sens:
            estimator.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            estimator.fit(self.X_train, self.y_train)
        classifier_list = estimator.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, ensemble_strat

    def optimized_adaboost(self, n_estimators=17, criterion="gini", max_depth=2, max_features="sqrt", splitter="random"):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """


        ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, max_features=max_features, splitter=splitter),n_estimators=n_estimators)
        if self.ignore_sens:
            ab.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            ab.fit(self.X_train, self.y_train)
        classifier_list = ab.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "AdaBoostOpt"


    def adaboost_classic(self, n_estimators=17, max_depth=2, splitter="random"):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """


        ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion="entropy", max_depth=max_depth, max_features="log2", splitter=splitter), n_estimators=n_estimators)
        if self.ignore_sens:
            ab.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            ab.fit(self.X_train, self.y_train)
        classifier_list = ab.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "AdaBoostClassic", ab


    def rf_classic(self, n_estimators=17, max_depth=2, criterion="gini"):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """


        rf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
        if self.ignore_sens:
            rf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            rf.fit(self.X_train, self.y_train)
        classifier_list = rf.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "RandomForestClassic", rf


    def fax(self, method="MIM"):
        """Run the FaX algorithm

        Parameters
        ----------
        method: str
            FaX method that should be used: "MIM" or "OPT"


        Returns/Output
        ----------
        model: Trained model

        pred: list
            List containing the predictions

        "FaX": Name of the algorithm chosen
        """
        X2 = self.X_train.loc[:, self.X_train.columns != self.sens_attrs[0]]
        Z2 = self.X_train[self.sens_attrs[0]].to_frame()
        Y2 = self.y_train

        X3 = self.X_test.loc[:, self.X_test.columns != self.sens_attrs[0]]

        if method == "MIM":
            model = FaX_methods.MIM(X2, Z2, Y2)
        elif method == "OPT":
            model = FaX_methods.OIM(X2, Z2, Y2)

        pred = model.predict(X3)

        return model, pred, "FaX"


    def smote(self):
        """Preprocess the dataset using Fair-SMOTE and the train a classifier on the new dataset.

        Parameters: -


        Returns/Output
        ----------
        clf: Trained classifier on the preprocessed dataset

        prediction: list
            List containing the predictions

        "Fair-SMOTE": Name of the algorithm chosen
        """
        label = list(self.y_train.columns)[0]
        train_df = copy.deepcopy(self.X_train)
        train_df[label] = self.y_train
        dataset_orig_train = copy.deepcopy(train_df)
        train_df.reset_index(drop=True, inplace=True)
        cols = train_df.columns
        smt = smote(train_df)
        train_df = smt.run()
        train_df.columns = cols
        y_train_new = train_df[label]
        X_train_new = train_df.drop(label, axis=1)

        dict_cols = dict()
        cols = list(train_df.columns)
        for i, col in enumerate(cols):
            dict_cols[i] = col
        #Find Class & protected attribute distribution
        zero_zero = len(dataset_orig_train[(dataset_orig_train[label] == 0)
            & (dataset_orig_train[self.sens_attrs[0]] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train[label] == 0)
            & (dataset_orig_train[self.sens_attrs[0]] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train[label] == 1)
            & (dataset_orig_train[self.sens_attrs[0]] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train[label] == 1)
            & (dataset_orig_train[self.sens_attrs[0]] == 1)])
        maximum = max(zero_zero,zero_one,one_zero,one_one)
        if maximum == zero_zero:
            zero_one_to_be_incresed = maximum - zero_one
            one_zero_to_be_incresed = maximum - one_zero
            one_one_to_be_incresed = maximum - one_one
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_one)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_one_one)
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_new = df_new.append(df_zero_zero)
        if maximum == zero_one:
            zero_zero_to_be_incresed = maximum - zero_zero
            one_zero_to_be_incresed = maximum - one_zero
            one_one_to_be_incresed = maximum - one_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_one_one)
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_new = df_new.append(df_zero_one)
        if maximum == one_zero:
            zero_zero_to_be_incresed = maximum - zero_zero
            zero_one_to_be_incresed = maximum - zero_one
            one_one_to_be_incresed = maximum - one_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_zero_one)
            df_new = df_new.append(df_one_one)
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_new = df_new.append(df_one_zero)
        if maximum == one_one:
            zero_zero_to_be_incresed = maximum - zero_zero
            one_zero_to_be_incresed = maximum - one_zero
            zero_one_to_be_incresed = maximum - zero_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 0)]
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_zero_one)
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[self.sens_attrs[0]] == 1)]
            df_new = df_new.append(df_one_one)


        X_train_new, y_train_new = df_new.loc[:, df_new.columns != label], df_new[label]
        clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        clf.fit(X_train_new, y_train_new)
        prediction = clf.predict(self.X_test)

        return clf, prediction, "Fair-SMOTE"


    def lfr(self):
        """Return a fair representation.

        Parameters: -


        Returns
        -------
        model: Trained LFR model

        prediction: list of predicted label for our testdata X_test

        "LFR": string of the used model name
        """
        label = list(self.y_train.columns)[0]
        train_df = pd.merge(self.X_train, self.y_train, left_index=True, right_index=True)
        test_df = pd.merge(self.X_test, self.y_test, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=self.sens_attrs)
        dataset_test = BinaryLabelDataset(df=test_df, label_names=[label], protected_attribute_names=self.sens_attrs)

        ###Only binary now
        privileged_groups = []
        unprivileged_groups = []
        priv_dict = dict()
        unpriv_dict = dict()
        priv_val = self.favored
        if priv_val == 0:
            priv_dict[self.sens_attrs[0]] = 0
            unpriv_dict[self.sens_attrs[0]] = 1
        elif priv_val == 1:
            priv_dict[self.sens_attrs[0]] = 1
            unpriv_dict[self.sens_attrs[0]] = 0

        privileged_groups = [priv_dict]
        unprivileged_groups = [unpriv_dict]

        model = LFR(unprivileged_groups, privileged_groups)
        model = model.fit(dataset_train)
        dataset_transf_test = model.transform(dataset_test)

        preds = list(dataset_transf_test.labels)
        prediction = [preds[i][0] for i in range(len(preds))]

        return model, prediction, "LFR"
