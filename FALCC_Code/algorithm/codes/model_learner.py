"""
In this python file multiple classification models are trained.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from algorithm.codes import AdaBoostClassifierMult


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
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, ignore_sens=False):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
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
