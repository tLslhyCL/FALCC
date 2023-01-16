"""
This python file runs the FairBoost [1] algorithm.
[1] Bhaskaruni, D., Hu, H., Lan, C. "Improving Prediction Fairness via Model Ensemble". 2019.
"""
import math
import copy
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class FairBoost:
    """This class is used to use each single trained classifier.

    Parameters
    ----------
    index: string
        String name of the index.

    pred_id_list: list of ids
        List of all indices of the prediction dataset.

    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    favored: tuple
        Tuple containing the values of the favored group.

    label: string
        String name of the label.

    base_estimator: classifier
        Base estimator for the FairBoost algorithm.

    iterations: int
        Number of iterations for the FairBoost algorithm.
    """
    def __init__(self, index, pred_id_list, sens_attrs, favored, label, base_estimator, iterations=6):
        self.index = index
        self.pred_id_list = pred_id_list
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.label = label
        self.base_estimator = base_estimator
        self.iterations = iterations


    def fit_predict(self, X, y, X_pred, y_pred, r=0.1):
        """Trains the FairBoost estimators on the training data and then predicts the labels
        for X_pred.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, m_features)
            Training data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y: array-like, shape (n_samples)
            Label vector relative to the prediction data X.

        X_pred: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_pred: array-like, shape (n_samples)
            Label vector relative to the prediction data X_pred.

        r: float
            Threshold that determines for which value the local group fairness is violated.


        Returns/Output
        ----------
        "[model]_prediction_output.csv": Output DataFrame file in .csv format for each model.
            Contains: index, value of sensitive attributes, label, predicted value.
        """
        dataset_length = len(X)
        sample_weight = [1/dataset_length for i in range(dataset_length)]
        estimator = self.base_estimator
        trained_estimator = []
        alphas = []
        #TRAIN
        for i in range(self.iterations):
            error = math.inf
            X = X.loc[:, X.columns != "prediction"]
            current_estimator = copy.deepcopy(estimator.fit(X, y, sample_weight))
            estimator_predictions = current_estimator.predict(X)
            X["prediction"] = estimator_predictions
            full_error = 0
            error_fair = 0
            wrong_classified = []
            for j in range(len(estimator_predictions)):
                full_error += sample_weight[j]
                nbrs = NearestNeighbors(n_neighbors=31, algorithm='kd_tree').fit(X.values)
                indices = nbrs.kneighbors(X.iloc[j].values.reshape(1, -1), return_distance=False)
                real_indices = X.index[indices].tolist()
                nearest_neighbors_df = X.loc[real_indices[0]]
                favored_val = 0
                discriminated = 0
                grouped_df = nearest_neighbors_df.groupby(self.sens_attrs)
                if len(grouped_df) == 2:
                    for key, item in grouped_df:
                        part_df = grouped_df.get_group(key)
                        if key == self.favored:
                            total_fav = len(part_df)
                            for i, row in part_df.iterrows():
                                favored_val += row["prediction"]
                        else:
                            total_discr = len(part_df)
                            for i, row in part_df.iterrows():
                                discriminated += row["prediction"]
                    ppr_fav = favored_val/total_fav
                    ppr_discr = discriminated/total_discr
                    if (ppr_fav - ppr_discr) > r:
                        error_fair += sample_weight[j]
                        wrong_classified.append(j)

            error = error_fair/(full_error + 1e-24)
            alpha = np.log((1-error)/(error + 1e-24))

            trained_estimator.append(current_estimator)
            alphas.append(alpha)

            for j in wrong_classified:
                sample_weight[j] = sample_weight[j] * math.exp(alpha)


        #PREDICT
        pred_df = pd.DataFrame(columns=[self.index, self.label, "FairBoost"])
        sens_count = 1
        for attr in self.sens_attrs:
            pred_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        X2_pred = copy.deepcopy(X_pred)
        for i, row in X2_pred.iterrows():
            if row[self.sens_attrs[0]] == 0:
                row[self.sens_attrs[0]] = 1
            else:
                row[self.sens_attrs[0]] = 0
        count = 0
        for i, row in X_pred.iterrows():
            alpha_total = 0
            pred = 0
            for j, estimator in enumerate(trained_estimator):
                pred += estimator.predict(X_pred.loc[i].values.reshape(1, -1))[0] * alphas[j]
                alpha_total += alphas[j]
            pred = pred/alpha_total
            pred_df.at[count, self.index] = i
            for sens in self.sens_attrs:
                pred_df.at[count, sens] = row[sens]
            pred_df.at[count, self.label] = y_pred.loc[i].values[0]
            pred_df.at[count, "FairBoost"] = 0 if pred < 0.5 else 1
            count = count + 1
        return pred_df
