"""
This python file contains the decouple algorithms by Dwork et al. [1] Currently the first
decouple algorithm is implemented (if sbt = True). Also a variant of the first algorithm,
with the difference that classifiers are also trained on outer-group samples (part of
the second algorithm), is implemented.
[1] Dwork, C., Immorlica, N., Kalai, A., Leiserson, M. "Decoupled Classifiers for Group-Fair
    and Efficient Machine Learning". 2018.
"""
import warnings
import copy
import pandas as pd
import joblib

class Decouple:
    """This class is used to use the decouple functions.

    Parameters
    ----------
    metricer: Object of the Metrics class.
        Created object of the Metrics class, so its functions can be used.

    index: string
        String name of the index.

    pred_id_list: list of ids
        List of all indices of the prediction dataset.

    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    label: string
        String name of the target column.

    favored: tuple of float
        Tuple of the values of the favored group.

    model_list: list of strings
        List of the model names.
    """
    def __init__(self, metricer, index, pred_id_list, sens_attrs, label, favored, model_comb):
        self.metricer = metricer
        self.index = index
        self.pred_id_list = pred_id_list
        self.sens_attrs = sens_attrs
        self.label = label
        self.favored = favored
        self.model_comb = model_comb


    def decouple(self, model_test, X_pred, y_pred, metric, weight, sbt):
        """This function implements the DCS-LA algorithm.

        Parameters
        ----------
        model_test: DataFrame, shape (n_samples, m_features)
            Each entry contains index, model name, values of the sensitive attributes, number
            and probability of positive predicted values, number and probability of wrong
            predicted label, number of entries in the group. An entry is made for each model
            + sensitive group combination.

        X_pred: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_pred: array-like, shape (n_samples)
            Label vector relative to the prediction data X_pred.

        metric: string
            Name of the metric which should be used to get the best result.

        weight: float (0-1)
            Value to balance the accuracy and fairness parts of the metrics.
            Under 0.5: Give fairness higher importance.
            Over 0.5: Give accuracy higher importance.

        sbt: boolean
            True, if the dataset was split before training each classifier.


        Returns/Output
        ----------
        pred_df: Output DataFrame
            Contains: index, value of sensitive attributes, label, predicted value,
            model used for prediction, model combination used for prediction.
        """
        #Get the globally best model combination according to the given metric and weight.
        comb_list, group_tuple = self.metricer.fairness_metric(model_test, self.model_comb,\
            self.favored, metric, weight)
        comb = comb_list[0]
        warnings.simplefilter(action='ignore', category=FutureWarning)

        #Create a dataframe containing the important columns. Add the sensitive attributes in
        #the first for-loop.
        if not sbt:
            decouple = "Decouple"
        else:
            decouple = "Decouple-SBT"
        pred_df = pd.DataFrame(columns=[self.index, self.label, decouple, "model_used",\
            "model_comb"])
        sens_count = 1
        count = 0

        for attr in self.sens_attrs:
            pred_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        X2_pred = copy.deepcopy(X_pred)

        #Iterate over every entry in the prediction dataframe and use the corresponding model of
        #the best global model combination.
        for i, row in X_pred.iterrows():
            pred_df.at[count, self.index] = self.pred_id_list[count]
            for attr in self.sens_attrs:
                pred_df.at[count, attr] = row[attr]
            pred_df.at[count, self.label] = y_pred.iloc[count].values[0]

            pred_sens = []
            for attr in self.sens_attrs:
                pred_sens.append(row[attr])
            #Find the corresponding group of the group_tuple to get the correct model for the
            #prediction. If there is only one sensitive attribute chosen go to the corresponding
            #"else" method.
            if len(group_tuple) != 1:
                for j, gt in enumerate(group_tuple):
                    group_found = True
                    for k in range(len(self.sens_attrs)):
                        if pred_sens[k] != gt[k]:
                            group_found = False
                            X2_pred.loc[i, self.sens_attrs[0]] = abs(X_pred.loc[i, self.sens_attrs[0]] - 1)
                    if group_found:
                        group = comb[j]
            else:
                group_found = True
                for k in range(len(self.sens_attrs)):
                    if pred_sens[k] != group_tuple[k]:
                        group_found = False
                        X2_pred.loc[i, self.sens_attrs[0]] = abs(X_pred.loc[i, self.sens_attrs[0]] - 1)
                if group_found:
                    group = comb[j]

            #Load the corresponding model of the model combination and predict the label of the
            #current entry of the prediction dataset.
            joblib_file = group
            used_model = joblib.load(joblib_file)
            prediction = used_model.predict(X_pred.iloc[count].values.reshape(1, -1))
            pred_df.at[count, decouple] = prediction[0]
            pred_df.at[count, "model_used"] = group
            pred_df.at[count, "model_comb"] = comb

            count = count + 1

        return pred_df
