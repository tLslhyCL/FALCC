"""
This python file contains the implementations for the (original) FALCES [1] framework.
[1] Lässig, N., Oppold, S., Herschel, M. "Metrics and Algorithms for Locally Fair and Accurate
    Classifications using Ensembles". 2022.
"""
import warnings
import copy
import pandas as pd
import joblib
from aif360.datasets import BinaryLabelDataset
from sklearn.neighbors import NearestNeighbors

class FALCES:
    """This class is used to use both fair, dynamic model ensemble algorithms.

    Parameters
    ----------
    metricer: Object of the Metrics class.
        Created object of the Metrics class, so its functions can be used.

    index: string
        String name of the index.

    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    label: string
        String name of the target column.

    favored: tuple of float
        Tuple of the values of the favored group.

    model_list: list of strings
        List of the model names.

    X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
        Test data vector, where n_samples is the number of samples and
        m_features is the number of features.

    model_comb: list of tuple
        All possible model combinations.

    model_dict: dictionary
        Dictionary containing model-related information

    link: str
        Link of the output directory.

    fairinput: boolean
        Is set to True, if we use fair classifiers as input.

    pre_processed: boolean
        Is set to True, if the datasets have been properly preprocessed. For the current
        new approach, this is a requirement.
    """
    def __init__(self, metricer, index, sens_attrs, label, favored, model_list, X_test, model_comb,
        model_dict, link, fairinput, pre_processed=True):
        self.metricer = metricer
        self.index = index
        self.sens_attrs = sens_attrs
        self.label = label
        self.favored = favored
        self.model_list = model_list
        self.X_test = X_test
        self.model_comb = model_comb
        self.model_dict = model_dict
        self.link = link
        self.fairinput = fairinput
        self.pre_processed = pre_processed


    def predict(self, test_output, X_pred, y_pred, strategy, metric, weight, sbt, knn_size,
        comb_list_global=None):
        """This function implements the naive fair dynamic model ensemble approach.

        Parameters
        ----------
        test_output: DataFrame, shape (n_samples, m_features)
            Dataframe on which the models were tested.

        X_pred: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_pred: array-like, shape (n_samples)
            Label vector relative to the prediction data X_pred.

        strategy: string ("naive" or "performance-efficient")
            Indicates if the PFA version of the algorithm is used or not.

        metric: string
            Name of the metric which should be used to get the best result.

        weight: float (0-1)
            Value to balance the accuracy and fairness parts of the metrics.
            Under 0.5: Give fairness higher importance.
            Over 0.5: Give accuracy higher importance.

        sbt: boolean
            If set to True, the classifiers were trained on splitted datasets.

        knn_size: int
            Amount of kNN of each group that is considered.

        comb_list_global: list of tuple
            This list contains the tuples of the globally best model combinations.

        Returns/Output
        ----------
        pred_df: Output DataFrame
            Contains: index, value of sensitive attributes, label, predicted value,
            model used for prediction, model combination used for prediction.
        """
        #Create dataframe and add the sensitive attributes as columns.
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if strategy == "naive":
            if not sbt:
                nfd = "FALCES"
            else:
                nfd = "FALCES-SBT"
        elif strategy == "performance-efficient":
            if not sbt:
                nfd = "FALCES-PFA"
            else:
                nfd = "FALCES-SBT-PFA"

        pred_df = pd.DataFrame(columns=[self.index, self.label, nfd, "model_used", "model_comb"])
        sens_count = 1
        pred_count = 0
        for attr in self.sens_attrs:
            pred_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        X2_pred = copy.deepcopy(X_pred)

        #Build the KNN Tree for each group. Only possible if the data is preprocessed accordingly.
        grouped_df = self.X_test.groupby(self.sens_attrs)
        knn_tree_list = []
        if self.pre_processed:
            for key, item in grouped_df:
                part_df = grouped_df.get_group(key)
                nbrs = NearestNeighbors(n_neighbors=knn_size, algorithm='kd_tree').fit(part_df.values)
                knn_tree_list.append(nbrs)

        if self.fairinput:
            ##For FaX if needed
            X3 = copy.deepcopy(X_pred)
            X3 = X3.loc[:, X3.columns != self.sens_attrs[0]]

            ##For LFR if needed
            lfr_pred_df = pd.merge(X_pred, y_pred, left_index=True, right_index=True)
            dataset_pred = BinaryLabelDataset(df=lfr_pred_df, label_names=[self.label], protected_attribute_names=self.sens_attrs)
            lfr_model = joblib.load(self.link + "LFR_model.pkl")
            dataset_transf_pred = lfr_model.transform(dataset_pred)
            lfr_preds = list(dataset_transf_pred.labels)
            lfr_prediction = [lfr_preds[i][0] for i in range(len(lfr_preds))]

        #Iterate over every point of the prediction dataset.
        for i in range(len(X_pred)):
            index_list = []
            #Get the indices of the x nearest neighbors of each sensitive group and add them to
            #the same dataframe.
            j = 0
            for key, item in grouped_df:
                part_df = grouped_df.get_group(key)
                if self.pre_processed:
                    nbrs = knn_tree_list[j]
                    indices = nbrs.kneighbors(X_pred.iloc[i].values.reshape(1, -1),
                        return_distance=False)
                    real_indices = part_df.index[indices].tolist()
                    index_list.extend(real_indices)
                else:
                    indices = []
                    distance_dict = {}
                    columns = [attr for attr in part_df.columns if attr != self.label]
                    for k in range(len(part_df)):
                        distance = 0
                        for attr in columns:
                            if part_df.iloc[k][attr] == X_pred.iloc[i][attr]:
                                distance += 1
                        distance_dict[k] = distance
                    distance_tuple = sorted(distance_dict.items(), key=lambda x: x[1])
                    for k in range(0,knn_size):
                        indices.append(distance_tuple[k][0])
                    real_indices = part_df.index[indices].tolist()
                    index_list.extend(real_indices)
                j = j + 1
            #Check to which group the current entry belongs to.
            count = 0
            for key, item in grouped_df:
                if isinstance(key, tuple):
                    if key == tuple(X_pred[self.sens_attrs].iloc[i]):
                        group_count = count
                    else:
                        X2_pred.iloc[i][self.sens_attrs[0]] = abs(X_pred.iloc[i][self.sens_attrs[0]] - 1)
                else:
                    if key == tuple(X_pred[self.sens_attrs].iloc[i])[0]:
                        group_count = count
                    else:
                        X2_pred.iloc[i][self.sens_attrs[0]] = abs(X_pred.iloc[i][self.sens_attrs[0]] - 1)
                count = count + 1

            if isinstance(index_list[0], list):
                index_list = [index for sublist in index_list for index in sublist]

            df = test_output.loc[index_list]
            #Test all possible model combinations on this dataframe & return the best combination
            #based on given metric and weight.
            if sbt:
                df2 = self.metricer.test_score_sbt(df, self.model_dict)
            else:
                df2 = self.metricer.test_score(df, self.model_list)
            if strategy == "naive":
                comb_list, group_tuple = self.metricer.fairness_metric(df2, self.model_comb,
                    self.favored, metric, weight)
            elif strategy == "performance-efficient":
                comb_list, group_tuple = self.metricer.fairness_metric(df2, comb_list_global,
                    self.favored, metric, weight)
            comb = comb_list[0]
            #Use the model of the group in the combination, where the entry belongs to.
            joblib_file = comb[group_count]
            used_model = joblib.load(joblib_file)
            if "FaX" in joblib_file:
                prediction = used_model.predict(X3.iloc[i].values.reshape(1, -1))[0]
            elif "LFR" in joblib_file:
                prediction = lfr_prediction[i]
            else:
                prediction = used_model.predict(X_pred.iloc[i].values.reshape(1, -1))[0]
            pred_index = y_pred.index[i]
            true_value = y_pred.iloc[i]
            pred_df = self.metricer.prediction_output_add(pred_df, X_pred, pred_count, self.index,
                pred_index, true_value, prediction, nfd, comb[group_count], comb)
            pred_count = pred_count + 1

        return pred_df


    def efficient_offline(self, model_test, metric, weight, threshold, comb_amount):
        """This function implements the offline filtering aspect of the global unfair combinations
        in performance-efficient fair dynamic model ensemble approach.

        Parameters
        ----------
        model_test: DataFrame, shape (n_samples, m_features)
            Each entry contains index, model name, values of the sensitive attributes, number
            and probability of positive predicted values, number and probability of wrong
            predicted label, number of entries in the group. An entry is made for each model
            + sensitive group combination.

        metric: string
            Name of the metric which should be used to get the best result.

        weight: float (0-1)
            Value to balance the accuracy and fairness parts of the metrics.
            Under 0.5: Give fairness higher importance.
            Over 0.5: Give accuracy higher importance.

        threshold: float
            Each combination which metric value score stays under that threshold is added to the
            output list. If no combination has its value under the threshold, the best combination
            is returned.

        comb_amount: integer
            Number of combinations which are returned.


        Returns/Output
        ----------
        comb_list_global: list of tuples
            This list contains all globally fair model combinations.
        """
        #Get the globally best model combinations according to the given metric and weight, as well\
        #as threshold and comb_amount.
        comb_list_global, group_tuple = self.metricer.fairness_metric(model_test, self.model_comb,\
            self.favored, metric, weight, threshold, comb_amount)

        return comb_list_global
