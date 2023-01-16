"""
This file includes the FALCC class & is used to call the 3rd step and online phase
of the FALCC algorithm.
"""
import copy
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

class FALCC:
    """This class calls runs the 3rd step and online phase of the FALCC algorithm.

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

    d: dictionary
        Dictionary containing model-related information.

    proxy: string
        Name of the proxy strategy used

    weight_dict: dictionary
        Dictionary containing the column weights (if a proxy strategy is used)

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.

    pre_processed: boolean
        Is set to True, if the datasets have been properly preprocessed. For the current
        new approach, this is a requirement.
    """
    def __init__(self, metricer, index, sens_attrs, label, favored, model_list, X_test,
        model_comb, d, proxy, weight_dict=None, ignore_sens=False, pre_processed=True):
        self.metricer = metricer
        self.index = index
        self.sens_attrs = sens_attrs
        self.label = label
        self.favored = favored
        self.model_list = model_list
        self.X_test = X_test
        self.model_comb = model_comb
        self.d = d
        self.proxy = proxy
        self.weight_dict = weight_dict
        self.ignore_sens = ignore_sens
        self.pre_processed = pre_processed


    def predict(self, model_dict, X_pred, y_pred, sbt, kmeans):
        """This function predicts the label of each prediction sample for FALCC/FALCC-SBT
        (the online phase).

        Parameters
        ----------
        model_dict: DataFrame, shape (n_samples, m_features)
            Dataframe on which the models were tested.

        X_pred: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_pred: array-like, shape (n_samples)
            Label vector relative to the prediction data X_pred.

        sbt: boolean
            If set to True, the classifiers were trained on splitted datasets.

        kmeans: Object
            Object instantiated from the clustering algorithm. Contains each cluster.


        Returns/Output
        ----------
        pred_df: Output DataFrame
            Contains: index, value of sensitive attributes, label, predicted value,
            model used for prediction, model combination used for prediction.
        """
        if not sbt:
            cluster_model = "FALCC"
        else:
            cluster_model = "FALCC-SBT"

        pred_df = pd.DataFrame(columns=[self.index, self.label, cluster_model, "model_used",\
            "model_comb"])
        pred_count = 0

        sens_count = 1
        X_pred_cluster = copy.deepcopy(X_pred)
        for attr in self.sens_attrs:
            pred_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1
            X_pred_cluster = X_pred_cluster.loc[:, X_pred_cluster.columns != attr]

        if self.proxy in ("reweigh", "remove"):
            for col in list(X_pred_cluster.columns):
                if col in self.weight_dict:
                    X_pred_cluster[col] *= self.weight_dict[col]
                else:
                    X_pred_cluster = X_pred_cluster.loc[:, X_pred_cluster.columns != col]

        Z_pred = copy.deepcopy(X_pred)
        if self.ignore_sens:
            for sens in self.sens_attrs:
                Z_pred = Z_pred.loc[:, Z_pred.columns != sens]
        Z2_pred = copy.deepcopy(Z_pred)

        for i in range(len(X_pred)):
            sens_value = []
            for attr in self.sens_attrs:
                sens_value.append(float(X_pred.iloc[i][attr]))

            cluster_results = kmeans.predict(X_pred_cluster.iloc[i].values.reshape(1, -1))

            model = model_dict[cluster_results[0]][str(sens_value)]
            used_model = joblib.load(model)
            if str(sens_value) == "[0.0]":
                Z2_pred.iloc[i][self.sens_attrs[0]] = 1.0
            elif str(sens_value) == "[1.0]":
                Z2_pred.iloc[i][self.sens_attrs[0]] = 0.0
            elif str(sens_value) == "[1]":
                Z2_pred.iloc[i][self.sens_attrs[0]] = 0
            else:
                Z2_pred.iloc[i][self.sens_attrs[0]] = 1

            prediction = used_model.predict(Z_pred.iloc[i].values.reshape(1, -1))[0]

            pred_df.at[pred_count, self.index] = y_pred.index[i]
            for attr in self.sens_attrs:
                pred_df.at[pred_count, attr] = X_pred.loc[y_pred.index[i], attr]
            pred_df.at[pred_count, self.label] = y_pred.iloc[i].values[0]
            pred_df.at[pred_count, cluster_model] = prediction
            pred_df.at[pred_count, "model_used"] = model

            pred_count = pred_count + 1

        return pred_df


    def cluster_offline(self, X_test_cluster, kmeans, test_df, metric, weight, link,
        other_folder=None, sbt=True):
        """The third step of the offline phase of FALCC and FALCC-SBT.

        Parameters
        ----------
        X_test_cluster: {array-like, sparse matrix}, shape (n_samples, m_features)
            Dataset without sensitive attributes, but added cluster numbers.

        kmeans: k-means object
            Already includes generated cluster via kmeans.

        test_df: DataFrame, shape (n_samples, m_features)
            Each entry contains a sample and its prediction per model.

        metric: string
            Name of the metric which should be used to get the best result.

        weight: float (0-1)
            Value to balance the accuracy and fairness parts of the metrics.
            Under 0.5: Give fairness higher importance.
            Over 0.5: Give accuracy higher importance.

        link: string
            Directory where data should be saved.

        other_folder: string
            Additional folder string, solely used for the 2nd experiment.

        sbt: boolean
            If set to True, the classifiers were trained on splitted datasets.


        Returns/Output
        ----------
        model_dict: dictionary
            Returns a dictionary which saves on which classifier should be used on the
            corresponding samples (per cluster and sensitive attribute values).
        """
        clustered_df = X_test_cluster.groupby("cluster")
        model_dict = dict()
        column_list = test_df.columns

        groups = test_df[self.sens_attrs].drop_duplicates(self.sens_attrs).reset_index(drop=True)
        actual_num_of_groups = len(groups)
        sensitive_groups = []
        sens_cols = groups.columns
        for i, row in groups.iterrows():
            sens_grp = []
            for col in sens_cols:
                sens_grp.append(row[col])
            sensitive_groups.append(tuple(sens_grp))

        for key, item in clustered_df:
            part_df = clustered_df.get_group(key)
            part_df2 = test_df.merge(part_df, on="index", how="inner")[column_list]
            groups2 = part_df2[self.sens_attrs].drop_duplicates(self.sens_attrs).reset_index(drop=True)
            num_of_groups = len(groups2)
            cluster_sensitive_groups = []
            for i, row in groups2.iterrows():
                sens_grp = []
                for col in sens_cols:
                    sens_grp.append(row[col])
                cluster_sensitive_groups.append(tuple(sens_grp))

            #If a cluster does not contain samples of all groups, it will take the k nearest neighbors
            #(default value = 15) to test the model combinations
            if num_of_groups != actual_num_of_groups:
                cluster_center = kmeans.cluster_centers_[key]
                for sens_grp in sensitive_groups:
                    if sens_grp not in cluster_sensitive_groups:
                        if len(self.sens_attrs) == 1:
                            sens_grp = sens_grp[0]
                        grouped_df = self.X_test.groupby(self.sens_attrs)
                        for key2, item2 in grouped_df:
                            if key2 == sens_grp:
                                knn_df = grouped_df.get_group(key2)
                                for sens_attr in self.sens_attrs:
                                    knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                                nbrs = NearestNeighbors(n_neighbors=15, algorithm='kd_tree').fit(knn_df.values)
                                indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                                real_indices = self.X_test.index[indices].tolist()
                                nearest_neighbors_df = test_df.loc[real_indices[0]]
                                part_df2 = part_df2.append(nearest_neighbors_df)

            if not sbt:
                model_test = self.metricer.test_score(part_df2, self.model_list)
                if other_folder != None:
                    model_test.to_csv(link + str(other_folder) + "/" + str(key) + "_inaccuracy_testphase.csv",
                        index_label=self.index)
                else:
                    model_test.to_csv(link + str(key) + "_inaccuracy_testphase.csv",
                        index_label=self.index)
            else:
                model_test = self.metricer.test_score_sbt(part_df2, self.d)
                if other_folder != None:
                    model_test.to_csv(link + str(other_folder) + "/" + str(key) + "_inaccuracy_testphase_sbt.csv",
                        index_label=self.index)
                else:
                    model_test.to_csv(link + str(key) + "_inaccuracy_testphase_sbt.csv",
                        index_label=self.index)
            comb_list_global, group_tuple = self.metricer.fairness_metric(model_test,
                self.model_comb, self.favored, metric, weight, comb_amount=1)

            subdict = dict()
            for i, gt in enumerate(group_tuple):
                dict_key = []
                for j in gt:
                    dict_key.append(float(j))
                subdict[str(dict_key)] = comb_list_global[0][i]

            model_dict[key] = subdict

        return model_dict, kmeans
