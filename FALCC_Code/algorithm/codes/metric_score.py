"""
In this python file, the best classifier(s) or combination(s) of classifiers in terms of
accuracy & fairness are determined and returned.
"""
import math
import pandas as pd
import numpy as np


def fairness_score(metric, unfairness, model, fav_pppv, total_pppv, total_fp, total_fn, groups):
    """This function updates the fairness/unfairness score of a model combination
    based on the given metric.

    Parameters
    ----------
    metric: string
        Name of the fairness metric which should be used.

    unfairness: float
        Current unfairness value of the model combination which will be updated.

    model: DataFrame, shape (1 sample, m_features)
        Current tested model with index, model name, number and probability of positive
        predicted values, number and probability of wrong predicted label, number of
        entries in the group + values of the sensitive attributes.

    fav_pppv: list of length 3 of floats (0-1)
        First list: Pr(z=1)
        Second list: Pr(z=1|y=0)
        Third list: Pr(z=1|y=1)

    total_pppv: list of length 3 of floats (0-1)
        First list: Pr(z=1)
        Second list: Pr(z=1|y=0)
        Third list: Pr(z=1|y=1)

    total_fp: integer
        Amount of false positives.

    total_fn: integer
        Amount of false negatives.

    groups: integer
        Number of sensitive groups (used for old mean metric)

    Returns
    ----------
    unfairness: float
        Updated unfairness value of the model combination.
    """
    if metric == "demographic_parity":
        unfairness = unfairness + abs(model["pppv"].iloc[0] - total_pppv[0])
    elif metric == "equalized_odds":
        if model["pppv_y0"].iloc[0] != -1 and model["pppv_y1"].iloc[0] != -1 :
            unfairness = (unfairness + 0.5*abs(model["pppv_y0"].iloc[0] - total_pppv[1])
                + 0.5*abs(model["pppv_y1"].iloc[0] - total_pppv[2]))
        elif model["pppv_y0"].iloc[0] == -1:
            unfairness = unfairness + abs(model["pppv_y1"].iloc[0] - total_pppv[2])
        elif model["pppv_y1"].iloc[0] == -1:
            unfairness = unfairness + abs(model["pppv_y0"].iloc[0] - total_pppv[1])
    elif metric == "equal_opportunity":
        if model["pppv_y1"].iloc[0] != -1:
            unfairness = unfairness + abs(model["pppv_y1"].iloc[0] - total_pppv[2])
    elif metric == "treatment_equality":
        if model["fp"].iloc[0]+model["fn"].iloc[0] > 0 and total_fp+total_fn > 0:
            unfairness = (unfairness + abs(model["fp"].iloc[0]/(model["fp"].iloc[0]+model["fn"].iloc[0])
                - total_fp/(total_fp+total_fn)))
        elif model["fp"].iloc[0]+model["fn"].iloc[0] > 0:
            unfairness = (unfairness + abs(model["fp"].iloc[0]/(model["fp"].iloc[0]+model["fn"].iloc[0])
                - 0.5))
        elif total_fp+total_fn > 0:
            unfairness = unfairness + abs(0.5 - total_fp/(total_fp+total_fn))
    elif metric == "impact":
        if fav_pppv == model["pppv"].iloc[0]:
            pass
        elif fav_pppv < model["pppv"].iloc[0]:
            unfairness = unfairness + abs(1 - fav_pppv/model["pppv"].iloc[0])
        else:
            unfairness = unfairness + abs(1 - model["pppv"].iloc[0]/fav_pppv)
    elif metric == "consistency":
        if total_pppv[0] > 0.5:
            unfairness = 1 - total_pppv[0]
        else:
            unfairness = total_pppv[0]
    return unfairness


class Metrics():
    """This class is used to check the accuracy and fairness of given data via one of the
    given metrics.

    Parameters
    ----------
    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    label: string
        String name of the target column.
    """
    def __init__(self, sens_attrs, label):
        self.sens_attrs = sens_attrs
        self.label = label


    def prediction_output_add(self, df, data_input, count, index, pred_index,
        true_value, prediction, model, model_used, model_comb=None):
        """This function adds a new row to the dataframe.

        Parameters
        ----------
        df: DataFrame, shape (n_samples, m_features)
            Format of the output dataframe containing an index, sensitive attributes, label,
            predicted value, name of algorithm used, name of classifier used, combination of
            classifiers used for an entry.

        data_input: DataFrame, shape (n_samples, m_features)
            Dataframe containing informations about the values of the sensitive attributes
            for an entry.

        count: integer
            Number where the row is added in the dataframe.

        index: string
            String name of the index column.

        pred_index: integer
            Index number of the predicted entry.

        true_value: integer (0-1)
            True binary value of the entry.

        prediction: integer (0-1)
            Predicted value of the entry.

        model: string
            Name of the algorithm/model used to determine whích classifiers are used.

        model_used: string
            Name of the model/classifier used for the prediction.

        model_comb: tuple of strings
            Tuple of the combination of classifiers used for the prediction.


        Returns
        ----------
        df: Dataframe, shape (n_sample, m_features)
            Updated dataframe with the newly added row.
        """
        df.at[count, index] = pred_index
        for attr in self.sens_attrs:
            df.at[count, attr] = data_input.loc[pred_index, attr]
        df.at[count, self.label] = true_value.values[0]
        df.at[count, model] = prediction
        df.at[count, "model_used"] = model_used
        df.at[count, "model_comb"] = model_comb
        return df


    def test_score(self, df, models):
        """This function takes a dataframe with predicted labels of models as input and puts out
        how accurate and how many values are positive predicted per model and per sensitive
        group.

        Parameters
        ----------
        df: DataFrame, shape (n_samples, m_features)
            Dataframe on which the models were tested.

        models: list of strings
            List of model names which are tested on the dataframe.


        Returns
        ----------
        model_test: DataFrame, shape (n_samples, m_features)
            Each entry contains index, model name, values of the sensitive attributes, number
            and probability of positive predicted values, number and probability of wrong
            predicted label, number of entries in the group. An entry is made for each model
            + sensitive group combination.
        """
        #Create a dataframe containing all important columns of the returned dataframe.
        model_test = pd.DataFrame(columns=["model", "inacc_sum", "ppv_sum", "num_of_elem",\
            "inaccuracy", "pppv", "inacc_sum_y0", "ppv_sum_y0", "num_of_elem_y0",\
            "inaccuracy_y0", "pppv_y0", "inacc_sum_y1", "ppv_sum_y1", "num_of_elem_y1",\
            "inaccuracy_y1", "pppv_y1", "fp", "fn"])
        sens_count = 1
        for attr in self.sens_attrs:
            model_test.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        #Get the number of sensitive groups.
        groups = df[self.sens_attrs].drop_duplicates(self.sens_attrs).reset_index(drop=True)
        num_of_groups = len(groups)

        #Add every model for each sensitive group. Fill the other values with 0.
        count = 0
        for model in models:
            for i in range(num_of_groups):
                model_test.at[count, "model"] = model
                for attr in self.sens_attrs:
                    model_test.at[count, attr] = groups.at[i, attr]
                count = count + 1
            model_test.fillna(0, inplace=True)

        #Update the other column values by iterating through the input dataframe for each model
        #which contains the predicted label by each model and check how many elements there are,
        #how many are wrongly predicted and how many have a positive predicted label.
        count = 0
        for model in models:
            item_count = 0
            for index, row in df.iterrows():
                sens_list = []
                #Data is like this: model, sens_list, label,...:
                #Example: dectree, [1,0,1], 0,...
                for attr in self.sens_attrs:
                    sens_list.append(row[attr])
                true_value = row[self.label]
                pred = row[model]
                #Iterate over all sensitive groups and check for each if the sensitive attribute
                #values of the entry match them. If found, update the values and exit the for loop
                #to get the next entry.
                for i in range(num_of_groups):
                    position = count + i
                    if np.array_equal(model_test[self.sens_attrs].iloc[[position]].values,\
                        df[self.sens_attrs].iloc[[item_count]].values):
                        model_test.at[position, "inacc_sum"] += abs(true_value - pred)
                        model_test.at[position, "ppv_sum"] += pred
                        model_test.at[position, "num_of_elem"] += 1
                        if true_value == 0:
                            model_test.at[position, "inacc_sum_y0"] += abs(true_value - pred)
                            model_test.at[position, "ppv_sum_y0"] += pred
                            model_test.at[position, "num_of_elem_y0"] += 1
                            if pred == 1:
                                model_test.at[position, "fp"] += 1
                        else:
                            model_test.at[position, "inacc_sum_y1"] += abs(true_value - pred)
                            model_test.at[position, "ppv_sum_y1"] += pred
                            model_test.at[position, "num_of_elem_y1"] += 1
                            if pred == 0:
                                model_test.at[position, "fn"] += 1
                        break
                item_count = item_count + 1

            #Get the overall inaccuracy probability and probability of positive predicted values
            #for each model + sensitive group combination.
            for i in range(num_of_groups):
                position = count + i
                try:
                    model_test.at[position, "inaccuracy"] = round(model_test.at[position, "inacc_sum"]\
                        /model_test.at[position, "num_of_elem"], 8)
                    model_test.at[position, "pppv"] = round(model_test.at[position, "ppv_sum"]\
                        /model_test.at[position, "num_of_elem"], 8)
                except ZeroDivisionError:
                    model_test.at[position, "inaccuracy"] = 0
                    model_test.at[position, "pppv"] = 0

                try:
                    model_test.at[position, "inaccuracy_y0"] = round(model_test.at[position, "inacc_sum_y0"]\
                        /model_test.at[position, "num_of_elem_y0"], 8)
                    model_test.at[position, "pppv_y0"] = round(model_test.at[position, "ppv_sum_y0"]\
                        /model_test.at[position, "num_of_elem_y0"], 8)
                except ZeroDivisionError:
                    model_test.at[position, "inaccuracy_y0"] = 0
                    model_test.at[position, "pppv_y0"] = -1

                try:
                    model_test.at[position, "inaccuracy_y1"] = round(model_test.at[position, "inacc_sum_y1"]\
                        /model_test.at[position, "num_of_elem_y1"], 8)
                    model_test.at[position, "pppv_y1"] = round(model_test.at[position, "ppv_sum_y1"]\
                        /model_test.at[position, "num_of_elem_y1"], 8)
                except ZeroDivisionError:
                    model_test.at[position, "inaccuracy_y1"] = 0
                    model_test.at[position, "pppv_y1"] = -1

            count = count + num_of_groups

        return model_test


    def test_score_sbt(self, df, dictionary):
        """This function takes a dataframe with predicted labels of models as input and puts out
        how accurate and how many values are positive predicted per model and per sensitive
        group.

        Parameters
        ----------
        df: DataFrame, shape (n_samples, m_features)
            Dataframe on which the models were tested.

        dictionary: dictionary
            This dictionary contains the models and on which groups they have been trained.


        Returns
        ----------
        model_test: DataFrame, shape (n_samples, m_features)
            Each entry contains index, model name, values of the sensitive attributes, number
            and probability of positive predicted values, number and probability of wrong
            predicted label, number of entries in the group. An entry is made for each model
            + sensitive group combination.
        """
        #Create a dataframe containing all important columns of the returned dataframe.
        model_test = pd.DataFrame(columns=["model", "inacc_sum", "ppv_sum", "num_of_elem",\
            "inaccuracy", "pppv", "inacc_sum_y0", "ppv_sum_y0", "num_of_elem_y0",\
            "inaccuracy_y0", "pppv_y0", "inacc_sum_y1", "ppv_sum_y1", "num_of_elem_y1",\
            "inaccuracy_y1", "pppv_y1", "fp", "fn"])
        sens_count = 1
        for attr in self.sens_attrs:
            model_test.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        #Add every model for each sensitive group. Fill the other values with 0.
        grouped_df = df.groupby(self.sens_attrs)

        count = 0
        for key, item in grouped_df:
            part_df = grouped_df.get_group(key)
            for model, value in dictionary[key].items():
                sens_attr_col = 0
                model_test.at[count, "model"] = model
                model_test.at[count, "inacc_sum"] = 0
                model_test.at[count, "ppv_sum"] = 0
                model_test.at[count, "num_of_elem"] = 0
                model_test.at[count, "inacc_sum_y0"] = 0
                model_test.at[count, "ppv_sum_y0"] = 0
                model_test.at[count, "num_of_elem_y0"] = 0
                model_test.at[count, "fp"] = 0
                model_test.at[count, "inacc_sum_y1"] = 0
                model_test.at[count, "ppv_sum_y1"] = 0
                model_test.at[count, "num_of_elem_y1"] = 0
                model_test.at[count, "fn"] = 0
                if len(self.sens_attrs) == 1:
                    model_test.at[count, self.sens_attrs[0]] = key
                else:
                    for attr in self.sens_attrs:
                        model_test.at[count, attr] = key[sens_attr_col]
                        sens_attr_col += 1
                for index, row in part_df.iterrows():
                    true_value = row[self.label]
                    pred = row[model]
                    model_test.at[count, "inacc_sum"] += abs(true_value - pred)
                    model_test.at[count, "ppv_sum"] += pred
                    model_test.at[count, "num_of_elem"] += 1
                    if true_value == 0:
                        model_test.at[count, "inacc_sum_y0"] += abs(true_value - pred)
                        model_test.at[count, "ppv_sum_y0"] += pred
                        model_test.at[count, "num_of_elem_y0"] += 1
                        if pred == 1:
                            model_test.at[count, "fp"] += 1
                    else:
                        model_test.at[count, "inacc_sum_y1"] += abs(true_value - pred)
                        model_test.at[count, "ppv_sum_y1"] += pred
                        model_test.at[count, "num_of_elem_y1"] += 1
                        if pred == 0:
                            model_test.at[count, "fn"] += 1
                count = count + 1

        for i in range(len(model_test)):
            try:
                model_test.at[i, "inaccuracy"] = round(model_test.at[i, "inacc_sum"]\
                    /model_test.at[i, "num_of_elem"], 8)
                model_test.at[i, "pppv"] = round(model_test.at[i, "ppv_sum"]\
                    /model_test.at[i, "num_of_elem"], 8)
            except ZeroDivisionError:
                model_test.at[i, "inaccuracy"] = 0
                model_test.at[i, "pppv"] = 0

            try:
                model_test.at[i, "inaccuracy_y0"] = round(model_test.at[i, "inacc_sum_y0"]\
                    /model_test.at[i, "num_of_elem_y0"], 8)
                model_test.at[i, "pppv_y0"] = round(model_test.at[i, "ppv_sum_y0"]\
                    /model_test.at[i, "num_of_elem_y0"], 8)
            except ZeroDivisionError:
                model_test.at[i, "inaccuracy_y0"] = 0
                model_test.at[i, "pppv_y0"] = 0

            try:
                model_test.at[i, "inaccuracy_y1"] = round(model_test.at[i, "inacc_sum_y1"]\
                    /model_test.at[i, "num_of_elem_y1"], 8)
                model_test.at[i, "pppv_y1"] = round(model_test.at[i, "ppv_sum_y1"]\
                    /model_test.at[i, "num_of_elem_y1"], 8)
            except ZeroDivisionError:
                model_test.at[i, "inaccuracy_y1"] = 0
                model_test.at[i, "pppv_y1"] = 0

        return model_test


    def fairness_metric(self, df, model_comb, favored, metric, weight=0.5, threshold=0,
        comb_amount=1):
        """This function returns the best combination(s) of models for the tested dataset region.

        Parameters
        ----------
        df: DataFrame, shape (n_samples, m_features)
            Each entry contains index, model name, values of the sensitive attributes, number
            and probability of positive predicted values, number and probability of wrong
            predicted label, number of entries in the group. An entry is made for each model
            + sensitive group combination.

        models: list of strings OR list of tuples of strings
            IF list of strings: List of all model names which are considered. Each possible
            combinations of each models are created.
            IF list of tuples: List of fixed combinations which are considered.

        favored: tuple of float
            Tuple of the values of the favored group.

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


        Returns
        ----------
        best_comb_list: list of list of strings
            List of best model combinations.

        group_list: list of float
            List of the values of the sensitive attributes/keys for each sensitive group.
        """
        best_comb_val = math.inf
        grouped_df = df.groupby(self.sens_attrs)
        group_list = []
        #TypeError raises if i[0] is a float instead of a tuple, which happens if only one
        #sensitive attribute is chosen. This is a workaround so it can be treated the same
        #as if multiple sensitive attributes ar chosen.
        try:
            for i in grouped_df:
                group_list.append(list(i[0]))
        except TypeError:
            group_list = []
            for i in grouped_df:
                group_list_2 = []
                group_list_2.append(i[0])
                group_list.append(group_list_2)

        best_comb = model_comb[0]
        best_comb_list = []
        all_comb_list = []
        for comb in model_comb:
            count = 0
            inaccuracy = 0
            unfairness = 0
            sum_of_elem = 0
            wrong_predicted = 0
            total_ppv = 0
            total_size = 0
            total_ppv_y0 = 0
            total_size_y0 = 0
            total_ppv_y1 = 0
            total_size_y1 = 0
            total_fp = 0
            total_fn = 0
            #Get the favored group values and total values to test against.
            for key, item in grouped_df:
                part_df = grouped_df.get_group(key)
                model = part_df[part_df.model == comb[count]]
                total_ppv = total_ppv + model["ppv_sum"].iloc[0]
                total_size = total_size + model["num_of_elem"].iloc[0]
                total_ppv_y0 = total_ppv_y0 + model["ppv_sum_y0"].iloc[0]
                total_size_y0 = total_size_y0 + model["num_of_elem_y0"].iloc[0]
                total_ppv_y1 = total_ppv_y1 + model["ppv_sum_y1"].iloc[0]
                total_size_y1 = total_size_y1 + model["num_of_elem_y1"].iloc[0]
                total_fp = total_fp + model["fp"].iloc[0]
                total_fn = total_fn + model["fn"].iloc[0]
                if key == favored:
                    fav_pppv = model["pppv"].iloc[0]
                    fav_ppv = model["ppv_sum"].iloc[0]
                    fav_size = model["num_of_elem"].iloc[0]
                    fav_pppv_y0 = model["pppv_y0"].iloc[0]
                    fav_ppv_y0 = model["ppv_sum_y0"].iloc[0]
                    fav_size_y0 = model["num_of_elem_y0"].iloc[0]
                    fav_pppv_y1 = model["pppv_y1"].iloc[0]
                    fav_ppv_y1 = model["ppv_sum_y1"].iloc[0]
                    fav_size_y1 = model["num_of_elem_y1"].iloc[0]
                count = count + 1
            total_pppv_list = []
            total_pppv_list.append(total_ppv/total_size)
            if total_size_y0 != 0:
                total_pppv_list.append(total_ppv_y0/total_size_y0)
            else:
                total_pppv_list.append(-1)
            if total_size_y1 != 0:
                total_pppv_list.append(total_ppv_y1/total_size_y1)
            else:
                total_pppv_list.append(-1)
            fav_pppv_list = []
            fav_pppv_list.append(fav_pppv)
            fav_pppv_list.append(fav_pppv_y0)
            fav_pppv_list.append(fav_pppv_y1)
            #Iterate again for formula.
            count = 0
            for key, item in grouped_df:
                part_df = grouped_df.get_group(key)
                #Get only the model of the combination for the current sensitive group.
                model = part_df[part_df.model == comb[count]]
                sum_of_elem = sum_of_elem + model["num_of_elem"].iloc[0]
                #EXPAND THIS ONE FOR ACCURACY METRICS?##############################################
                wrong_predicted = wrong_predicted + model["inacc_sum"].iloc[0]
                #CALL EXPANDABLE FAIRNESS_SCORE FUNCTION############################################
                unfairness = fairness_score(metric, unfairness, model, fav_pppv_list,\
                    total_pppv_list, total_fp, total_fn, len(grouped_df))
                ####################################################################################
                count = count + 1
            inaccuracy = wrong_predicted/sum_of_elem
            #Overall metric score of the current model combination.
            if metric in ("mean", "elift"):
                new_comb_val = weight * inaccuracy + (1 - weight)/len(grouped_df) * unfairness
            elif metric == "old-mean":
                new_comb_val = weight * inaccuracy + (1 - weight)/total_size * unfairness
            else:
                new_comb_val = weight * inaccuracy + (1 - weight)/(len(grouped_df)-1) * unfairness
            #If the current model combination outperforms the current best score, define this one
            #as best combination (for now). If the value stays under the given threshold, add this
            #combination to the list containing the best model combinations.
            if isinstance(new_comb_val, pd.Series):
                new_comb_val = new_comb_val.values[0]
            if new_comb_val < best_comb_val:
                best_comb_val = new_comb_val
                best_comb = comb
            if new_comb_val <= threshold:
                best_comb_list.append(comb)
            all_comb_list.append((comb, new_comb_val))

        group_list = tuple(group_list)

        #If comb_amount is set to a value x > 1. Return the x best model combinations.
        #If neither the comb_amount or threshold are manually set to a different value
        #OR if no combination is under the given threshold, return only the best model
        #combination as a list (so it can be handled like if multiple combinations are returned).
        #Else (threshold manually set) return a list of all combinations of models which metric
        #score are under the given threshold.
        comb_amount = min(comb_amount, len(all_comb_list))
        if comb_amount != 1:
            sorted_list = sorted(all_comb_list, key=lambda x: x[1])
            best_comb_list = []
            for i in range(comb_amount):
                best_comb_list.append(sorted_list[i][0])
        elif (comb_amount == 1 and threshold == 0) or (best_comb_list == []):
            best_comb_list = list([best_comb])

        return best_comb_list, group_list
