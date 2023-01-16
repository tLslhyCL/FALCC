"""
This python file uses each single classifier to predict the label.
"""
import re
import pandas as pd
import joblib

class Classifier:
    """This class is used to use each single trained classifier.

    Parameters
    ----------
    index: string
        String name of the index.

    pred_id_list: list of ids
        List of all indices of the prediction dataset.

    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    label: string
        String name of the target column.

    model_list: list of strings
        List of the model names.
    """
    def __init__(self, index, pred_id_list, sens_attrs, label, model_list):
        self.index = index
        self.pred_id_list = pred_id_list
        self.sens_attrs = sens_attrs
        self.label = label
        self.model_list = model_list


    def single_class_pred(self, X_pred, y_pred):
        """This function puts out a result dataset for each trained model of the model list.

        Parameters
        ----------
        X_pred: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_pred: array-like, shape (n_samples)
            Label vector relative to the prediction data X_pred.


        Returns/Output
        ----------
        "[model]_prediction_output.csv": Output DataFrame file in .csv format for each model.
            Contains: index, value of sensitive attributes, label, predicted value.
        """
        #Create dataframe and add the sensitive attributes as columns. For each model predict the
        #label of each entry of the prediction dataset by using the trained models.
        for model in self.model_list:
            prediction_df = pd.DataFrame(columns=[self.index, self.label, model])
            count = 0
            sens_count = 1

            for attr in self.sens_attrs:
                prediction_df.insert(sens_count, attr, None)
                sens_count = sens_count + 1

            for i, row in X_pred.iterrows():
                prediction_df.at[count, self.index] = self.pred_id_list[count]
                for attr in self.sens_attrs:
                    prediction_df.at[count, attr] = row[attr]
                prediction_df.at[count, self.label] = y_pred.iloc[count].values[0]

                joblib_file = model
                used_model = joblib.load(joblib_file)
                prediction = used_model.predict(X_pred.iloc[count].values.reshape(1, -1))
                prediction_df.at[count, model] = prediction[0]

                count = count + 1

            prediction_df.to_csv(re.sub(".pkl", "", model) + "_prediction_output.csv", index=False)
