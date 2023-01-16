"""
Here the training phase takes place, for the normal and sbt version.
"""
import itertools
import pandas as pd
from algorithm.codes import Models, ModelOps

class RunTraining:
    """This class runs the training for the algorithms.

    Parameters
    ----------
    X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
        Test data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_test: array-like, shape (n_samples)
        Label vector relative to the test data x_test.

    test_id_list: list of ids
        List of all indices of the test dataset.

    sens_attrs: list of strings
        List of the column names of sensitive attributes in the dataset.

    index: str
        Name of the index column.

    label: str
        Name of the label column.

    link: str
        String of the folder location + prefix.

    ignore_sens: bool
        Set to True if the sensitive attributes additionally should be ignored for the prediction.
    """
    def __init__(self, X_test, y_test, test_id_list, sens_attrs, index, label, link, ignore_sens=False):
        self.X_test = X_test
        self.y_test = y_test
        self.test_id_list = test_id_list
        self.sens_attrs = sens_attrs
        self.index = index
        self.label = label
        self.link = link
        self.ignore_sens = ignore_sens


    def train(self, model_training_list, X_train, y_train, sample_weight, modelsize):
        """Train all classifiers.

        Parameters
        ----------
        model_training_list: list of strings
            List containing all names of the classifiers/models that should be trained.

        X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
            Training data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_train: array-like, shape (n_samples)
            Label vector relative to the training data X_train.

        sample_weight: np.array
            Numpy array of the weight of the samples. None, if no reweighing has taken place.

        modelsize: int
            Amount of models that should be trained (for AdaBoost and Random Forest Classifiers)


        Returns/Output
        ----------
        test_df: DataFrame
            Result of the test as dataframe.

        d: dictionary
            Dictionary containing the models as keys and predictions as values.

        model_list: list of str
            List of strings of filenames of the trained models.

        model_comb: list of tuple
            List of tuples of model combinations.
        """
        model = Models(X_train, self.X_test, y_train, self.y_test, self.sens_attrs, self.ignore_sens)

        model_list = []

        MOps = ModelOps({})
        for i in model_training_list:
            if i in ("RandomForest", "AdaBoost"):
                filename_list = MOps.run(model, i, sample_weight, self.link, modelsize)
                for j in filename_list:
                    model_list.append(j)
            else:
                filename = MOps.run(model, i, sample_weight, self.link, modelsize)
                model_list.append(filename)
        d = MOps.return_dict()

        #Testphase of the trained models containing the index, label and sensitive attributes as
        #columns along the prediction of each trained model.
        test_df = pd.DataFrame(columns=[self.index, self.label])
        sens_count = 1
        count = 0
        for attr in self.sens_attrs:
            test_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        for i, val in self.y_test.iterrows():
            test_df.at[count, self.index] = self.test_id_list[count]
            for attr in self.sens_attrs:
                test_df.at[count, attr] = self.X_test.loc[i, attr]
            test_df.at[count, self.label] = val.values[0]
            count = count + 1

        col = len(test_df.columns)
        for key, value in d.items():
            test_df.insert(col, key, None)
            for i in range(len(test_df.index)):
                test_df.at[i, key] = value[1][i]

        #Create combinations of models to check.
        grouped_df = test_df.groupby(self.sens_attrs)
        model_comb = []
        for i in range(len(grouped_df)):
            model_comb.append(model_list)
        model_comb = list(itertools.product(*model_comb))
        test_df = test_df.set_index(self.index)

        return test_df, d, model_list, model_comb



    def sbt_train(self, model_training_list, X_train, y_train, train_id_list, sample_weight,
        key_list, modelsize):
        """Train all classifiers on the splitted dataset.

        Parameters
        ----------
        model_training_list: list of strings
            List containing all names of the classifiers/models that should be trained.

        X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
            Training data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y_train: array-like, shape (n_samples)
            Label vector relative to the training data x_train.

        train_id_list: list of ids
            List of all indices of the training dataset.

        sample_weight: np.array
            Numpy array of the weight of the samples. None, if no reweighing has taken place.

        key_list: list
            List of the sensitive attribute values.

        modelsize: int
            Amount of models that should be trained (for AdaBoost and Random Forest Classifiers)


        Returns/Output
        ----------
        test_df: DataFrame
            Result of the test as dataframe.

        d: dictionary
            Dictionary containing the models as keys and predictions as values.

        model_list: list of str
            List of strings of filenames of the trained models.

        model_comb: list of tuple
            List of tuples of model combinations.
        """
        d = {}
        model_list = []
        test_df = pd.DataFrame(columns=[self.index, self.label])
        sens_count = 1
        for attr in self.sens_attrs:
            test_df.insert(sens_count, attr, None)
            sens_count = sens_count + 1

        #Perform training for each classifier sparately
        for key in key_list:
            d_group = {}
            sample_weight_group = [] if sample_weight is not None else None
            add_list = []
            train_group_id_list = []
            count = 0
            for i, row in X_train.iterrows():
                if isinstance(key, tuple):
                    if key == tuple(X_train[self.sens_attrs].loc[i]):
                        add_list.append(i)
                        train_group_id_list.append(train_id_list[count])
                        if sample_weight is not None:
                            sample_weight_group.append(sample_weight[count])
                else:
                    if key == tuple(X_train[self.sens_attrs].loc[i])[0]:
                        add_list.append(i)
                        train_group_id_list.append(train_id_list[count])
                        if sample_weight is not None:
                            sample_weight_group.append(sample_weight[count])
                count += 1

            add_test = []
            test_group_id_list = []
            count = 0
            for i, row in self.X_test.iterrows():
                if isinstance(key, tuple):
                    if key == tuple(self.X_test[self.sens_attrs].loc[i]):
                        add_test.append(i)
                        test_group_id_list.append(self.test_id_list[count])
                else:
                    if key == tuple(self.X_test[self.sens_attrs].loc[i])[0]:
                        add_test.append(i)
                        test_group_id_list.append(self.test_id_list[count])
                count += 1

            X_train_group = X_train.loc[add_list]
            y_train_group = y_train.loc[add_list]
            X_test_group = self.X_test.loc[add_test]
            y_test_group = self.y_test.loc[add_test]

            model = Models(X_train_group, X_test_group, y_train_group, y_test_group, self.sens_attrs, self.ignore_sens)

            #Dictionary containing all models of the following form: {Group_key: {Model Name:
            #[(1) Saved Model as .pkl, (2) Prediction of the model for our test data,
            #(3) True label of the test data],...}, ...}
            #Train and save each model on the training data set
            MOps = ModelOps({})
            for i in model_training_list:
                if i in ("RandomForest", "AdaBoost"):
                    filename_list = MOps.run(model, i, sample_weight,
                        self.link + str(key) + "_", modelsize)
                    for j in filename_list:
                        model_list.append(j)
                else:
                    filename = MOps.run(model, i, sample_weight_group,
                        self.link + str(key) + "_", modelsize)
                    model_list.append(filename)
            d_group = MOps.return_dict()

            d[key] = d_group

            #Testphase of the trained models containing the index, label and sensitive attributes
            #as columns along the prediction of each trained model.
            test_df_group = pd.DataFrame(columns=[self.index, self.label])
            sens_count = 1
            count = 0
            for attr in self.sens_attrs:
                test_df_group.insert(sens_count, attr, None)
                sens_count = sens_count + 1

            for i, val in y_test_group.iterrows():
                test_df_group.at[count, self.index] = test_group_id_list[count]
                for attr in self.sens_attrs:
                    test_df_group.at[count, attr] = X_test_group.loc[i, attr]
                test_df_group.at[count, self.label] = val.values[0]
                count = count + 1

            col = len(test_df_group.columns)
            for modelname, value in d_group.items():
                #model = modelname + "_" + str(key)
                test_df_group.insert(col, modelname, None)
                for i in range(len(test_df_group.index)):
                    test_df_group.at[i, modelname] = value[1][i]

            test_df = pd.concat([test_df, test_df_group], axis=0, ignore_index=False)


        model_list = []
        for key, value in d.items():
            group_models = []
            for key2, value2 in d[key].items():
                group_models.append(key2)
            model_list.append(group_models)
        model_comb = list(itertools.product(*model_list))
        test_df = test_df.set_index(self.index)

        return test_df, d, model_list, model_comb
