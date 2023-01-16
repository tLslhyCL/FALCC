"""
This code coordinates the training of each classifier and calls the corresponding training
functions to be executed.
"""
import joblib


class ModelOps():
    """This class calls all classifiers to train their models.

    Parameters
    ----------
    model_dict: dictionary
        Saves information about each trained classifier.

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.
    """
    def __init__(self, model_dict):
        self.model_dict = model_dict


    def return_dict(self):
        """
        Returns the model dictionary.
        """
        return self.model_dict


    def run(self, model_obj, model, sample_weight, folder, modelsize=10):
        """Takes as input the model that will be trained and will return the trained model
        name and will save the model as .pkl & also save some informations in the dictionary.

        Parameter
        -------
        model_obj: Object
            Instance of the Model class.

        model_name: str
            Name of the classifier that should be trained and saved.

        sample_weight: np.array
            Numpy array of the weight of the samples. None, if no reweighing has taken place.

        folder: str
            String of the folder location + prefix.

        modelsize: int
            Amount of models that should be trained (for AdaBoost and Random Forest Classifiers)


        Returns
        -------
        joblib_file: str
            Name of the .pkl file of the trained classifier.
        """
        if model == "DecisionTree":
            classifier, prediction, model_name = model_obj.decision_tree(sample_weight)
        elif model == "LinearSVM":
            classifier, prediction, model_name = model_obj.linear_svm(sample_weight)
        elif model == "NonlinearSVM":
            classifier, prediction, model_name = model_obj.nonlinear_svm(sample_weight)
        elif model == "LogisticRegression":
            classifier, prediction, model_name = model_obj.log_regr(sample_weight)
        elif model == "SoftmaxRegression":
            classifier, prediction, model_name = model_obj.softmax_regr(sample_weight)
        elif model == "AdaBoost":
            classifier_list, prediction_list, model_name = model_obj.adaboost(modelsize)
            joblist_file_list = []
            with open(folder + 'adaboost.txt', 'w') as f:
                f.write(str(classifier_list))
            for i, pred in enumerate(prediction_list):
                d_list = []
                joblib_file = folder + model_name + "_" + str(i) + "_model.pkl"
                joblib.dump(classifier_list[i], joblib_file)
                d_list.append(joblib_file)
                d_list.append(pred)
                self.model_dict[joblib_file] = d_list
                joblist_file_list.append(joblib_file)

            return joblist_file_list

        d_list = []
        joblib_file = folder + model_name + "_model.pkl"
        joblib.dump(classifier, joblib_file)
        d_list.append(joblib_file)
        d_list.append(prediction)
        #Dictionary containing all models of the following form: {Model Name: [(1) Saved Model
        #as .pkl, (2) Prediction of the model for our test data]
        #Train and save each model on the training data set.
        self.model_dict[joblib_file] = d_list

        return joblib_file
