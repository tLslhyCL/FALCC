"""
Call methods for the offline phase of the FALCC algorithm. Also runs the offline phase of the
FALCES [2] algorithm and its variants & runs the other algorithms.
"""
import warnings
import argparse
import shelve
import ast
import copy
import math
import joblib
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from kneed import KneeLocator
import algorithm
from algorithm.codes import Metrics
from algorithm.parameter_estimation import log_means
#from algorithm.codes.FaX_AI.FaX_methods import MIM
#from algorithm.codes.Fair_SMOTE.SMOTE import smote
#from algorithm.codes.Fair_SMOTE.Generate_Samples import generate_samples
#from algorithm.codes.iFair_helper.iFair import iFair
from sklearn.linear_model import LogisticRegression
from aif360.algorithms.preprocessing import *
from aif360.datasets import BinaryLabelDataset



warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="Directory and name of the input .csv file.")
parser.add_argument("-o", "--output", type=str, help="Directory of the generated output files.")
parser.add_argument("--testsize", default=0.5, type=float, help="Dataset is randomly split into\
    training and test datasets. This value indicates the size of the test dataset. Default value: 0.5")
parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
    of each entry. Default given column name: index.")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--sbt", type=str, default=True, help="Choose if you want to split the data\
    before training each classifier. Default: True")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--metric", default="mean", type=str, help="Metric which will be used to test\
    the classifier combinations. Default metric: mean.")
parser.add_argument("--weight", default=0.5, type=float, help="The weight value is used for\
    balancing the importance of accuracy vs fairness. Default value: 0.5")
parser.add_argument("--preprocessed", default=True, type=str, help="Indicates if the dataset\
    has been preprocessed properly. It is a requirement for the cluster variant. Default value: True")
parser.add_argument("--algorithm", default="cluster", type=str, help="Name of the algorithm\
    which should be used. For evaluation purposes, this is currently ignored. Default: Cluster.")
parser.add_argument("--trained_models", default=None, type=str, help="Location of the trained models in .pkl\
    format.")
parser.add_argument("--training", default="adaboost", type=str, help="Name of the model training\
    strategy. If set to None, it requires --models to be set. Default: adaboost (AdaptedAdaBoost).")
parser.add_argument("--testall", default=True, type=str, help="If set to True, all other implemented\
    algorithms will be run as well. Default: True.")
parser.add_argument("--ignore_sens", default=False, type=str, help="If set to True, the protected attribute(s)\
    will be further ignored for the training and prediction phase. Default: False.")
#Some of the following variables are not necessary for all algorithm types.
parser.add_argument("--favored", default=None, type=str, help="Tuple of favored group.\
    Otherwise some metrics can't be used. Default: None.")
parser.add_argument("--threshold", default=0, type=float, help="Gives a threshold for the best\
    global combinations. Needed for the performance-efficient algorithm iff comb-amount is not set.")
parser.add_argument("--comb_amount", default=20, type=int, help="Gives the amount of best global\
    combinations which will be used for the local combination search. This or threshold are needed\
    for performance-efficient algorithm. Default: 20")
parser.add_argument("--knn", default=15, type=int, help="Amount of kNN considered per group for\
    the metrics of the FALCES algorithms without clustering. Default: 15")
parser.add_argument("--falces_version", "--fv", default="NEW", type=str, help="Which FALCES\
    variant (without clustering) is used. The 'NEW' one improves the runtime. Default: NEW")
parser.add_argument("--modelsize", default=6, type=int, help="Number of iterations for the\
    AdaBoost classifier of the clustering algorithm variant. Default: 10")
parser.add_argument("--ccr", default="[-1,-1]", type=str, help="Minimum and maximum amount of clusters\
    that should be generated. If value = -1 the parameter will be estimated. Default: [-1,-1]")
parser.add_argument("--cluster_algorithm", "--ca", default="elbow", type=str, help="Name of the\
    clustering algorithm that should be used to estimate the amount of clusters, if no clustersize\
    is given. Currently supported: [LOGmeans, elbow]. Default: elbow")
parser.add_argument("--fairboost", default=True, type=str, help="Set to true if the FairBoost\
    should and can be run. Default: True")
parser.add_argument("--ifair", default=True, type=str, help="Set to true if the iFair algorithm\
    should and can be run. Default: True")
parser.add_argument("--proxy", default="no", type=str, help="Set if proxy technique should be used.\
    Options: no, reweigh, remove. Default: no.")
parser.add_argument("--allowed", type=str, help="List of attributes which should not be reweighed.")
#Prediction size & randomstate is only important for the evaluation of the model.
parser.add_argument("--predsize", default=0.3, type=float, help="Size of the prediction dataset in\
    relation to the test dataset (only used for the evaluation). Default value: 0.")
parser.add_argument("--randomstate", default=-1, type=int, help="Randomstate of the splits.")
args = parser.parse_args()

input_file = args.input
link = args.output
testsize = float(args.testsize)
index = args.index
sens_attrs = ast.literal_eval(args.sensitive)
if args.sbt == "True":
    sbt = True
else:
    sbt = False
label = args.label
metric = args.metric
weight = float(args.weight)
if args.preprocessed == "True":
    pre_processed = True
else:
    pre_processed = True
algo_type = args.algorithm
favored = ast.literal_eval(args.favored)
threshold = float(args.threshold)
comb_amount = int(args.comb_amount)
knn_size = int(args.knn)
falces_version = args.falces_version
modelsize = int(args.modelsize)
ccr = ast.literal_eval(args.ccr)
predsize = float(args.predsize)
randomstate = int(args.randomstate)
cluster_algorithm = args.cluster_algorithm
training = args.training
proxy = args.proxy
allowed = ast.literal_eval(args.allowed)
if args.testall == "True":
    testall = True
else:
    testall = False
if args.ignore_sens == "True":
    ignore_sens = True
else:
    ignore_sens = False
if args.fairboost == "True":
    fairboost = True
else:
    fairboost = False
if args.ifair == "True":
    ifair = True
else:
    ifair = False
if args.trained_models != None:
    trained_models = ast.literal_eval(args.trained_models)

if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)

if training == "fair":
    fairinput = True
else:
    fairinput = False

#Read the input dataset & split it into training, test & prediction dataset.
#Prediction dataset only needed for evaluation, otherwise size is automatically 0.
df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
#Hard set atm

X = df.loc[:, df.columns != label]
y = df[label]

X_train, X_testphases, y_train, y_testphases = train_test_split(X, y, test_size=testsize,
    random_state=randomstate)
X_test, X_pred, y_test, y_pred = train_test_split(X_testphases, y_testphases,
    test_size=predsize, random_state=randomstate)

train_id_list = []
for i, row in X_train.iterrows():
    train_id_list.append(i)

test_id_list = []
for i, row in X_test.iterrows():
    test_id_list.append(i)

pred_id_list = []
for i, row in X_pred.iterrows():
    pred_id_list.append(i)

y_train = y_train.to_frame()
y_test = y_test.to_frame()
y_pred = y_pred.to_frame()

#AdaBoost Training & Testing results of each classifier.
if training == "single_classifiers":
    model_training_list = ["DecisionTree", "LinearSVM", "NonlinearSVM",\
        "LogisticRegression", "SoftmaxRegression"]
elif training == "fair":
    model_training_list = ["FaX", "Fair-SMOTE", "LFR"]
elif training == "opt_random_forest":
    model_training_list = ["OptimizedRandomForest"]
elif training == "opt_adaboost":
    model_training_list = ["OptimizedAdaBoost"]


if training != "no":
    run_main = algorithm.RunTraining(X_test, y_test, test_id_list, sens_attrs, index, label, favored, link,
        ignore_sens)
    if not "sample_weight" in locals():
        sample_weight = None

    test_df, d, model_list, model_comb = run_main.train(model_training_list, X_train, y_train,
        sample_weight, modelsize, [])
    test_df.to_csv(link + "testdata_predictions.csv", index_label=index)
    test_df = test_df.sort_index()

    if training != "fair":
        key_list = []
        grouped_df = df.groupby(sens_attrs)
        for key, items in grouped_df:
            key_list.append(key)
        test_df_sbt, d_sbt, model_list_sbt, model_comb_sbt = run_main.sbt_train(model_training_list,
            X_train, y_train, train_id_list, sample_weight, key_list, modelsize, [])
        test_df_sbt.to_csv(link + "testdata_sbt_predictions.csv", index_label=index)
        test_df_sbt = test_df_sbt.sort_index()
else:
    d = dict()
    model_list = []
    test_df = pd.DataFrame(columns=[index, label])
    test_df[index] = list(y_test.index)
    test_df[label] = y_test[label]
    for sens in sens_attrs:
        test_df[sens] = X_test[sens]
    for tm in trained_models:
        used_model = joblib.load(tm)
        prediction = used_model.predict(X_test)
        test_df[tm] = prediction
        model_list.append(tm)
        d_list = []
        d_list.append(tm)
        d_list.append(prediction)
        d[tm] = d_list
    groups = len(df.groupby(sens_attrs))
    model_comb = list(itertools.combinations_with_replacement(model_list, groups))
    
    test_df.to_csv(link + "testdata_predictions.csv", index_label=index)
    test_df = test_df.sort_index()


#Find the best global model combinations.
#Needed for FALCES-PFA & FALCES-PFA-SBT.
metricer = Metrics(sens_attrs, label)
model_test = metricer.test_score(test_df, model_list)
model_test.to_csv(link + "inaccuracy_testphase.csv", index_label=index)
if training != "no" and training != "fair":
    model_test_sbt = metricer.test_score_sbt(test_df_sbt, d_sbt)
    model_test_sbt.to_csv(link + "inaccuracy_testphase_sbt.csv", index_label=index)
    model_test_sbt = model_test_sbt.sort_index()


if testall == True:
    #Decouple Algorithm Predictions
    if training != "no" and training != "fair":
        decouple_alg = algorithm.Decouple(metricer, index, pred_id_list, sens_attrs, label, favored, model_comb_sbt, link, fairinput)
        df_result = decouple_alg.decouple(model_test_sbt, X_pred, y_pred, metric, weight, sbt=True)
        df_result.to_csv(link + "Decouple-SBT_prediction_output.csv", index=False)
    decouple_alg = algorithm.Decouple(metricer, index, pred_id_list, sens_attrs, label, favored, model_comb, link, fairinput)
    df_result = decouple_alg.decouple(model_test, X_pred, y_pred, metric, weight, sbt=False)
    df_result.to_csv(link + "Decouple_prediction_output.csv", index=False)

    
    if len(sens_attrs) == 1:
        #Run the FairBoost Algorithm
        if fairboost:
            fb = algorithm.FairBoost(index, pred_id_list, sens_attrs, favored, label, DecisionTreeClassifier())
            df_result = fb.fit_predict(X_train, y_train, X_pred, y_pred, r=0.1)
            df_result.to_csv(link + "FairBoost_prediction_output.csv", index=False)

        if ifair:
            classifier = LogisticRegression(solver='lbfgs')
            sens_attrs_ids = []
            for sens in sens_attrs:
                sens_attrs_ids.append(X_train.columns.get_loc(sens))
            sens_attrs_ids.sort(reverse=True)
            X_train = X_train.to_numpy()
            model = iFair(max_iter=10)
            model.fit(X_train, sens_attrs_ids)
            X_test = self.X_test.to_numpy()
            X_train_new = model.transform(X_train)

            classifier.fit(X_train_new, y_train.to_numpy())
            pred = classifier.predict(X_pred)
            res_df = copy.deepcopy(X_pred)
            res_df["iFair"] = pred
            res_df.to_csv(link + "iFair_prediction_output.csv")

        #FaX Classifier Predictions
        ##Only work for binary groups
        X2 = X_train.loc[:, X_train.columns != sens_attrs[0]]
        Z2 = X_train[sens_attrs[0]].to_frame()
        Y2 = y_train

        X3 = X_pred.loc[:, X_pred.columns != sens_attrs[0]]
        Z3 = X_pred[sens_attrs[0]].to_frame()
        Y3 = y_pred

        fax = MIM(X2, Z2, Y2)
        prediction = fax.predict(X3)
        res_df = copy.deepcopy(X_pred)
        res_df["FaX"] = prediction
        res_df.to_csv(link + "FaX_prediction_output.csv")

        #SMOTE Predictions
        train_df = copy.deepcopy(X_train)
        train_df[label] = y_train
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
            & (dataset_orig_train[sens_attrs[0]] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train[label] == 0)
            & (dataset_orig_train[sens_attrs[0]] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train[label] == 1)
            & (dataset_orig_train[sens_attrs[0]] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train[label] == 1)
            & (dataset_orig_train[sens_attrs[0]] == 1)])
        maximum = max(zero_zero,zero_one,one_zero,one_one)
        if maximum == zero_zero:
            zero_one_to_be_incresed = maximum - zero_one
            one_zero_to_be_incresed = maximum - one_zero
            one_one_to_be_incresed = maximum - one_one
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_one)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_one_one)
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_new = df_new.append(df_zero_zero)
        if maximum == zero_one:
            zero_zero_to_be_incresed = maximum - zero_zero
            one_zero_to_be_incresed = maximum - one_zero
            one_one_to_be_incresed = maximum - one_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_one_one)
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_new = df_new.append(df_zero_one)
        if maximum == one_zero:
            zero_zero_to_be_incresed = maximum - zero_zero
            zero_one_to_be_incresed = maximum - zero_one
            one_one_to_be_incresed = maximum - one_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_zero_one)
            df_new = df_new.append(df_one_one)
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_new = df_new.append(df_one_zero)
        if maximum == one_one:
            zero_zero_to_be_incresed = maximum - zero_zero
            one_zero_to_be_incresed = maximum - one_zero
            zero_one_to_be_incresed = maximum - zero_one
            df_zero_zero = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_one_zero = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 0)]
            df_zero_one = dataset_orig_train[(dataset_orig_train[label] == 0)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'',dict_cols)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'',dict_cols)
            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'',dict_cols)
            df_new = copy.deepcopy(df_zero_zero)
            df_new = df_new.append(df_one_zero)
            df_new = df_new.append(df_zero_one)
            df_one_one = dataset_orig_train[(dataset_orig_train[label] == 1)
                & (dataset_orig_train[sens_attrs[0]] == 1)]
            df_new = df_new.append(df_one_one)

        X_train_new, y_train_new = df_new.loc[:, df_new.columns != label], df_new[label]
        clf = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        clf.fit(X_train_new, y_train_new)
        prediction = clf.predict(X_pred)
        res_df = copy.deepcopy(X_pred)
        res_df["Fair-SMOTE"] = prediction
        res_df.to_csv(link + "Fair-SMOTE_prediction_output.csv")


        #LFR Predictions
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        pred_df = pd.merge(X_pred, y_pred, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[label], protected_attribute_names=sens_attrs)
        dataset_pred = BinaryLabelDataset(df=pred_df, label_names=[label], protected_attribute_names=sens_attrs)
        full_dataset = BinaryLabelDataset(df=df, label_names=[label], protected_attribute_names=sens_attrs)

        ###Only binary now
        privileged_groups = []
        unprivileged_groups = []
        priv_dict = dict()
        unpriv_dict = dict()
        priv_val = favored
        if favored == 0:
            priv_dict[sens_attrs[0]] = 0
            unpriv_dict[sens_attrs[0]] = 1
        elif favored == 1:
            priv_dict[sens_attrs[0]] = 1
            unpriv_dict[sens_attrs[0]] = 0

        privileged_groups = [priv_dict]
        unprivileged_groups = [unpriv_dict]

        model = LFR(unprivileged_groups, privileged_groups)
        model = model.fit(dataset_train)
        dataset_transf_train = model.transform(dataset_train)
        dataset_transf_pred = model.transform(dataset_pred)

        preds = list(dataset_transf_pred.labels)
        prediction = [preds[i][0] for i in range(len(preds))]
        res_df = copy.deepcopy(X_pred)
        res_df["LFR"] = prediction
        res_df.to_csv(link + "LFR_prediction_output.csv")



#Estimate the clustersize and then create the clusters
if proxy == "reweigh":
    with open(link + "reweighing_attributes.txt", 'w') as outfile:
        X_test_new = copy.deepcopy(X_test)
        df_new = copy.deepcopy(df)
        weight_dict = dict()
        cols = list(df_new.columns)
        cols.remove(label)
        for sens in sens_attrs:
            cols.remove(sens)

        for col in cols:
            if col in allowed:
                weight_dict[col] = 1
                continue
            x_arr = df_new[col].to_numpy()
            col_diff = 0
            for sens in sens_attrs:
                z_arr = df_new[sens]
                sens_corr = abs(pearsonr(x_arr, z_arr)[0])
                if math.isnan(sens_corr):
                    sens_corr = 1
                col_diff += (1 - sens_corr)
            col_weight = col_diff/len(sens_attrs)
            weight_dict[col] = col_weight
            df_new[col] *= col_weight
            X_test_new[col] *= col_weight
            outfile.write(col + ": " + str(col_weight) + "\n")
    df_new.to_csv("Datasets/reweigh/" + input_file + ".csv", index_label=index)
elif proxy == "remove":
    with open(link + "removed_attributes.txt", 'w') as outfile:
        X_test_new = copy.deepcopy(X_test)
        df_new = copy.deepcopy(df)
        weight_dict = dict()
        cols = list(df_new.columns)
        cols.remove(label)
        for sens in sens_attrs:
            cols.remove(sens)

        for col in cols:
            cont = False
            if col in allowed:
                weight_dict[col] = 1
                continue
            x_arr = df_new[col].to_numpy()
            col_diff = 0
            for sens in sens_attrs:
                z_arr = df_new[sens]
                pearson = pearsonr(x_arr, z_arr)
                sens_corr = abs(pearson[0])
                if math.isnan(sens_corr):
                    sens_corr = 1
                if sens_corr > 0.5 and pearson[1] < 0.05:
                    X_test_new = X_test_new.loc[:, X_test_new.columns != col]
                    cont = True
                    outfile.write(col + "\n")
                    continue
            if not cont:
                weight_dict[col] = 1
        df_new.to_csv("Datasets/removed/" + input_file + ".csv", index_label=index)
else:
    X_test_new = X_test

X_test_cluster = copy.deepcopy(X_test_new)
for sens in sens_attrs:
    X_test_cluster = X_test_cluster.loc[:, X_test_cluster.columns != sens]

#If the clustersize is fixed (hence min and max clustersize has the same value)
if ccr[0] == ccr[1] and ccr[0] != -1:
    clustersize = ccr[0]
else:
    sens_groups = len(X_test_new.groupby(sens_attrs))
    if ccr[0] == -1:
        min_cluster = min(len(X_test_cluster.columns), int(len(X_test_cluster)/(50*sens_groups)))
    else:
        min_cluster = ccr[0]
    if ccr[1] == -1:
        max_cluster = min(int(len(X_test_cluster.columns)**2/2), int(len(X_test_cluster)/(10*sens_groups)))
    else:
        max_cluster = ccr[1]

    #ELBOW
    #The following implements the Elbow Method, using the KneeLocator to perform the
    #manual step of finding the elbow point.
    if cluster_algorithm == "elbow":
        k_range = range(min_cluster, max_cluster)
        inertias = []
        for k in k_range:
            km = KMeans(n_clusters = k)
            km.fit(X_test_cluster)
            inertias.append(km.inertia_)
        y = np.zeros(len(inertias))

        kn = KneeLocator(k_range, inertias, curve='convex', direction='decreasing')
        clustersize = kn.knee - 1

    #LOGMEANS
    if cluster_algorithm == "LOGmeans":
        clustersize = log_means(X_test_cluster, min_cluster, max_cluster)


#Save the number of generated clusters as metadata
with open(link + "clustersize.txt", 'w') as outfile:
    outfile.write(str(clustersize))

#Apply the k-means algorithm on the validation dataset
kmeans = KMeans(clustersize).fit(X_test_cluster)
cluster_results = kmeans.predict(X_test_cluster)
X_test_cluster["cluster"] = cluster_results

#Shelve all variables and save it the folder.
filename = link + "cluster.out"
my_shelf = shelve.open(filename, 'n')
for key in dir():
    try:
        my_shelf["kmeans"] = kmeans
    except:
        pass
my_shelf.close()

if proxy == "no":
    weight_dict = None

#Run all offline versions of the FALCC and FALCES algorithms
falcc = algorithm.FALCC(metricer, index, sens_attrs, label, favored, model_list, X_test,
    model_comb, d, proxy, link, fairinput, weight_dict, ignore_sens, pre_processed)
model_dict, kmeans = falcc.cluster_offline(X_test_cluster, kmeans, test_df, metric, weight, sbt=False)

if training != "no" and training != "fair":
    falccsbt = algorithm.FALCC(metricer, index, sens_attrs, label, favored, model_list_sbt,
        X_test, model_comb_sbt, d_sbt, proxy, link, fairinput,  weight_dict, ignore_sens, pre_processed)
    model_dict_sbt, kmeans = falccsbt.cluster_offline(X_test_cluster, kmeans, test_df_sbt,
        metric, weight, sbt=True)

if testall == True:
    falces = algorithm.FALCES(metricer, index, sens_attrs, label, favored, model_list,
        X_test, model_comb, d, link, fairinput, pre_processed)
    global_model_comb = falces.efficient_offline(model_test, metric, weight, threshold, comb_amount)

    if training != "no" and training != "fair":
        falcessbt = algorithm.FALCES(metricer, index, sens_attrs, label, favored, model_list_sbt,
            X_test, model_comb_sbt, d_sbt, link, fairinput, pre_processed)
        global_model_comb_sbt = falcessbt.efficient_offline(model_test_sbt, metric, weight, threshold,
            comb_amount)

    falcesnew = algorithm.FALCESNew(metricer, index, sens_attrs, label, favored, model_list,
        X_test, model_comb, d, link, fairinput, pre_processed)
    if training != "no" and training != "fair":
        falcesnewsbt = algorithm.FALCESNew(metricer, index, sens_attrs, label, favored, model_list_sbt,
            X_test, model_comb_sbt, d_sbt, link, fairinput, pre_processed)


#Shelve all variables and save it the folder.
filename = link + "shelve.out"
zmy_shelf = shelve.open(filename, 'n')

for key in dir():
    try:
        zmy_shelf[key] = globals()[key]
    except:
        pass
zmy_shelf.close()
