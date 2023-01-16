"""
Call methods for the offline phase of the FALCC algorithm. Also runs the offline phase of the
FALCES [2] algorithm and its variants & calls the algorithms for FairBoost [3] and Decouple [1].
[1] Dwork, C., Immorlica, N., Kalai, A., Leiserson, M. "Decoupled Classifiers for Group-Fair
    and Efficient Machine Learning". 2018.
[2] Lässig, N., Oppold, S., Herschel, M. "Metrics and Algorithms for Locally Fair and Accurate
    Classifications using Ensembles". 2022.
[3] Bhaskaruni, D., Hu, H., Lan, C. "Improving Prediction Fairness via Model Ensemble". 2019.
"""
#https://deslib.readthedocs.io/en/latest/modules/util/diversity.html
#look up predictions in dictionary
#wwrite julia
import warnings
import argparse
import os
import shelve
import re
import time
import ast
import copy
import math
import itertools
import numpy as np
import pandas as pd
import deslib.util.diversity as div
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import pearsonr
from kneed import KneeLocator
import algorithm
from algorithm.codes import Metrics
from algorithm.parameter_estimation import log_means

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
    which should be used. Default: Cluster.")
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
parser.add_argument("--proxy", default="no", type=str, help="Set if proxy technique should be used.\
    Options: no, bt, at / before and after training (Default: no).")
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
if args.trained_models != None:
    trained_models = ast.literal_eval(args.trained_models)

if randomstate == -1:
    import random
    randomstate = random.randint(1,1000)

#Read the input dataset & split it into training, test & prediction dataset.
#Prediction dataset only needed for evaluation, otherwise size is automatically 0.
df = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)


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
start = time.time()
if training == "single_classifiers":
    model_training_list = ["DecisionTree", "LinearSVM", "NonlinearSVM",\
        "LogisticRegression", "SoftmaxRegression"]
else:
    model_training_list = ["AdaBoost"]
model_training_list = ["DecisionTree", "LinearSVM", "NonlinearSVM",\
    "SoftmaxRegression", "AdaBoost"]


run_main = algorithm.RunTraining(X_test, y_test, test_id_list, sens_attrs, index, label, link,
    ignore_sens)
if not "sample_weight" in locals():
    sample_weight = None

test_df, d, model_list, model_comb = run_main.train(model_training_list, X_train, y_train,
    sample_weight, modelsize)
test_df.to_csv(link + "testdata_predictions.csv", index_label=index)
test_df = test_df.sort_index()
key_list = []
grouped_df = df.groupby(sens_attrs)
for key, items in grouped_df:
    key_list.append(key)
test_df_sbt, d_sbt, model_list_sbt, model_comb_sbt = run_main.sbt_train(model_training_list,
    X_train, y_train, train_id_list, sample_weight, key_list, modelsize)
test_df_sbt.to_csv(link + "testdata_sbt_predictions.csv", index_label=index)
test_df_sbt = test_df_sbt.sort_index()

model_comb_list = itertools.combinations(model_list, 4)
#y_results = y_test[label].to_list()

if len(sens_attrs) == 1:
    q_list = []
    for comb in model_comb_list:
        q_min = 1
        q_avg = 0
        double_comb = list(itertools.combinations(comb, 2))
        for dc in double_comb:
            y_results = []
            y1_results = []
            y2_results = []
            model1_fav = re.split("/", dc[0])
            model1_prot = copy.deepcopy(model1_fav)
            model1_fav[-1] = "1_" + model1_fav[-1]
            model1_prot[-1] = "0_" + model1_prot[-1]
            model1_fav = "/".join(model1_fav)
            model1_prot = "/".join(model1_prot)
            model2_fav = re.split("/", dc[1])
            model2_prot = copy.deepcopy(model2_fav)
            model2_fav[-1] = "1_" + model2_fav[-1]
            model2_prot[-1] = "0_" + model2_prot[-1]
            model2_fav = "/".join(model2_fav)
            model2_prot = "/".join(model2_prot)
            for i, row in y_test.iterrows():
                y_results.append(row[label])
                if X_test.at[i, sens_attrs[0]] == 0:
                    y1_results.append(test_df_sbt.at[i, model1_prot])
                    y2_results.append(test_df_sbt.at[i, model2_prot])
                else:
                    y1_results.append(test_df_sbt.at[i, model1_fav])
                    y2_results.append(test_df_sbt.at[i, model2_fav])
            q_score = div.Q_statistic(y_results, y1_results, y2_results)
            if q_score < q_min:
                q_min = q_score
            q_avg += (1 + q_score)/2
        q_avg = q_avg/len(double_comb)
        q_list.append((comb,q_avg,q_min))
    q_list.sort(key=lambda tup: tup[1], reverse=True)
else:
    q_list = []
    for comb in model_comb_list:
        q_min = 1
        q_avg = 0
        double_comb = list(itertools.combinations(comb, 2))
        for dc in double_comb:
            y_results = []
            y1_results = []
            y2_results = []
            model1_fav = re.split("/", dc[0])
            model1_prot = copy.deepcopy(model1_fav)
            model1_prot1 = copy.deepcopy(model1_fav)
            model1_prot2 = copy.deepcopy(model1_fav)
            model1_prot3 = copy.deepcopy(model1_fav)
            model1_fav[-1] = str(favored) + "_" + model1_fav[-1]
            model1_prot1[-1] = "(0, 1)_" + model1_prot[-1]
            model1_prot2[-1] = "(1, 0)_" + model1_prot[-1]
            model1_prot3[-1] = "(0, 0)_" + model1_prot[-1]
            model1_fav = "/".join(model1_fav)
            model1_prot1 = "/".join(model1_prot1)
            model1_prot2 = "/".join(model1_prot2)
            model1_prot3 = "/".join(model1_prot3)
            model2_fav = re.split("/", dc[1])
            model2_prot = copy.deepcopy(model2_fav)
            model2_prot1 = copy.deepcopy(model2_fav)
            model2_prot2 = copy.deepcopy(model2_fav)
            model2_prot3 = copy.deepcopy(model2_fav)
            model2_fav[-1] = str(favored) + "_" + model2_fav[-1]
            model2_prot1[-1] = "(0, 1)_" + model2_prot[-1]
            model2_prot2[-1] = "(1, 0)_" + model2_prot[-1]
            model2_prot3[-1] = "(0, 0)_" + model2_prot[-1]
            model2_fav = "/".join(model2_fav)
            model2_prot1 = "/".join(model2_prot1)
            model2_prot2 = "/".join(model2_prot2)
            model2_prot3 = "/".join(model2_prot3)
            for i, row in y_test.iterrows():
                y_results.append(row[label])
                if X_test.at[i, sens_attrs[0]] == 0 and X_test.at[i, sens_attrs[1]] == 1:
                    y1_results.append(test_df_sbt.at[i, model1_prot1])
                    y2_results.append(test_df_sbt.at[i, model2_prot1])
                elif X_test.at[i, sens_attrs[0]] == 1 and X_test.at[i, sens_attrs[1]] == 0:
                    y1_results.append(test_df_sbt.at[i, model1_prot2])
                    y2_results.append(test_df_sbt.at[i, model2_prot2])
                elif X_test.at[i, sens_attrs[0]] == 0 and X_test.at[i, sens_attrs[1]] == 0:
                    y1_results.append(test_df_sbt.at[i, model1_prot3])
                    y2_results.append(test_df_sbt.at[i, model2_prot3])
                else:
                    y1_results.append(test_df_sbt.at[i, model1_fav])
                    y2_results.append(test_df_sbt.at[i, model2_fav])
            q_score = div.Q_statistic(y_results, y1_results, y2_results)
            if q_score < q_min:
                q_min = q_score
            q_avg += (1 + q_score)/2
        q_avg = q_avg/len(double_comb)
        q_list.append((comb,q_avg,q_min))
    q_list.sort(key=lambda tup: tup[1], reverse=True)


with open(link + "q_scores.txt", 'w') as outfile:
    for qtup in q_list:
        outfile.write(str(qtup[0]) + " -- min: " + str(round(qtup[2],3)) + "; avg: " + str(round(qtup[1],3)) + "\n")

flag = [True, False, False, False, False, False, False, False, False, False, False]
q_val = [q_list[-1][2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
q_avg_list = [q_list[-1][1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
comb0 = False
comb1 = False
comb2 = False
comb3 = False
comb4 = False
combn0 = False
combn1 = False
combn2 = False
combn3 = False
combn4 = False
for comb in q_list:
    if comb[1] > 0.9 and not flag[10]:
        flag[10] = True
        comb4 = comb[0]
        q_val[10] = comb[2]
        q_avg_list[10] = comb[1]
    elif comb[1] > 0.8 and comb[1] <= 0.9 and not flag[9]:
        flag[9] = True
        comb3 = comb[0]
        q_val[9] = comb[2]
        q_avg_list[9] = comb[1]
    elif comb[1] > 0.7 and comb[1] <= 0.8 and not flag[8]:
        flag[8] = True
        comb2 = comb[0]
        q_val[8] = comb[2]
        q_avg_list[8] = comb[1]
    elif comb[1] > 0.6 and comb[1] <= 0.7 and not flag[7]:
        flag[7] = True
        comb1 = comb[0]
        q_val[7] = comb[2]
        q_avg_list[7] = comb[1]
    elif comb[1] > 0.5 and comb[1] <= 0.6 and not flag[6]:
        flag[6] = True
        comb0 = comb[0]
        q_val[6] = comb[2]
        q_avg_list[6] = comb[1]
    elif comb[1] > 0 and comb[1] <= 0.1 and not flag[1]:
        flag[1] = True
        combn4 = comb[0]
        q_val[1] = comb[2]
        q_avg_list[1] = comb[1]
    elif comb[1] > 0.1 and comb[1] <= 0.2 and not flag[2]:
        flag[2] = True
        combn3 = comb[0]
        q_val[2] = comb[2]
        q_avg_list[2] = comb[1]
    elif comb[1] > 0.2 and comb[1] <= 0.3 and not flag[3]:
        flag[3] = True
        combn2 = comb[0]
        q_val[3] = comb[2]
        q_avg_list[3] = comb[1]
    elif comb[1] > 0.3 and comb[1] <= 0.4 and not flag[4]:
        flag[4] = True
        combn1 = comb[0]
        q_val[4] = comb[2]
        q_avg_list[4] = comb[1]
    elif comb[1] > 0.4 and comb[1] <= 0.5 and not flag[5]:
        flag[5] = True
        combn0 = comb[0]
        q_val[5] = comb[2]
        q_avg_list[5] = comb[1]

d_list = [q_list[-1][0], combn4, combn3, combn2, combn1, combn0, comb0, comb1, comb2, comb3, comb4]


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

    #XMEANS -- Needed XMeans code relies on other implementation and is not published in our repo.
    #if cluster_algorithm == "XMeans":
    #    xm_clust = XMeans(max_cluster)
    #    xm_clust.fit(X_test_cluster.to_numpy())
    #    clustersize = xm_clust.n_clusters

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


metricer = Metrics(sens_attrs, label)
d2 = copy.deepcopy(d)

for i, comblist in enumerate(d_list):
    if not flag[i]:
        continue
    d = dict()
    for model in comblist:
        d[model] = d2[model]

    try:
        os.makedirs(link + str(i))
    except FileExistsError:
        # directory already exists
        pass

    #metricer = Metrics(sens_attrs, label)
    if len(sens_attrs) == 1:
        test_df_sbt2 = copy.deepcopy(test_df_sbt)
        comblist1 = []
        comblist2 = []
        d_sbt2 = copy.deepcopy(d_sbt)
        for m in model_list:
            m1 = re.split("/", m)
            m2 = copy.deepcopy(m1)
            m1[-1] = "0_" + m1[-1]
            m2[-1] = "1_" + m2[-1]
            m1 = "/".join(m1)
            m2 = "/".join(m2)
            if m not in comblist:
                test_df_sbt2 = test_df_sbt2.drop(columns=[m1,m2])
                del d_sbt2[0][m1]
                del d_sbt2[1][m2]
            else:
                comblist1.append(m1)
                comblist2.append(m2)
    else:
        test_df_sbt2 = copy.deepcopy(test_df_sbt)
        comblist1 = []
        comblist2 = []
        comblist3 = []
        comblist4 = []
        d_sbt2 = copy.deepcopy(d_sbt)
        for m in model_list:
            m1 = re.split("/", m)
            m2 = copy.deepcopy(m1)
            m3 = copy.deepcopy(m1)
            m4 = copy.deepcopy(m1)
            m1[-1] = "(0, 0)_" + m1[-1]
            m2[-1] = "(0, 1)_" + m2[-1]
            m1 = "/".join(m1)
            m2 = "/".join(m2)
            m3[-1] = "(1, 0)_" + m3[-1]
            m4[-1] = "(1, 1)_" + m4[-1]
            m3 = "/".join(m3)
            m4 = "/".join(m4)
            if m not in comblist:
                test_df_sbt2 = test_df_sbt2.drop(columns=[m1,m2,m3,m4])
                del d_sbt2[(0, 0)][m1]
                del d_sbt2[(0, 1)][m2]
                del d_sbt2[(1, 0)][m3]
                del d_sbt2[(1, 1)][m4]
            else:
                comblist1.append(m1)
                comblist2.append(m2)
                comblist3.append(m3)
                comblist4.append(m4)

    model_test_sbt = metricer.test_score_sbt(test_df_sbt2, d_sbt2)
    model_test_sbt.to_csv(link + "inaccuracy_testphase_sbt.csv", index_label=index)
    model_test_sbt = model_test_sbt.sort_index()

    combs = list(itertools.product(comblist1, comblist2))
    comblist = comblist1 + comblist2

    if proxy == "no":
        weight_dict = None
    #Run all offline versions of the FALCC and FALCES algorithms
    falccsbt = algorithm.FALCC(metricer, index, sens_attrs, label, favored, comblist,
        X_test, combs, d_sbt2, proxy, weight_dict, ignore_sens, pre_processed)
    model_dict_sbt, kmeans = falccsbt.cluster_offline(X_test_cluster, kmeans, test_df_sbt2,
        metric, weight, link, other_folder=i, sbt=True)

    df = falccsbt.predict(model_dict_sbt, X_pred, y_pred, True, kmeans)
    df.to_csv(link + str(i) + "/FALCC-SBT_prediction_output.csv", index=False)
    with open(link + str(i) + "/q_score.txt", 'w') as outfile:
        outfile.write(str(q_val[i]) + "\n")
        outfile.write(str(q_avg_list[i]))

    if testall:
        falcessbt = algorithm.FALCES(metricer, index, sens_attrs, label, favored, comblist,
            X_test, combs, d_sbt2, pre_processed)
        global_model_comb_sbt = falcessbt.efficient_offline(model_test_sbt, metric, weight, threshold,
            comb_amount)

        falcesnewsbt = algorithm.FALCESNew(metricer, index, sens_attrs, label, favored, comblist,
            X_test, combs, d_sbt2, pre_processed)

        df = falcesnewsbt.predict(test_df_sbt2, X_pred, y_pred, "performance-efficient", metric, weight, True,
            knn_size, global_model_comb_sbt)
        df.to_csv(link + str(i) + "/FALCES-SBT-PFA-NEW_prediction_output.csv", index=False)

        df = falcessbt.predict(test_df_sbt2, X_pred, y_pred, "performance-efficient", metric, weight, True,
            knn_size, global_model_comb_sbt)
        df.to_csv(link + str(i) + "/FALCES-SBT-PFA_prediction_output.csv", index=False)

        df = falcesnewsbt.predict(test_df_sbt2, X_pred, y_pred, "naive", metric, weight, True, knn_size)
        df.to_csv(link + str(i) + "/FALCES-SBT-NEW_prediction_output.csv", index=False)

        df = falcessbt.predict(test_df_sbt2, X_pred, y_pred, "naive", metric, weight, True, knn_size)
        df.to_csv(link + str(i) + "/FALCES-SBT_prediction_output.csv", index=False)

        decouple_alg = algorithm.Decouple(metricer, index, pred_id_list, sens_attrs, label, favored, combs)
        df = decouple_alg.decouple(model_test_sbt, X_pred, y_pred, metric, weight, sbt=True)
        df.to_csv(link + str(i) + "/Decouple-SBT_prediction_output.csv", index=False)

        if fairboost and i == 0:
            fb = algorithm.FairBoost(index, pred_id_list, sens_attrs, favored, label, DecisionTreeClassifier())
            df_result = fb.fit_predict(X_train, y_train, X_pred, y_pred, r=0.1)
            df_result.to_csv(link + "FairBoost_prediction_output.csv", index=False)
