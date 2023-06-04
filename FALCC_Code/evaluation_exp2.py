"""
This code evaluates the results of the experiments based on several metrics.
"""
import warnings
import argparse
import ast
import shelve
import copy
import re
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from algorithm.codes import DiversityMeasures

warnings.simplefilter(action='ignore')
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, help="Directory containing the shelve.out file and\
    all important data")
parser.add_argument("--ds", type=str, help="Name of the dataset")
parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
parser.add_argument("--favored", default=None, type=str, help="Tuple of favored group.\
    Otherwise some metrics can't be used. Default: None.")
parser.add_argument("--label", type=str, help="Column name of the target value.")
parser.add_argument("--proxy", type=str, help="If the proxy technique was used.")
parser.add_argument("--models", type=str, help="List of models that were trained.")
args = parser.parse_args()

link = args.folder
ds = args.ds
sens_attrs = ast.literal_eval(args.sensitive)
favored = ast.literal_eval(args.favored)
label = args.label
proxy = args.proxy
model_list = ast.literal_eval(args.models)


df = pd.read_csv(link + model_list[0] + "_prediction_output.csv", index_col="index")
test_df = pd.read_csv(link + "testdata_predictions.csv", index_col="index")

#Combine the output of all models
for i, model in enumerate(model_list):
    if i == 0:
        continue
    df2 = pd.read_csv(link + model + "_prediction_output.csv", index_col="index")
    df = pd.merge(df, df2[[model]], how="inner", left_index=True, right_index=True)
df.to_csv(link + "MODEL_COMPARISON.csv", index=False)

#Read the original dataset
if proxy == "no":
    orig_dataset = pd.read_csv("Datasets/" + ds + ".csv", index_col="index")
elif proxy == "reweigh":
    orig_dataset = pd.read_csv("Datasets/reweigh/" + ds + ".csv", index_col="index")
elif proxy == "remove":
    orig_dataset = pd.read_csv("Datasets/removed/" + ds + ".csv", index_col="index")
data_index_list = []
for i, rows in df.iterrows():
    data_index_list.append(i)
#Dataset includes all prediction data samples in original form
dataset = orig_dataset.loc[data_index_list]

#Create DataFrame with all columns for the evaluation result.
result_df = pd.DataFrame(columns=["dataset", "model", "accuracy", "demographic_parity",
    "equalized_odds", "equal_opportunity", "treatment_equality", "lrd_dp", "lrd_eod", "lrd_eop",
    "lrd_te", "impact", "f1_score", "group_testsize", "group_predsize"])

#Now evaluate each model according to the metrics implemented.
model_count = 0

filename = link + "cluster.out"
my_shelf = shelve.open(filename)
for key in my_shelf:
    kmeans = my_shelf["kmeans"]
my_shelf.close()

X = orig_dataset.loc[:, orig_dataset.columns != label]
y = orig_dataset[label]

for model in model_list:
    nr = int(re.findall(r'\d+', model)[0])
    if nr % 2 == 0:
        modelpkl = joblib.load(link + "AdaBoostClassic" + str(nr) + ".pkl")
    else:
        modelpkl = joblib.load(link + "RandomForestClassic" + str(nr) + ".pkl")
    #print(modelpkl.estimators_)

    DivMeasure = DiversityMeasures()
    result_df.at[model_count, "min_q"] = DivMeasure.QS_score(modelpkl, X, y, maxScore=True)
    result_df.at[model_count, "entropy"] = DivMeasure.entropy_score(modelpkl, X, y)
    result_df.at[model_count, "model"] = model
    result_df.at[model_count, "dataset"] = ds

    grouped_df = df.groupby(sens_attrs)
    counter_score = 0
    total_ppv = 0
    total_size = 0
    total_ppv_y0 = 0
    total_size_y0 = 0
    total_ppv_y1 = 0
    total_size_y1 = 0
    fav_ppv = 0
    fav_size = 0
    wrong_predicted = 0
    wrong_predicted_y0 = 0
    wrong_predicted_y1 = 0
    total_fp = 0
    total_fn = 0
    num_pos = 0
    num_neg = 0
    group_predsize = []
    #Get the favored group to test against and also the averages over the whole dataset
    for key, item in grouped_df:
        predsize = 0
        part_df = grouped_df.get_group(key)
        for i, row in part_df.iterrows():
            predsize += 1
            total_ppv = total_ppv + row[model]
            total_size = total_size + 1
            wrong_predicted = wrong_predicted + abs(row[model] - row[label])
            #counter_score = counter_score + abs(row[model] - cdf.loc[i, model])
            if row[label] == 0:
                total_ppv_y0 = total_ppv_y0 + row[model]
                total_size_y0 = total_size_y0 + 1
                wrong_predicted_y0 = wrong_predicted_y0 + abs(row[model] - row[label])
                if row[model] == 1:
                    total_fp += 1
                    num_pos += 1
                else:
                    num_neg += 1
            elif row[label] == 1:
                total_ppv_y1 = total_ppv_y1 + row[model]
                total_size_y1 = total_size_y1 + 1
                wrong_predicted_y1 = wrong_predicted_y1 + abs(row[model] - row[label])
                if row[model] == 0:
                    total_fn += 1
                    num_neg += 1
                else:
                    num_pos += 1
            if key == favored:
                fav_ppv = fav_ppv + row[model]
                fav_size = fav_size + 1
        group_predsize.append(predsize)


    group_testsize = []
    grouped2 = test_df.groupby(sens_attrs)
    for key, item in grouped2:
        testsize = 0
        part2 = grouped2.get_group(key)
        for i, row in part2.iterrows():
            testsize += 1
        group_testsize.append(testsize)

    inaccuracy = wrong_predicted/total_size * 100
    result_df.at[model_count, "inaccuracy"] = inaccuracy
    result_df.at[model_count, "accuracy"] = 100 - inaccuracy
    result_df.at[model_count, "group_testsize"] = group_testsize
    result_df.at[model_count, "group_predsize"] = group_predsize
    tp = num_pos - total_fp
    try:
        f1 = tp/(tp + 0.5*(total_fp+total_fn))
    except Exception:
        f1 = 0
    result_df.at[model_count, "f1_score"] = f1 * 100
    result_df.at[model_count, "fp"] = total_fp
    result_df.at[model_count, "fn"] = total_fn
    result_df.at[model_count, "num_pos"] = num_pos
    result_df.at[model_count, "num_neg"] = num_neg
    testsize = 0
    predsize = 0
    for i in range(len(group_testsize)):
        testsize = testsize + group_testsize[i]
        predsize = predsize + group_predsize[i]


    #Iterate again for formula
    count = 0
    dp = 0
    eq_odd = 0
    eq_opp = 0
    tr_eq = 0
    impact = 0
    fp = 0
    fn = 0
    for key, item in grouped_df:
        model_ppv = 0
        model_size = 0
        model_ppv_y0 = 0
        model_size_y0 = 0
        model_ppv_y1 = 0
        model_size_y1 = 0
        part_df = grouped_df.get_group(key)
        for i, row in part_df.iterrows():
            model_ppv = model_ppv + row[model]
            model_size = model_size + 1
            if row[label] == 0:
                model_ppv_y0 = model_ppv_y0 + row[model]
                model_size_y0 = model_size_y0 + 1
                if row[model] == 1:
                    fp += 1
            elif row[label] == 1:
                model_ppv_y1 = model_ppv_y1 + row[model]
                model_size_y1 = model_size_y1 + 1
                if row[model] == 0:
                    fn += 1

        dp = dp + abs(model_ppv/model_size - total_ppv/total_size)
        eq_odd = (eq_odd + 0.5*abs(model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0)
            + 0.5*abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1))
        eq_opp = eq_opp + abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1)
        if fp+fn == 0 and total_fp+total_fn == 0:
            pass
        elif fp+fn == 0:
            tr_eq = tr_eq + abs(0.5 - total_fp/(total_fp+total_fn))
        elif total_fp+total_fn == 0:
            tr_eq = tr_eq + abs(fp/(fp+fn) - 0.5)
        else:
            tr_eq = tr_eq + abs(fp/(fp+fn) - total_fp/(total_fp+total_fn))
        if fav_ppv == 0:
            impact = impact + 0
        else:
            if fav_ppv/fav_size >= model_ppv/model_size:
                impact = impact + abs(1 - (model_ppv/model_size)/(fav_ppv/fav_size))
            else:
                impact = impact + abs(1 - (fav_ppv/fav_size)/(model_ppv/model_size))


    #Remove the 'cluster' column
    cluster_cols = list(dataset.columns)
    cluster_cols.remove(label)
    for sens in sens_attrs:
        cluster_cols.remove(sens)
    for col in cluster_cols:
        #part_df[col] = pd.to_numeric(df[col])
        df_bigger = dataset[dataset[col] > 0.5]
        df_lower = dataset[dataset[col] <= 0.5]
        total_ppv_bigger = 0
        total_ppv_lower = 0
        count_bigger = 0
        count_lower = 0
        for i, row in df_bigger.iterrows():
            total_ppv_bigger += df.loc[i, model]
            count_bigger += 1
        for i, row in df_lower.iterrows():
            total_ppv_lower += df.loc[i, model]
            count_lower += 1

        grouped_dataset = dataset.groupby(sens_attrs)
        for key_inner, item_inner in grouped_df:
            group_ppv_bigger = 0
            group_ppv_lower = 0
            group_count_bigger = 0
            group_count_lower = 0
            part_dataset = grouped_dataset.get_group(key_inner)
            id_list = []
            for i, row in part_dataset.iterrows():
                id_list.append(i)

            df_local_bl = dataset.loc[id_list]

            df_bigger_local = part_dataset[part_dataset[col] > 0.5]
            df_lower_local = part_dataset[part_dataset[col] <= 0.5]
            for i, row in df_bigger_local.iterrows():
                group_ppv_bigger += df.loc[i, model]
                group_count_bigger += 1
            for i, row in df_lower_local.iterrows():
                group_ppv_lower += df.loc[i, model]
                group_count_lower += 1


    result_df.at[model_count, "demographic_parity"] = dp/(len(grouped_df)) * 100
    result_df.at[model_count, "equalized_odds"] = eq_odd/(len(grouped_df)) * 100
    result_df.at[model_count, "equal_opportunity"] = eq_opp/(len(grouped_df)) * 100
    result_df.at[model_count, "treatment_equality"] = tr_eq/(len(grouped_df)) * 100
    result_df.at[model_count, "impact"] = impact/(len(grouped_df) - 1) * 100

    model_count = model_count + 1


dataset2 = copy.deepcopy(dataset)
dataset2 = dataset2.loc[:, dataset2.columns != "index"]
dataset2 = dataset2.loc[:, dataset2.columns != label]
for sens in sens_attrs:
    dataset2 = dataset2.loc[:, dataset2.columns != sens]
if proxy == "remove":
    outfile = open(link + "removed_attributes.txt", 'r')
    Lines = outfile.readlines()   
    for line in Lines:
        line = line.replace("\n", "")
        dataset2 = dataset2.loc[:, dataset2.columns != line]


cluster_results = kmeans.predict(dataset2)
dataset2["cluster"] = cluster_results
clustered_df = dataset2.groupby("cluster")

groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
actual_num_of_groups = len(groups)
sensitive_groups = []
sens_cols = groups.columns
for i, row in groups.iterrows():
    sens_grp = []
    for col in sens_cols:
        sens_grp.append(row[col])
    sensitive_groups.append(tuple(sens_grp))


#models_lrd = [0 for i in range(len(model_list))]
models_lrd_dp = []
models_lrd_eod = []
models_lrd_eop = []
models_lrd_te = []
models_lrd_const = []
#Calculate local region discrimination (lrd) of the model.
for model in model_list:
    total_size = 0
    lrd_dp = 0
    lrd_eod = 0
    lrd_eop = 0
    lrd_te = 0
    lrd_const = 0
    clusters = 0
    for key, item in clustered_df:
        clusters += 1
        part_df = clustered_df.get_group(key)
        index_list = []
        for i, row in part_df.iterrows():
            index_list.append(i)
        df_local = df.loc[index_list]
        groups2 = df_local[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
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
                    if len(sens_attrs) == 1:
                        sens_grp = sens_grp[0]
                    grouped_df = dataset.groupby(sens_attrs)
                    for key_inner, item_inner in grouped_df:
                        if key_inner == sens_grp:
                            knn_df = grouped_df.get_group(key_inner)
                            for sens_attr in sens_attrs:
                                knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                            knn_df = knn_df.loc[:, knn_df.columns != "index"]
                            knn_df = knn_df.loc[:, knn_df.columns != label]
                            if proxy == "remove":
                                outfile = open(link + "removed_attributes.txt", 'r')
                                Lines = outfile.readlines()
                                for line in Lines:
                                    line = line.replace("\n", "")
                                    knn_df = knn_df.loc[:, knn_df.columns != line]
                            nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(knn_df.values)
                            indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                            real_indices = df.index[indices].tolist()
                            nearest_neighbors_df = df.loc[real_indices[0]]
                            df_local = df_local.append(nearest_neighbors_df)


        #Get average of whole cluster
        for i, row_outer in df_local.iterrows():
            count = 0
            total_ppv = 0
            count_y0 = 0
            total_ppv_y0 = 0
            count_y1 = 0
            total_ppv_y1 = 0
            total_fp = 0
            total_fn = 0
            for j, row in df_local.iterrows():
                total_ppv = total_ppv + row[model]
                count = count + 1
                if row[label] == 0:
                    total_ppv_y0 = total_ppv_y0 + row[model]
                    count_y0 = count_y0 + 1
                    if row[model] == 1:
                        total_fp += 1
                if row[label] == 1:
                    total_ppv_y1 = total_ppv_y1 + row[model]
                    count_y1 = count_y1 + 1
                    if row[model] == 0:
                        total_fn += 1
            total_pppv = total_ppv/count
            if count_y0 > 0:
                total_pppv_y0 = total_ppv_y0/count_y0
            else:
                total_pppv_y0 = 0
            if count_y1 > 0:
                total_pppv_y1 = total_ppv_y1/count_y1
            else:
                total_pppv_y1 = 0


        df_local2 = df_local.groupby(sens_attrs)
        #Get average PPPV of one kNN group:
        lrd_local_dp = 0
        lrd_local_eod = 0
        lrd_local_eop = 0
        lrd_local_te = 0
        lrd_local_const = 0
        cluster_count = 0
        cluster_count_y0 = 0
        cluster_count_y1 = 0
        for key_inner, item_inner in df_local2:
            group_ppv = 0
            group_count = 0
            group_ppv_y0 = 0
            group_count_y0 = 0
            group_ppv_y1 = 0
            group_count_y1 = 0
            fp = 0
            fn = 0
            part_df_local = df_local2.get_group(key_inner)
            for j, row in part_df_local.iterrows():
                group_ppv = group_ppv + row[model]
                group_count = group_count + 1
                cluster_count = cluster_count + 1
                if row[label] == 0:
                    group_ppv_y0 = group_ppv_y0 + row[model]
                    group_count_y0 = group_count_y0 + 1
                    cluster_count_y0 = cluster_count_y0 + 1
                    if row[model] == 1:
                        fp += 1
                elif row[label] == 1:
                    group_ppv_y1 = group_ppv_y1 + row[model]
                    group_count_y1 = group_count_y1 + 1
                    cluster_count_y1 = cluster_count_y1 + 1
                    if row[model] == 0:
                        fn += 1
            group_pppv = group_ppv/group_count

            if group_count_y1 > 0:
                group_pppv_y1 = group_ppv_y1/group_count_y1
                lrd_local_eop = lrd_local_eop + abs(group_pppv_y1 - total_pppv_y1)
                if group_count_y0 > 0:
                    group_pppv_y0 = group_ppv_y0/group_count_y0
                    lrd_local_eod = (lrd_local_eod + 0.5*abs(group_pppv_y0 - total_pppv_y0)
                        + 0.5*abs(group_pppv_y1 - total_pppv_y1))
                else:
                    lrd_local_eod = lrd_local_eod + abs(group_pppv_y1 - total_pppv_y1)
            else:
                group_pppv_y0 = group_ppv_y0/group_count_y0
                lrd_local_eod = lrd_local_eod + abs(group_pppv_y0 - total_pppv_y0)

            lrd_local_dp = lrd_local_dp + abs(total_pppv - group_pppv)
            if fp+fn == 0 and total_fp+total_fn == 0:
                pass
            elif fp+fn == 0:
                lrd_local_te = lrd_local_te + abs(0.5 - total_fp/(total_fp+total_fn))
            elif total_fp+total_fn == 0:
                lrd_local_te = lrd_local_te + abs(fp/(fp+fn) - 0.5)
            else:
                lrd_local_te = lrd_local_te + abs(fp/(fp+fn) - total_fp/(total_fp+total_fn))

        lrd_dp = lrd_dp + lrd_local_dp/len(grouped_df) * cluster_count
        lrd_eod = lrd_eod + lrd_local_eod/len(grouped_df) * cluster_count
        lrd_eop = lrd_eop + lrd_local_eop/len(grouped_df) * cluster_count
        lrd_te = lrd_te + lrd_local_te/len(grouped_df) * cluster_count
        lrd_const = lrd_const + min(1-total_pppv, total_pppv) * cluster_count
        total_size += cluster_count

    models_lrd_dp.append(lrd_dp/total_size)
    models_lrd_eod.append(lrd_eod/total_size)
    models_lrd_eop.append(lrd_eop/total_size)
    models_lrd_te.append(lrd_te/total_size)
    models_lrd_const.append(lrd_const/total_size)


models_consistency = [0 for model in model_list]
#CONSISTENCY TEST, COMPARE PREDICTION TO PREDICTIONS OF NEIGHBORS
consistency = 0
dataset = dataset.loc[:, dataset.columns != "index"]
dataset = dataset.loc[:, dataset.columns != label]
dataset3 = copy.deepcopy(dataset)
for sens in sens_attrs:
    dataset3 = dataset3.loc[:, dataset3.columns != sens]
for i, row_outer in df.iterrows():
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(dataset.values)
    indices = nbrs.kneighbors(dataset.loc[i].values.reshape(1, -1),\
        return_distance=False)
    real_indices = df.index[indices].tolist()
    df_local = df.loc[real_indices[0]]
    model_count = 0
    for model in model_list:
        knn_ppv = 0
        knn_count = 0
        for j, row in df_local.iterrows():
            knn_ppv = knn_ppv + row[model]
            knn_count = knn_count + 1
        knn_pppv = knn_ppv/knn_count
        models_consistency[model_count] = models_consistency[model_count] + abs(df.loc[i][model] - knn_pppv)
        model_count = model_count + 1
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(dataset3.values)
    indices = nbrs.kneighbors(dataset3.loc[i].values.reshape(1, -1),\
        return_distance=False)
    real_indices = df.index[indices].tolist()
    df_local = df.loc[real_indices[0]]
    model_count = 0
    for model in model_list:
        knn_ppv = 0
        knn_count = 0
        for j, row in df_local.iterrows():
            knn_ppv = knn_ppv + row[model]
            knn_count = knn_count + 1
        knn_pppv = knn_ppv/knn_count
        model_count = model_count + 1


model_count = 0
for model in model_list:
    result_df.at[model_count, "consistency"] = models_consistency[model_count]/len(df) * 100
    result_df.at[model_count, "lrd_dp"] = models_lrd_dp[model_count] * 100
    result_df.at[model_count, "lrd_eod"] = models_lrd_eod[model_count] * 100
    result_df.at[model_count, "lrd_eop"] = models_lrd_eop[model_count] * 100
    result_df.at[model_count, "lrd_te"] = models_lrd_te[model_count] * 100
    result_df.at[model_count, "lrd_const"] = models_lrd_const[model_count] * 100
    model_count = model_count + 1

result_df.to_csv(link + "EVALUATION.csv")
