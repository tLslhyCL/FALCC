"""
Call methods for the online phase of the FALCC and FALCES[1] algorithm and its variants.
[1] Lässig, N., Oppold, S., Herschel, M. "Metrics and Algorithms for Locally Fair and Accurate
    Classifications using Ensembles". 2022.
"""
import warnings
import argparse
import shelve
import time
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, help="Directory containing the shelve.out file and\
    all important data")
#TL argument only needed for evaluation, since both versions are tested
parser.add_argument("--sbt", type=str, default=True, help="If the data was split before training\
    each classifier. Default: True")
args = parser.parse_args()

link = args.folder

runtime_analysis = pd.DataFrame(columns=["algorithm", "time"])
analysis_counter = 0

#Load the shelve data
filename = link + "shelve.out"
zmy_shelf = shelve.open(filename)
for key in zmy_shelf:
    globals()[key] = zmy_shelf[key]
zmy_shelf.close()

#Version 1: Algorithm is the cluster approach
start = time.time()
df = falcc.predict(model_dict, X_pred, y_pred, False, kmeans)
df.to_csv(link + "FALCC_prediction_output.csv", index=False)
runtime_analysis.at[analysis_counter, "algorithm"] = "FALCC"
runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
analysis_counter = analysis_counter + 1

if training != "no":
    start = time.time()
    df = falccsbt.predict(model_dict_sbt, X_pred, y_pred, True, kmeans)
    df.to_csv(link + "FALCC-SBT_prediction_output.csv", index=False)
    runtime_analysis.at[analysis_counter, "algorithm"] = "FALCC-SBT"
    runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
    analysis_counter = analysis_counter + 1

#Version 2: Algorithm is FALCES(-SBT)(-PFA)(-NEW).
if testall:
    start = time.time()
    df = falcesnew.predict(test_df, X_pred, y_pred, "performance-efficient", metric, weight, False,
        knn_size, global_model_comb)
    df.to_csv(link + "FALCES-PFA-NEW_prediction_output.csv", index=False)
    runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-PFA-NEW"
    runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
    analysis_counter = analysis_counter + 1

    start = time.time()
    df = falces.predict(test_df, X_pred, y_pred, "performance-efficient", metric, weight, False,
    knn_size, global_model_comb)
    df.to_csv(link + "FALCES-PFA_prediction_output.csv", index=False)
    runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-PFA"
    runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
    analysis_counter = analysis_counter + 1

    if training != "no":
        start = time.time()
        df = falcesnewsbt.predict(test_df_sbt, X_pred, y_pred, "performance-efficient", metric, weight, True,
            knn_size, global_model_comb_sbt)
        df.to_csv(link + "FALCES-SBT-PFA-NEW_prediction_output.csv", index=False)
        runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-SBT-PFA-NEW"
        runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
        analysis_counter = analysis_counter + 1

        start = time.time()
        df = falcessbt.predict(test_df_sbt, X_pred, y_pred, "performance-efficient", metric, weight, True,
            knn_size, global_model_comb_sbt)
        df.to_csv(link + "FALCES-SBT-PFA_prediction_output.csv", index=False)
        runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-SBT-PFA"
        runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
        analysis_counter = analysis_counter + 1


    start = time.time()
    df = falcesnew.predict(test_df, X_pred, y_pred, "naive", metric, weight, False, knn_size)
    df.to_csv(link + "FALCES-NEW_prediction_output.csv", index=False)
    runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-NEW"
    runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
    analysis_counter = analysis_counter + 1

    start = time.time()
    df = falces.predict(test_df, X_pred, y_pred, "naive", metric, weight, False, knn_size)
    df.to_csv(link + "FALCES_prediction_output.csv", index=False)
    runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES"
    runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
    analysis_counter = analysis_counter + 1

    if training != "no":
        start = time.time()
        df = falcesnewsbt.predict(test_df_sbt, X_pred, y_pred, "naive", metric, weight, True, knn_size)
        df.to_csv(link + "FALCES-SBT-NEW_prediction_output.csv", index=False)
        runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-SBT-NEW"
        runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
        analysis_counter = analysis_counter + 1

        start = time.time()
        df = falcessbt.predict(test_df_sbt, X_pred, y_pred, "naive", metric, weight, True, knn_size)
        df.to_csv(link + "FALCES-SBT_prediction_output.csv", index=False)
        runtime_analysis.at[analysis_counter, "algorithm"] = "FALCES-SBT"
        runtime_analysis.at[analysis_counter, "time"] = str(time.time() - start)
        analysis_counter = analysis_counter + 1

runtime_analysis.to_csv(link + "runtime_evaluation_online.csv")
