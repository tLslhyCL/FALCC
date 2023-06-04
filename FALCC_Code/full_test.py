"""
Python file/script for evaluation purposes.
"""
import subprocess
import os
import pandas as pd

###We use default argument values for the following parameters, however they can also be set.
###The "check_call" function of the offline phase has to be adapted if they are set.
#testsize = 0.5
#predsize = 0.3
#weight = 0.5
#knn = 15
#list of already trained models (if training is set to "no")
#trained_models_list = [] 

#####################################################################################
#Here we need to specify:
#(1) the dataset name(s)
input_file_list = ["communities", "implicit30", "social30"]
#(2) the name(s) of the sensitive attributes as a list
sens_attrs_list = [["race"], ["sensitive"], ["sensitive"]]
#(3) the value of the favored group
favored_list = [(1), (0), (0)]
#(4) the name of the label
label_list = ["crime", "label", "label"]
#(5) the metric for which the results should be optimized:
#"demographic_parity", "equalized_odds", "equal_opportunity", "treatment_equality"
metric = "demographic_parity"
#(6) which training strategy is used:
#"opt_adaboost" (for our proposed strategy), "opt_random_forest",
#"adaboost" (for our (old) AdaptedAdaboost strategy), "single_classifiers", "no" if own models are used
training = "opt_adaboost"
#(7) if a proxy strategy is used ("no", "reweigh", "remove")
proxy = "reweigh"
#(8) list of allowed "proxy" attributes (required, if reweigh or remove strategy is chosen)
allowed_list = [[""], [""], [""]]
#(9) the minimum and maximum clustersize (if set to -1, we use our automatic approach)
ccr = [-1,-1]
#(10) which cluster parameter estimation strategy to choose (needed depending on ccr)
#"LOGmeans", "elbow"
ca = "LOGmeans"
#(11) randomstate; if set to -1 it will randomly choose a randomstate
randomstate = -1
#(12) run only FALCC or also the other algorithms
testall = False
#(13) if the FairBoost and iFair algorithms should be run
fairboost_list = [True, True, True]
ifair_list = [True, True, True]
#####################################################################################

for loop, input_file in enumerate(input_file_list):
    sensitive = sens_attrs_list[loop]
    label = label_list[loop]
    favored = favored_list[loop]
    allowed = allowed_list[loop]
    fairboost = fairboost_list[loop]
    ifair = ifair_list = [loop]
    
    link = "Results/" + str(proxy) + "_" + str(input_file) + "/"

    try:
        os.makedirs(link)
    except FileExistsError:
        pass

    #Run offline and online phases
    subprocess.check_call(['python', '-Wignore', 'main_offline.py', '-i', str(input_file),
        '-o', str(link), '--sensitive', str(sensitive), '--label', str(label),
        '--favored', str(favored), '--ccr', str(ccr), '--metric', str(metric),
        '--training', str(training), '--fairboost', str(fairboost), '--ifair', str(ifair),
        '--randomstate', str(randomstate), "--proxy", str(proxy), "--allowed", str(allowed),
        '--testall', str(testall), '--cluster_algorithm', str(ca)])
    subprocess.check_call(['python', '-Wignore', 'main_online.py', '--folder', str(link)])

    #Run evaluation
    if testall:
        if training == "fair" or training == "no":
            models = ["Decouple", "FALCES", "FALCES-PFA", "FALCES-NEW", "FALCES-PFA-NEW", "FALCC"]
        else:
            models = ["Decouple", "Decouple-SBT", "FALCES", "FALCES-PFA", "FALCES-SBT", "FALCES-SBT-PFA",\
                "FALCES-NEW", "FALCES-PFA-NEW", "FALCES-SBT-NEW", "FALCES-SBT-PFA-NEW", "FALCC", "FALCC-SBT"]
        if len(sensitive) > 1:
            models.append("FaX")
            models.append("Fair-SMOTE")
            models.append("LFR")
            if fairboost:
                models.append("FairBoost")
            if ifair:
                models.append("iFair")
    else:
        if training == "fair" or training == "no":
            models = ["FALCC"]
        else:
            models = ["FALCC", "FALCC-SBT"]
    subprocess.check_call(['python', '-Wignore', 'evaluation.py', '--ds', str(input_file),
        '--folder', str(link), '--sensitive', str(sensitive), '--label', str(label),
        '--favored', str(favored), '--proxy', str(proxy), '--models', str(models)])
