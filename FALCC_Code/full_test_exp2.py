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
#"adaboost" (for our AdaptedAdaBoost strategy), "single_classifiers", "no" if own models are used
training = "exp2"
#(7) if a proxy strategy is used ("no", "reweigh", "remove")
proxy = "reweigh"
#(8) list of allowed "proxy" attributes (required, if reweigh or remove strategy is chosen)
allowed_list = [[""], [""], [""]]
#(9) the minimum and maximum clustersize (if set to -1, we use our automatic approach)
ccr = [-1,-1]
#(10) which cluster parameter estimation strategy to choose (needed depending on ccr)
#"LOGmeans", "elbow"
ca = "LOGmeans"
#(11) randomstate
randomstate = 100
#(12) run only FALCC or also the other algorithms
testall = False
#(13) if the amount of sensitive groups is binary, the FairBoost algorithm can be run
#####################################################################################

for loop, input_file in enumerate(input_file_list):
    sensitive = sens_attrs_list[loop]
    label = label_list[loop]
    favored = favored_list[loop]
    allowed = allowed_list[loop]
    fairboost = False
    
    link = "Results/Exp2_" + str(proxy) + "_" + str(input_file) + "/"

    try:
        os.makedirs(link)
    except FileExistsError:
        pass

    #Run offline and online phases
    subprocess.check_call(['python', '-Wignore', 'main_offline_exp2.py', '-i', str(input_file),
        '-o', str(link), '--sensitive', str(sensitive), '--label', str(label),
        '--favored', str(favored), '--ccr', str(ccr), '--metric', str(metric),
        '--training', str(training), '--fairboost', str(fairboost), '--randomstate',
        str(randomstate), "--proxy", str(proxy), "--allowed", str(allowed),
        '--testall', str(testall), '--cluster_algorithm', str(ca)])

    models = ["FALCC" + str(i) for i in range(288)]

    for i in range(288):
        subprocess.check_call(['python', '-Wignore', 'evaluation_exp2.py', '--ds', str(input_file),
            '--folder', str(link), '--sensitive', str(sensitive), '--label', str(label),
            '--favored', str(favored), '--proxy', str(proxy), '--models', str(models)])
