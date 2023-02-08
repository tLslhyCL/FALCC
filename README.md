# FALCC

This repository contains the codes needed to reproduce the experiments of our submitted FAccT 2023 paper:
"FALCC: Efficiently performing locally fair and accurate classifications"


## STRUCTURE

The datasets can be found within 'FALCC_Code/Datasets/'.

The results will be stored within 'FALCC_Code/Results/'.


## IMPLEMENTATION

The code implements and runs the algorithms FALCC, Decouple [1], FALCES and its variants [2], and FairBoost [3].
For the algorithms of [1]-[3] we tried to implement the algorithms based on the information provided by the respective papers.

Further, we run LFR [4], iFair [5], FaX [6] & Fair-SMOTE [7]

- [1] Dwork C, Immorlica N, Kalai A, Leiserson M. "Decoupled Classifiers for Group-Fair
    and Efficient Machine Learning". 2018.

- [2] Lässig N, Oppold S, Herschel M. "Metrics and Algorithms for Locally Fair and Accurate Classifications using Ensembles". 2022.
    
- [3] Bhaskaruni D, Hu H, Lan C. "Improving Prediction Fairness via Model Ensemble". 2019.

- [4] Zemel R, Wu Y, Swersky K, Pitassi T, Dwork C. "Learning fair representations". 2013.

- [5] Lahoti P, Gummadi KP, Weikum G. "ifair: Learning individually fair data representations for algorithmic decision making.". 2019.

- [6] Grabowicz PA, Perello N, Mishra A. "Marrying fairness and explainability in supervised learning." 2022.

- [7] Chakraborty J, Majumder S, Menzies T. "Bias in machine learning software: Why? how? what to do?." 2021.


## HOWTO RUN

1. Required steps to run the files - Integration of other fair algorithms ([4]-[7]):
- LFR [4]: No steps required, as we use the AIF360 framework implementation
- iFair [5]:
- - Create a folder called "iFair_helper" in 'FALCC_Code/algorithm/codes'
- - Copy the content from the official implementation & put them inside this folder: https://github.com/plahoti-lgtm/iFair
- - Change the "fit" method to use the following parameters: 'def fit(self, X_train, sens_attrs, dataset=None)'
- - Adapt the first else-clause of the fit method to:
        X_train_new = copy.deepcopy(X_train)
        for sens in sens_attrs:
            selector = [x for x in range(X_train_new.shape[1]) if x != sens]
            X_train_new = X_train_new[:, selector]
        D_X_F = pairwise.euclidean_distances(X_train_new,
                                             X_train_new)
        l = X_train.shape[1] - len(sens_attrs)
- FaX [6]:
-- Create a folder called "FaX_AI" in 'FALCC_Code/algorithm/codes'
-- Copy the content from the official implementation & put them inside this folder: https://github.com/social-info-lab/FaX-AI
- Fair-SMOTE [7]:
-- Create a folder called "Fair_SMOTE" in 'FALCC_Code/algorithm/codes'
-- Copy "SMOTE.py" & "Generate_Samples.py" from the official implementation & put them inside this folder: https://github.com/joymallyac/Fair-SMOTE
-- Within "Generate_Samples.py":
--- Add a parameter called dict_cols to the "generate_samples" function.
--- Add the following else clause in the bottom of the "generate_samples" function: 'final_df = final_df.rename(columns=dict_cols, errors="raise")'
-- If non-binary sensitive groups are in the input, it will cause problems. In that case, the implementation within our framework has to be adapted,
    e.g. for the Adult Data Set with two sensitive attributes an example is given here: https://github.com/joymallyac/Fair-SMOTE/blob/master/Fair-SMOTE/Adult_Sex_Race.ipynb

2. Now the algorithms can be run. 
- For experiment 2: run "full_test_exp2.py". For this experiment it will run the non-SBT versions only.
- Otherwise run "full_test.py"
- Information on the possible input parameters and how to change datasets etc. are given within the respective files.
- For non-binary sensitive groups: Only run the adaboost or single_classifier training.

The code has been tested using a Windows PC. Other systems, like Mac, might create shelve files in other versions, requiring adaptations.
