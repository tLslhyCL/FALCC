# FALCC

This repository contains the codes needed to reproduce the experiments of our submitted FAccT 2023 paper:
"FALCC: Efficiently performing fair and accurate local classifications through local region clustering"

Run "full_test.py" and use that file to adapt the parameters (e.g. change datasets etc.).
The description is within that file. 
For the second experiment, run "full_test_exp2.py". For this experiment it will run the SBT versions only.
full_test.py is configurable to be used with other numerical datasets, while full_test_exp2.py might run into problems & is specialized for the experiments shown in the paper.

The code has been tested using a Windows PC. Other systems, like Mac might create shelve files in other versions, requiring adaptations.

The code runs the algorithms FALCC & FALCC-SBT, Decouple & Decouple-SBT [1], FALCES and its variants [2], and FairBoost [3]

- [1] Dwork, C., Immorlica, N., Kalai, A., Leiserson, M. "Decoupled Classifiers for Group-Fair
    and Efficient Machine Learning". 2018.

- [2] Lässig, N., Oppold, S., Herschel, M. "Metrics and Algorithms for Locally Fair and Accurate
    Classifications using Ensembles". 2022.
    
- [3] Bhaskaruni, D., Hu, H., Lan, C. "Improving Prediction Fairness via Model Ensemble". 2019.

For the algorithms of [1]-[3] we tried to implement the algorithms based on the information provided by the respective papers.


The datasets can be found within 'FALCC_Code/Datasets/'.
The results will be stored within 'FALCC_Code/Results/'.
