copied from https://anonymous.4open.science/r/ALT-OPT

# Environment

See requirements.txt

# Data

Code can download data. You can also put data in ../data

# Core Code 

main_optuna.py: script for hyperparameter tuning and test.

train_eval.py: script for training model. Specifically, train_altopt function optimize MLP in our model.

prop.py: Optimizing F. agd_forward use augmented gradient descent. exact_forward use exact solution of F.


# Reproducing Baselines
To reproduce AGD with $K\le 3$.
```
mkdir AGDtest
mkdir AGDtest/main
mkdir AGDtest/agdk10
mkdir AGDtest/exact
chmod +777 testall.sh
./testall.sh
```

To reproduce AGD with K=10,
```
chmod +777 testall_K=10.sh
./testall_K=10.sh
```

To reproduce EXACT
```
chmod +777 testall_exact.sh
./testall_exact.sh
```
