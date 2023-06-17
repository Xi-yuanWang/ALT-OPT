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
mkdir AGDtest/exact_cg
mkdir AGDtest/exact
mkdir AGDtest/altopt
chmod +777 testall.sh
./testall.sh
```

To reproduce EXACT
```
chmod +777 testall_exact.sh
./testall_exact.sh
```

To reproduce EXACT-cg
```
chmod +777 testall_exact_cg.sh
./testall_exact_cg.sh
```


To reproduce ALTOPT
```
chmod +777 testall_altopt.sh
./testall_altopt.sh