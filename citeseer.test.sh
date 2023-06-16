CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 1 --epoch 100 --proportion 0.3  --Fwd 3.69e-06  --K 3  --K0 4  --alpha 6.00e-01  --bn False  --dropout 8.50e-01  --gnnepoch 0  --hidden_channels 64  --lambda1 7.15e+00  --lambda2 9.65e+00  --loop 1  --loss CE  --lr 1.29e-03  --num_layers 1  --tailln True  --temperature 1.12e-01  --weight_decay 2.39e-03  --weightedloss True  > ./AGD/CiteSeer_0_0.3 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 1 --epoch 100 --proportion 0  --Fwd 1.40e-05  --K 2  --K0 8  --alpha 6.00e-01  --bn False  --dropout 6.50e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 4.35e+00  --lambda2 4.85e+00  --loop 1  --loss CE  --lr 1.07e-02  --num_layers 2  --tailln False  --temperature 1.13e-02  --weight_decay 4.26e-05  --weightedloss True  > ./AGD/CiteSeer_10_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 2.15e-04  --K 3  --K0 6  --alpha 7.00e-01  --bn True  --dropout 5.00e-02  --gnnepoch 60  --hidden_channels 64  --lambda1 1.90e+00  --lambda2 6.20e+00  --loop 1  --loss MSE  --lr 1.47e-02  --num_layers 2  --tailln False  --temperature 2.45e-02  --weight_decay 2.95e-02  --weightedloss False  > ./AGD/CiteSeer_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 5 --log_steps 1 --epoch 100 --proportion 0  --Fwd 6.78e-04  --K 3  --K0 14  --alpha 8.50e-01  --bn True  --dropout 5.00e-02  --gnnepoch 50  --hidden_channels 64  --lambda1 6.20e+00  --lambda2 5.10e+00  --loop 1  --loss MSE  --lr 2.63e-02  --num_layers 1  --tailln False  --temperature 1.20e-01  --weight_decay 2.19e-04  --weightedloss False  > ./AGD/CiteSeer_5_0 2>&1 & 
wait
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 1 --epoch 100 --proportion 0.3  --Fwd 2.79e-05  --K 3  --K0 20  --alpha 9.50e-01  --bn False  --dropout 3.50e-01  --gnnepoch 40  --hidden_channels 64  --lambda1 5.60e+00  --lambda2 6.60e+00  --loop 1  --loss MSE  --lr 5.27e-03  --num_layers 2  --tailln False  --temperature 3.53e-01  --weight_decay 3.52e-02  --weightedloss False  > ./AGD/Cora_0_0.3 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 1 --epoch 100 --proportion 0  --Fwd 1.35e-04  --K 1  --K0 19  --alpha 8.50e-01  --bn False  --dropout 4.50e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 8.90e+00  --lambda2 6.50e+00  --loop 1  --loss CE  --lr 1.88e-03  --num_layers 2  --tailln True  --temperature 8.34e-02  --weight_decay 2.25e-04  --weightedloss False  > ./AGD/Cora_10_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 8.29e-04  --K 2  --K0 20  --alpha 9.00e-01  --bn False  --dropout 4.00e-01  --gnnepoch 50  --hidden_channels 64  --lambda1 9.30e+00  --lambda2 6.80e+00  --loop 1  --loss CE  --lr 2.71e-02  --num_layers 2  --tailln True  --temperature 1.74e-01  --weight_decay 2.22e-05  --weightedloss False  > ./AGD/Cora_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 5 --log_steps 1 --epoch 100 --proportion 0  --Fwd 3.87e-04  --K 2  --K0 18  --alpha 8.50e-01  --bn False  --dropout 5.00e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 5.60e+00  --lambda2 2.15e+00  --loop 1  --loss CE  --lr 2.53e-02  --num_layers 2  --tailln False  --temperature 6.52e-02  --weight_decay 4.52e-02  --weightedloss True  > ./AGD/Cora_5_0 2>&1 & 
wait
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 1 --epoch 100 --proportion 0.3  --Fwd 4.83e-03  --K 3  --K0 18  --alpha 8.50e-01  --bn True  --dropout 3.50e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 5.40e+00  --lambda2 7.50e+00  --loop 1  --loss CE  --lr 1.79e-03  --num_layers 2  --tailln False  --temperature 7.37e-02  --weight_decay 1.34e-05  --weightedloss False  > ./AGD/PubMed_0_0.3 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 1 --epoch 100 --proportion 0  --Fwd 5.46e-03  --K 3  --K0 14  --alpha 9.50e-01  --bn False  --dropout 7.00e-01  --gnnepoch 0  --hidden_channels 64  --lambda1 5.80e+00  --lambda2 2.60e+00  --loop 1  --loss CE  --lr 2.79e-02  --num_layers 1  --tailln False  --temperature 9.91e-01  --weight_decay 1.66e-04  --weightedloss True  > ./AGD/PubMed_10_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 2.11e-02  --K 2  --K0 17  --alpha 9.50e-01  --bn True  --dropout 6.00e-01  --gnnepoch 50  --hidden_channels 64  --lambda1 4.00e+00  --lambda2 1.00e+01  --loop 1  --loss MSE  --lr 2.01e-03  --num_layers 2  --tailln True  --temperature 4.61e-02  --weight_decay 1.43e-04  --weightedloss True  > ./AGD/PubMed_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 5 --log_steps 1 --epoch 100 --proportion 0  --Fwd 4.89e-06  --K 2  --K0 13  --alpha 9.50e-01  --bn True  --dropout 4.00e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 4.95e+00  --lambda2 3.05e+00  --loop 1  --loss MSE  --lr 4.64e-03  --num_layers 1  --tailln True  --temperature 2.84e-01  --weight_decay 4.11e-02  --weightedloss False  > ./AGD/PubMed_5_0 2>&1 & 
wait
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset computers --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 7.33e-03  --K 3  --K0 26  --alpha 7.00e-01  --bn False  --dropout 0.00e+00  --gnnepoch 50  --hidden_channels 64  --lambda1 1.25e+00  --lambda2 4.95e+00  --loop 1  --loss CE  --lr 1.59e-02  --num_layers 2  --tailln True  --temperature 1.52e-02  --weight_decay 4.06e-06  --weightedloss True  > ./AGD/computers_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset cs --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 2.69e-03  --K 1  --K0 18  --alpha 7.50e-01  --bn True  --dropout 1.50e-01  --gnnepoch 20  --hidden_channels 64  --lambda1 6.45e+00  --lambda2 6.80e+00  --loop 1  --loss MSE  --lr 1.85e-03  --num_layers 2  --tailln True  --temperature 4.09e+00  --weight_decay 8.09e-03  --weightedloss True  > ./AGD/cs_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset ogbn-arxiv --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 1 --epoch 100 --proportion 0  --Fwd 1.00e-02  --K 2  --K0 18  --alpha 5.50e-01  --bn True  --dropout 0.00e+00  --gnnepoch 60  --hidden_channels 128  --lambda1 8.70e+00  --lambda2 5.50e-01  --loop 1  --loss MSE  --lr 1.23e-03  --num_layers 1  --tailln False  --temperature 7.08e-01  --weight_decay 3.60e-06  --weightedloss True  > ./AGD/ogbn-arxiv_0_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset photo --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 1.17e-03  --K 3  --K0 14  --alpha 8.50e-01  --bn True  --dropout 1.50e-01  --gnnepoch 50  --hidden_channels 64  --lambda1 1.05e+00  --lambda2 2.95e+00  --loop 1  --loss MSE  --lr 1.09e-03  --num_layers 2  --tailln True  --temperature 2.95e-02  --weight_decay 1.35e-05  --weightedloss True  > ./AGD/photo_20_0 2>&1 & 
wait
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset physics --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 1 --epoch 100 --proportion 0  --Fwd 8.32e-03  --K 2  --K0 21  --alpha 8.50e-01  --bn False  --dropout 4.00e-01  --gnnepoch 20  --hidden_channels 64  --lambda1 6.30e+00  --lambda2 5.50e-01  --loop 1  --loss CE  --lr 2.17e-02  --num_layers 1  --tailln True  --temperature 3.59e-02  --weight_decay 2.43e-03  --weightedloss True  > ./AGD/physics_20_0 2>&1 & 