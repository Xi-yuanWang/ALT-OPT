CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 100 --epoch 100 --proportion 0.3  --Fwd 1.82e-05  --K 10  --K0 8  --alpha 7.00e-01  --bn True  --dropout 7.50e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 1.16e+01  --lambda2 8.80e+00  --loop 1  --loss MSE  --lr 1.00e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 3.43e-01  --useGCN False  --weight_decay 3.95e-06  --weightedloss False  > ./AGDtest/agdk10/CiteSeer_0_0.3 2>&1 & 
CUDA_VISIBLE_DEVICES=2 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.40e-05  --K 10  --K0 8  --alpha 6.00e-01  --bn False  --dropout 6.50e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 4.35e+00  --lambda2 4.85e+00  --loop 1  --loss CE  --lr 1.07e-02  --num_layers 2  --tailln False  --temperature 1.13e-02  --weight_decay 4.26e-05  --weightedloss True  > ./AGDtest/agdk10/CiteSeer_10_0 2>&1 & 
CUDA_VISIBLE_DEVICES=3 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.40e-05  --K 10  --K0 8  --alpha 6.00e-01  --bn False  --dropout 6.50e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 4.35e+00  --lambda2 4.85e+00  --loop 1  --loss CE  --lr 1.07e-02  --num_layers 2  --tailln False  --temperature 1.13e-02  --weight_decay 4.26e-05  --weightedloss True > ./AGDtest/agdk10/CiteSeer_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.40e-05  --K 10  --K0 8  --alpha 6.00e-01  --bn False  --dropout 6.50e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 4.35e+00  --lambda2 4.85e+00  --loop 1  --loss CE  --lr 1.07e-02  --num_layers 2  --tailln True  --temperature 1.13e-02  --weight_decay 4.26e-05  --weightedloss True   > ./AGDtest/agdk10/CiteSeer_5_0 2>&1 &
wait 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --dataset CiteSeer --model AGD --prop None --runs 3 --random_splits 10 --fix_num 60 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.50e-03  --K 10  --K0 16  --alpha 6.00e-01  --bn True  --dropout 8.50e-01  --gnnepoch 50  --hidden_channels 64  --lambda1 4.25e+00  --lambda2 6.70e+00  --loop 1  --loss MSE  --lr 2.36e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 1.87e-02  --useGCN False  --weight_decay 4.57e-03  --weightedloss False  > ./AGDtest/agdk10/CiteSeer_60_0 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python3 -u main_optuna.py --test 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 100 --epoch 100 --proportion 0.3  --Fwd 2.92e-03  --K 10  --K0 12  --alpha 8.50e-01  --bn True  --dropout 7.00e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 7.50e-01  --lambda2 6.75e+00  --loop 1  --loss MSE  --lr 1.26e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 2.91e-02  --useGCN False  --weight_decay 6.05e-05  --weightedloss False  > ./AGDtest/agdk10/Cora_0_0.3 2>&1 & 
CUDA_VISIBLE_DEVICES=2 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 0 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --epoch 100 --proportion 0  --Fwd 0.0021  --K 10  --K0 21  --alpha 0.9  --bn False  --dropout 0.3  --gnnepoch 60  --hidden_channels 64  --lambda1 3.4  --lambda2 9.2  --loop 1  --loss CE  --lr 6.8e-03  --num_layers 2  --tailln True  --temperature 0.0267  --weight_decay 0  --weightedloss False   > ./AGDtest/agdk10/Cora_10_0 2>&1 & 
CUDA_VISIBLE_DEVICES=3 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 0 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 0.0287  --K 10  --K0 24  --alpha 0.8  --bn False  --dropout 0.35  --gnnepoch 30  --hidden_channels 64  --lambda1 4.5  --lambda2 3.05  --loop 1  --loss CE  --lr 2.78e-02  --num_layers 1  --tailln True  --temperature 1.39  --weight_decay 8.24e-06  --weightedloss False > ./AGDtest/agdk10/Cora_20_0 2>&1 & 
wait
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --epoch 100 --proportion 0  --Fwd 3.87e-04  --K 10  --K0 18  --alpha 8.50e-01  --bn False  --dropout 5.00e-01  --gnnepoch 30  --hidden_channels 64  --lambda1 5.60e+00  --lambda2 2.15e+00  --loop 1  --loss CE  --lr 2.53e-02  --num_layers 2  --tailln True  --temperature 6.52e-02  --weight_decay 4.52e-02  --weightedloss True > ./AGDtest/agdk10/Cora_5_0 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python3 -u main_optuna.py --test 1 --dataset Cora --model AGD --prop None --runs 3 --random_splits 10 --fix_num 60 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.84e-02  --K 10  --K0 17  --alpha 7.50e-01  --bn True  --dropout 8.00e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 1.87e+01  --lambda2 1.75e+01  --loop 1  --loss CE  --lr 2.55e-02  --num_layers 2  --softmaxF True  --tailln True  --temperature 6.41e-02  --useGCN False  --weight_decay 1.22e-06  --weightedloss False  > ./AGDtest/agdk10/Cora_60_0 2>&1 & 
CUDA_VISIBLE_DEVICES=2 python3 -u main_optuna.py --test 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 0 --log_steps 100 --epoch 100 --proportion 0.3  --Fwd 5.10e-02  --K 10  --K0 5  --alpha 6.00e-01  --bn True  --dropout 4.50e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 6.50e+00  --lambda2 1.88e+01  --loop 1  --loss MSE  --lr 6.96e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 1.39e-01  --useGCN False  --weight_decay 1.12e-03  --weightedloss False  > ./AGDtest/agdk10/PubMed_0_0.3 2>&1 & 
CUDA_VISIBLE_DEVICES=3 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --epoch 100 --proportion 0  --Fwd 5.46e-03  --K 10  --K0 14  --alpha 9.50e-01  --bn False  --dropout 7.00e-01  --gnnepoch 0  --hidden_channels 64  --lambda1 5.80e+00  --lambda2 2.60e+00  --loop 1  --loss CE  --lr 2.79e-02  --num_layers 1  --tailln False  --temperature 9.91e-01  --weight_decay 1.66e-04  --weightedloss True   > ./AGDtest/agdk10/PubMed_10_0 2>&1 &
wait 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 0 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 0.0105  --K 10  --K0 18  --alpha 0.9  --bn False  --dropout 0.65  --gnnepoch 80  --hidden_channels 64  --lambda1 8.85  --lambda2 18.4  --loop 1  --loss CE  --lr 2.72e-03  --num_layers 1  --tailln True  --temperature 4.60  --weight_decay  4.75e-6 --weightedloss False > ./AGDtest/agdk10/PubMed_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python3 -u main_optuna.py --test 1 --useGCN 0 --softmaxF 0 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --epoch 100 --proportion 0  --Fwd 0.01785 --K 10  --K0 18  --alpha 0.95  --bn True  --dropout 5.00e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 14.15  --lambda2 8.6  --loop 1  --loss CE  --lr 0.00275  --num_layers 1  --tailln True  --temperature 0.162  --weight_decay 1.17e-5  --weightedloss False   > ./AGDtest/agdk10/PubMed_5_0 2>&1 & 
CUDA_VISIBLE_DEVICES=2 python3 -u main_optuna.py --test 1 --dataset computers --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 4.45e-06  --K 10  --K0 2  --alpha 8.50e-01  --bn True  --dropout 5.50e-01  --gnnepoch 50  --hidden_channels 64  --lambda1 0.00e+00  --lambda2 4.70e+00  --loop 1  --loss CE  --lr 1.40e-03  --num_layers 1  --softmaxF False  --tailln True  --temperature 1.58e+00  --useGCN False  --weight_decay 1.39e-04  --weightedloss False  > ./AGDtest/agdk10/computers_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=3 python3 -u main_optuna.py --test 1 --dataset PubMed --model AGD --prop None --runs 3 --random_splits 10 --fix_num 60 --log_steps 100 --epoch 100  --Fwd 5.10e-02  --K 10  --K0 5  --alpha 6.00e-01  --bn True  --dropout 4.50e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 6.50e+00  --lambda2 1.88e+01  --loop 1  --loss MSE  --lr 6.96e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 1.39e-01  --useGCN False  --weight_decay 1.12e-03  --weightedloss False  > ./AGDtest/agdk10/PubMed_60_0 2>&1 & 
wait
CUDA_VISIBLE_DEVICES=3 python3 -u main_optuna.py --test 1 --dataset cs --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.95e-02  --K 10  --K0 23  --alpha 6.50e-01  --bn True  --dropout 2.00e-01  --gnnepoch 50  --hidden_channels 64  --lambda1 9.20e+00  --lambda2 3.30e+00  --loop 1  --loss MSE  --lr 1.26e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 3.27e-01  --useGCN False  --weight_decay 1.05e-02  --weightedloss False  > ./AGDtest/agdk10/cs_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=0 python3 -u main_optuna.py --test 1 --dataset photo --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 2.07e-02  --K 10  --K0 16  --alpha 8.00e-01  --bn True  --dropout 5.00e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 6.50e-01  --lambda2 3.60e+00  --loop 1  --loss MSE  --lr 3.53e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 1.90e+00  --useGCN False  --weight_decay 7.81e-03  --weightedloss False  > ./AGDtest/agdk10/photo_20_0 2>&1 & 
CUDA_VISIBLE_DEVICES=1 python3 -u main_optuna.py --test 1 --dataset physics --model AGD --prop None --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --epoch 100 --proportion 0  --Fwd 1.35e-06  --K 10  --K0 30  --alpha 8.50e-01  --bn True  --dropout 8.00e-01  --gnnepoch 60  --hidden_channels 64  --lambda1 1.07e+01  --lambda2 8.80e+00  --loop 1  --loss MSE  --lr 9.63e-03  --num_layers 2  --softmaxF True  --tailln True  --temperature 9.73e-02  --useGCN False  --weight_decay 3.49e-06  --weightedloss False  > ./AGDtest/agdk10/physics_20_0 2>&1 & 
wait


