d=CiteSeer; h=64;
m=AGD; prop=None; 
name=AGD5_CiteSeer;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD10_CiteSeer;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD20_CiteSeer;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10  --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD0.3_CiteSeer;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3  --random_splits 10 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &


d=CiteSeer; h=64;
m=EXACT; prop=None; 
name=EXACT5_CiteSeer;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=EXACT10_CiteSeer;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=EXACT20_CiteSeer;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10  --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=EXACT0.3_CiteSeer;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3  --random_splits 10 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &


#CUDA_VISIBLE_DEVICES=2 python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 1  --K 3 --epoch 100 --alpha 0.4 --dropout 0.9 --gnnepoch 200 --lambda1 0.4 --lambda2 6.25 --loop 1 --loss CE --lr 0.000472 --softmaxF 0 --useGCN 0 --weight_decay 2e-5 --weightedloss 0  --debug 1 --Fwd 0.03 --test

#d=CiteSeer; h=64;
#m=ALTOPT; prop=None; 
#name=ALTOPT5_CiteSeer;
#CUDA_VISIBLE_DEVICES=2 python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 1  --K 3 --epoch 100 --alpha 0.4 --dropout 0.9 --gnnepoch 120 --lambda1 0.4 --lambda2 6.25 --loop 2 --loss CE --lr 0.000472 --softmaxF 0 --useGCN 0 --weight_decay 2e-5 --weightedloss 0  --debug 1 --Fwd 0.03 --test

#d=CiteSeer; h=64;
#m=ALTOPT; prop=None; 
#name=LALTOPT5_CiteSeer;
#CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
#name=LALTOPT10_CiteSeer;
#CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
#name=LALTOPT20_CiteSeer;
#CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
#name=LALTOPT0.3_CiteSeer;
#CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &