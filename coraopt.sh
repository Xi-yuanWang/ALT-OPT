d=Cora; h=64;
m=ALTOPT; prop=None; 
name=LALTOPT5_Cora;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &
name=LALTOPT10_Cora;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &
name=LALTOPT20_Cora;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &
name=LALTOPT0.3_Cora;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &


d=PubMed; h=64;
m=ALTOPT; prop=None; 
name=LALTOPT5_PubMed;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &
name=LALTOPT10_PubMed;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &
name=LALTOPT20_PubMed;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &
name=LALTOPT0.3_PubMed;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./reform/${name}_log 2>&1 &