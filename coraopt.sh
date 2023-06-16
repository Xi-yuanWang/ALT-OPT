d=Cora; h=64;
m=AGD; prop=None; 
name=AGD5_Cora;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD10_Cora;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD20_Cora;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD60_Cora;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 60 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=AGD0.3_Cora;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &


d=PubMed; h=64;
m=AGD; prop=None; 
name=LAGD5_PubMed;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD10_PubMed;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD20_PubMed;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD60_PubMed;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD0.3_PubMed;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &

d=CiteSeer; h=64;
m=AGD; prop=None; 
name=LAGD5_CiteSeer;
CUDA_VISIBLE_DEVICES=0 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD10_CiteSeer;
CUDA_VISIBLE_DEVICES=1 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD20_CiteSeer;
CUDA_VISIBLE_DEVICES=3 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 20 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD60_CiteSeer;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --fix_num 10 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &
name=LAGD0.3_CiteSeer;
CUDA_VISIBLE_DEVICES=2 nohup python3 -u main_optuna.py --dataset ${d} --model ${m} --prop ${prop} --runs 3 --random_splits 10 --proportion 0.3 --log_steps 100 --weight_decay 0.0005 --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 --debug 1 > ./AGD/${name}_log 2>&1 &


d=physics; gpu=0; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_physics; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 > ./AGD/${name}_log 2>&1 &


d=computers; gpu=1; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_computers; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 > ./AGD/${name}_log 2>&1 &

d=photo; gpu=2; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_photo; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 > ./AGD/${name}_log 2>&1 &

d=cs; gpu=3; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_cs; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1  > ./AGD/${name}_log 2>&1 &