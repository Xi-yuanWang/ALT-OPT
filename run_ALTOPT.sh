d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 1 --alpha 0.1 --loss CE --lambda1 0.1 --lambda2 3.0 2>&1 | tee -a ./${name}.test;


CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 1 --alpha 0.1 --loss CE --lambda1 0.1 --lambda2 3.0


d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_noGCN5_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 2>&1 | tee -a ./reform/${name}_log;

d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_noGCN10_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 10 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 --useGCN 0 2>&1 | tee -a ./reform/${name}_log;

d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_noGCN20_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 --useGCN 0 2>&1 | tee -a ./reform/${name}_log;


d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_noGCN60_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 60 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 2>&1 | tee -a ./reform/${name}_log;


d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_noGCN0.3_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--proportion 0.3 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 2>&1 | tee -a ./reform/${name}_log;

exit

d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT5_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 > ./reform/${name}_log;

d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT10_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 10 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 2>&1 | tee -a ./reform/${name}_log;

d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT20_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 2>&1 | tee -a ./reform/${name}_log;


d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT60_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 60 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./reform/${name}_log;


d=CiteSeer; gpu=1; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT0.3_CiteSeer; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--proportion 0.3 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./reform/${name}_log;
