
d=CiteSeer; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_CiteSeer; echo ${name}; 

CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 --alpha 0.1 --loss MSE --lambda1 0.1 --lambda2 10.0 

CUDA_VISIBLE_DEVICES=${gpu} python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 --fix_num 5 --log_steps 10 --weight_decay  0.005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 500 --alpha 0 --debug 1 --useGCN 1 --alpha 0.1 --loss CE --lambda1 0.1 --lambda2 3.0 --weightedloss 0


 CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 --alpha 0.3 --loss MSE --lambda1 0.1 --lambda2 10.0 

 CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 --alpha 0.5 --loss MSE --lambda1 0.1 --lambda2 10.0 


 CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 --alpha 0.7 --loss MSE --lambda1 0.1 --lambda2 10.0 

 CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --useGCN 0 --alpha 0.9 --loss MSE --lambda1 0.1 --lambda2 10.0 