d=physics; gpu=0; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_physics; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 > ./reform/${name}_log 2>&1 &


d=computers; gpu=1; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_computers; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 > ./reform/${name}_log 2>&1 &

d=photo; gpu=2; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_photo; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 > ./reform/${name}_log 2>&1 &

d=cs; gpu=3; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD20_cs; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
nohup python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1  > ./reform/${name}_log 2>&1 &