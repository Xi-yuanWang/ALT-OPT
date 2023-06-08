
d=PubMed; gpu=2; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT5_PubMed; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 5 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=PubMed; gpu=2; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT10_PubMed; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 10 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=PubMed; gpu=2; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT20_PubMed; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0 \
 --debug 1 2>&1 | tee -a ./result/${name}_log;


d=PubMed; gpu=2; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT60_PubMed; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 60 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;


d=PubMed; gpu=2; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT0.3_PubMed; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--proportion 0.3 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;
