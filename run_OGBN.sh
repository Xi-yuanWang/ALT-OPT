d="ogbn-arxiv"; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_arxiv; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d="ogbn-products"; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT_products; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;

d=cs; gpu=3; h=64; echo hidden ${h};
m=ALTOPT; prop=None; name=LALTOPT20_cs; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 3 --random_splits 10 \
--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 2>&1 | tee -a ./result/${name}_log;