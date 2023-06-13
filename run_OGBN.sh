d="ogbn-arxiv"; gpu=3; h=64; echo hidden ${h};
m=AGD; prop=None; name=AGD_arxiv; echo ${name}; 

CUDA_VISIBLE_DEVICES=0 \
nohup python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 10 \
 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --const_split 1 >> ./AGD/${name}_log 2>&1 &

sleep 10

 CUDA_VISIBLE_DEVICES=1 \
nohup python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 10 \
 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --const_split 1 >> ./AGD/${name}_log 2>&1 &

CUDA_VISIBLE_DEVICES=2 \
nohup python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 10 \
 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --const_split 1 >> ./AGD/${name}_log 2>&1 &


 CUDA_VISIBLE_DEVICES=3 \
nohup python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 10 \
 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
 --debug 1 --const_split 1 >> ./AGD/${name}_log 2>&1 &

#d="ogbn-products"; gpu=3; h=64; echo hidden ${h};
#m=ALTOPT; prop=None; name=LALTOPT_products; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
#python3 -u main_optuna.py --dataset ${d}  --model ${m}  --prop ${prop} --runs 10 \
#--log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
# --debug 1 2>&1 | tee -a ./newloss/${name}_log;

#d=cs; gpu=3; h=64; echo hidden ${h};
#m=ALTOPT; prop=None; name=LALTOPT20_cs; echo ${name}; CUDA_VISIBLE_DEVICES=${gpu} \
#python3 -u main_optuna.py --dataset ${d} --model ${m}  --prop ${prop} --runs 10 \
#--fix_num 20 --log_steps 100 --weight_decay  0.0005  --lr 0.01 --dropout 0.5 --loop 1 --K 10 --epoch 100 --alpha 0\
# --debug 1 2>&1 | tee -a ./newloss/${name}_log;