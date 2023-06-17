import torch
import random
import argparse
import time
from dataset import get_dataset
from util import Logger, str2bool
from get_model import get_model
from train_eval import train, test
from train_eval import train_altopt, test_altopt, train_cs, test_cs, test1, train_appnp, test_appnp
from model import CorrectAndSmooth

import optuna

def parse_args():
    parser = argparse.ArgumentParser(description='ALTOPT')
    parser.add_argument('--seed', type=int, default=12321312)
    parser.add_argument('--test', type=str2bool, default=False) # to do test or hyperparameter tuning
    parser.add_argument('--log_steps', type=int, default=0) # step to print log
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--gnnepoch', type=int, default=100) # pretraining epoch
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--normalize_features', type=str2bool, default=True, help="whether to normalize node feature")
    parser.add_argument('--random_splits', type=int, default=0, help='default use fix split')

    parser.add_argument('--model', type=str, default='ALTOPT')
    parser.add_argument('--num_layers', type=int, default=2) # number of mlp layers
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None) # weight decay for model parameters
    parser.add_argument('--Fwd', type=float, default=None) # Weight decay for F tensor
    parser.add_argument('--earlystop', type=int, default=10) 
    parser.add_argument('--lr', type=float, default=None)

    parser.add_argument('--bn', type=str2bool, default=False) # whether to use batchnorm
    parser.add_argument('--tailln', type=str2bool, default=False) # number of propagation
    parser.add_argument('--K0', type=int, default=None) # number of propagation initially
    parser.add_argument('--K', type=int, default=None) # number of propagation
    parser.add_argument('--lambda1', type=float, default=None) # weight of ||F-MLP(X)||^2
    parser.add_argument('--lambda2', type=float, default=None) # weight of ||Y_T-F_T||^2
    parser.add_argument('--alpha', type=float, default=None) # GNN hyperparameter
    
    parser.add_argument('--sort_key', type=str, default='K') # used in optuna

    parser.add_argument('--softmaxF', type=str2bool, default=True) # whether to do softmax for F
    parser.add_argument('--useGCN', type=str2bool, default=True) # whether to use GCN instead of MLP
    parser.add_argument("--weightedloss", type=str2bool, default=True) #whether to use loss weighted by node F

    parser.add_argument("--onlyy", type=str2bool, default=False) # used in exact method
    parser.add_argument("--temperature", type=float, default=0.2)
    
    parser.add_argument('--loss', type=str, default=None, choices=["CE", "MSE"]) # loss function
    parser.add_argument('--loop', type=int, default=None, help='Iteration number of MLP each epoch')
    parser.add_argument('--fix_num', type=int, default=0, help='number of train sample each class')
    parser.add_argument('--proportion', type=float, default=0, help='proportion of train sample each class')
    # used in other baseline model
    parser.add_argument('--num_correct_layer', type=int, default=None)
    parser.add_argument('--correct_alpha', type=float, default=None)
    parser.add_argument('--num_smooth_layer', type=int, default=None)
    parser.add_argument('--smooth_alpha', type=float, default=None)
    parser.add_argument('--pro_alpha', type=float, default=None)
    parser.add_argument('--const_split', type=str2bool, default=False)

    args = parser.parse_args()
    args.ogb = True if 'ogb' in args.dataset.lower() else False
    if args.K0 is None:
        args.K0 = args.K
    if args.ogb:
        args.num_layers = 3
        args.weight_decay = 0
        args.hidden_channels = 256
    return args

def objective(trial=None):
    '''
    training and test routine
    '''
    args = parse_args()
    if trial is not None:
        args = set_up_trial(trial, args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if args.random_splits > 0:
        random_split_num = args.random_splits
        print(f'random split {random_split_num} times and each for {args.runs} runs')
    else:
        random_split_num = 1
        print(f'fix split and run {args.runs} times')

    logger = Logger(args.runs * random_split_num)

    total_start = time.perf_counter()

    ## data split
    for split in range(random_split_num):
        dataset, data, split_idx = get_dataset(args, split, defense=args.defense)
        data.psuedo_indices = None
        all_features = data.num_features
        args.num_class = data.y.max()+1
        train_idx = split_idx['train']
        data = data.to(device)
        if not isinstance(data.adj_t, torch.Tensor):
            data.adj_t = data.adj_t.to_symmetric()

        start = time.time()

        # build model
        model = get_model(args, dataset, all_features)
        print(model)
        if args.model == 'LP':
            result = test(model, data, split_idx, args=args)
            logger.add_result(split, result)
            continue

        model.reset_parameters()
        # preprocess node features
        if args.model in ['IAPPNP', 'ORTGNN', 'ALTOPT', "AGD", "EXACT", 'APPNP', 'MLP', 'CS']:
            model.propagate(data, args.K0)

        for run in range(args.runs):
            data.pseudo_mask = data.train_mask.clone()
            data.pseudo_label = data.y.clone()

            data.total_weight = torch.zeros(data.y.shape[0]).cuda()
            data.f = None
            runs_overall = split * args.runs + run
            model.reset_parameters()

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            t_start = time.perf_counter()
            best_acc = 0
            y_soft = None
            nic_step = 0
            args.current_epoch = 0
            
            # training process
            if args.model in ['ALTOPT', "EXACT", "AGD"]:
                for i in range(args.gnnepoch):
                    loss = train(model, data, train_idx, optimizer, args=args)
                args.current_epoch = 1
            for epoch in range(1, 1 + args.epochs):
                args.current_epoch = epoch
                if args.model in ['ALTOPT', "EXACT", "AGD"]:
                    loss = 0
                    model.propagate_update(data, K=args.K)
                    for ii in range(args.loop):
                        loss = train_altopt(model, data, train_idx, optimizer, args=args)
                    result = test_altopt(model, data, split_idx, args=args)
                elif args.model == 'CS':
                    loss = train_cs(model, data, train_idx, optimizer, args=args)
                    train_acc, val_acc, test_acc, out = test_cs(model, data, split_idx, args=args)
                    if val_acc > best_acc:
                        best_acc = val_acc
                        y_soft = out.softmax(dim=-1)
                    result = [train_acc, val_acc, test_acc]
                elif args.model == 'APPNP':
                    loss = train_appnp(model, data, train_idx, optimizer, args=args)
                    result = test_appnp(model, data, split_idx, args=args)
                else:
                    loss = train(model, data, train_idx, optimizer, args=args)
                    result = test(model, data, split_idx, args=args)

                if args.model != 'CS':
                    logger.add_result(runs_overall, result)
                if args.log_steps > 0:
                    if epoch % args.log_steps == 0:
                        train_acc, valid_acc, test_acc = result
                        print(f'Split: {split + 1:02d}, '
                              f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_acc:.2f}%, '
                              f'Valid: {100 * valid_acc:.2f}% '
                              f'Test: {100 * test_acc:.2f}%')
                train_acc, valid_acc, test_acc = result
                nic_step += 1
                if best_acc <= valid_acc:
                    nic_step = 0
                    best_acc = valid_acc
                # print(nic_step, best_acc)
                if nic_step > args.earlystop:
                    break
            if args.model == 'CS':
                print('best_acc', best_acc)

                if y_soft is None:
                    y_soft = torch.load('y_soft2.th')
                test1(model, data, y_soft, split_idx, args=args)
                CS = CorrectAndSmooth(args)
                print('correct and smooth')
                y_soft = CS.correct(data=data, mlp=y_soft, edge_weight=None)
                y_soft = CS.smooth(data=data, y_soft=y_soft, edge_weight=None)
                print('Done')
                train_acc, val_acc, test_acc, out = test_cs(model, data, split_idx, out=y_soft, args=args)
                result = [train_acc, val_acc, test_acc]
                logger.add_result(runs_overall, result)
                print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            t_end = time.perf_counter()
            duration = t_end - t_start
            if args.log_steps > 0:
                print(print(f'Split: {split + 1:02d}, 'f'Run: {run + 1:02d}'), 'time: ', duration)
                logger.print_statistics(runs_overall)
    print('run time now:', time.time()-start)
    total_end = time.perf_counter()
    total_duration = total_end - total_start
    print('total time: ', total_duration)
    logger.print_statistics()
    train1_acc, valid_acc, train2_acc, test_acc, \
    train1_var, valid_var, train2_var, test_var = logger.best_result(run=None, with_var=True) # to adjust

    if trial is not None:
        trial.set_user_attr("train", train2_var)
        trial.set_user_attr("valid", valid_var)
        trial.set_user_attr("test", test_var)

    return valid_acc

def set_up_trial(trial: optuna.Trial, args):
    args.lr     = trial.suggest_float('lr', 1e-3, 3e-2, log=True)
    args.weight_decay     = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    args.dropout     = trial.suggest_float('dropout', 0, 0.95, step=0.05)
    args.loss = trial.suggest_categorical("loss", ["CE", "MSE"])
    if args.ogb:
        args.hidden_channels = trial.suggest_int("hidden_channels", 64, 256, step=64)
        args.num_layers = trial.suggest_int("num_layers", 1, 3)
    else:
        args.hidden_channels = trial.suggest_int("hidden_channels", 64, 64, step=16)
        args.num_layers = trial.suggest_int("num_layers", 1, 2)
    if args.model == 'LP':
        args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)
    elif args.model in ['APPNP', 'IAPPNP', 'MLP']:
        args.alpha     = trial.suggest_uniform('alpha', 0, 1.00001)
        args.pro_alpha = trial.suggest_uniform('pro_alpha', 0, 1.00001)
        args.K = trial.suggest_uniform('K', 0, 1000)

    elif args.model in ['ElasticGNN', 'ALTOPT', 'ORTGNN', "EXACT", "AGD"]:
        args.alpha = trial.suggest_float('alpha', 0, 1.00001, step=0.05)
        args.loop = trial.suggest_int('loop', 1, 1)
        args.K = trial.suggest_int("K", 1, 3)
        args.K0 = trial.suggest_int("K0", 1, 30)
        args.lambda1 = trial.suggest_float('lambda1', 0, 20, step=0.05)
        args.lambda2 = trial.suggest_float('lambda2', 0, 20, step=0.05)
        args.useGCN = trial.suggest_categorical("useGCN", [False])
        args.softmaxF = trial.suggest_categorical("softmaxF", [True, False])
        args.Fwd     = trial.suggest_float('Fwd', 1e-6, 1e-1, log=True)
        args.gnnepoch = trial.suggest_int("gnnepoch", 0, 100, step=10)
        args.weightedloss = trial.suggest_categorical("weightedloss", [False])
        args.temperature = trial.suggest_float("temperature", 0.01, 10, log=True)
        args.bn = trial.suggest_categorical("bn", [True, False])
        args.tailln = trial.suggest_categorical("tailln", [True, False])
    
    elif args.model in ['MFGNN', 'MFGNN-Hidden']:
        args.lambda1 = trial.suggest_uniform('lambda1', 0, 1000)
        args.K = trial.suggest_uniform('K', 0, 1000)
        print('lambda1: ', args.lambda1)

    elif args.model in ['CS']:
        args.num_correct_layer = trial.suggest_uniform('num_correct_layer', 0, 100)
        args.num_smooth_layer = trial.suggest_uniform('num_smooth_layer', 0, 100)
        args.correct_alpha = trial.suggest_uniform('correct_alpha', 0, 1.0001)
        args.smooth_alpha = trial.suggest_uniform('smooth_alpha', 0, 1.0001)
        args.alpha = trial.suggest_uniform('alpha', 0, 1.00001)
    print('K: ', args.K)
    print('alpha: ', args.alpha)
    print('lr: ', args.lr)
    print('weight_decay: ', args.weight_decay)
    print('dropout: ', args.dropout)
    return args


if __name__ == "__main__":
    optuna_total_start = time.perf_counter()

    args = parse_args()
    if args.test:
        objective()
    else:
        study = optuna.create_study(storage=f"sqlite:///AGD/{args.dataset}_{args.model}_{args.fix_num}_{args.proportion}.db", study_name=f"{args.dataset}_{args.model}", direction="maximize",  load_if_exists=True)

        study.optimize(objective, n_trials=1500)

        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best params:", study.best_params)
        print("Best trial Value: ", study.best_trial.value)
        print("Best trial Acc: ", study.best_trial.user_attrs)

        optuna_total_end = time.perf_counter()
        optuna_total_duration = optuna_total_end - optuna_total_start
        print('optuna total time: ', optuna_total_duration)

